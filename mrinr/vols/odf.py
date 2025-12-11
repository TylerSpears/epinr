# -*- coding: utf-8 -*-

# Functions and objects that operate on orientation density functions (ODFs).

import collections
import functools
import shlex
import tempfile
from pathlib import Path
from typing import Optional

import dipy
import dipy.reconst.csdeconv
import dipy.reconst.shm
import einops
import nibabel as nib
import numpy as np
import torch

import mrinr
from mrinr._lazy_loader import LazyLoader

# Import MRTrix3 spherical harmonic basis functions into this module.
from ._mrtrix3_spharm import (  # noqa: F401
    norm_legendre_poly_cos_theta,
    spharm,
    spharm_basis_mrtrix3,
)

jax = LazyLoader("jax", globals(), "jax")
jnp = LazyLoader("jnp", globals(), "jax.numpy")

_SHOrderDegreeResult = collections.namedtuple(
    "_SHOrderDegreeResult", ("sh", "order", "degree")
)


def sample_sphere_coords(
    odf_coeffs: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    sh_order: int,
    sh_order_dim=1,
    mask: torch.Tensor = None,
    force_nonnegative: bool = True,
) -> torch.Tensor:
    """Sample a spherical function on the sphere with coefficients in the SH domain.

    Parameters
    ----------
    odf_coeffs : torch.Tensor
        Spherical function coefficients in the SH domain that define the fn to sample.
    theta : torch.Tensor
        A flattened Tensor of spherical polar coordinate theta in range [0, $\pi$].
    phi : torch.Tensor
        A flattened Tensor of spherical azimuth coordinate phi in range ($-\pi$, $\pi$].
    sh_order : int
        Even-valued spherical harmonics order, must match number of orders in odf_coeffs
    mask : torch.Tensor, optional
        Mask Tensor for masking "voxels" in the odf coeffs, by default None.

        Shape is assumed to be b x c x space_dim_0 [x space_dim_1 x ...], with c=1.

    Returns
    -------
    torch.Tensor
    """
    orig_spatial_shape = list(odf_coeffs.shape)
    orig_spatial_shape.pop(sh_order_dim)
    orig_spatial_shape = tuple(orig_spatial_shape)
    if odf_coeffs.ndim < 5:
        if odf_coeffs.ndim == 1:
            odf_coeffs = odf_coeffs[None]
        if odf_coeffs.ndim == 2:
            odf_coeffs = odf_coeffs[..., None, None, None]

    fn_coeffs = odf_coeffs.movedim(sh_order_dim, -1)
    fn_coeffs = fn_coeffs.reshape(-1, fn_coeffs.shape[-1])
    if mask is not None:
        fn_mask = mask.squeeze(sh_order_dim)
        # fn_mask = mask.movedim(1, -1)[..., 0]
        fn_mask = fn_mask.broadcast_to(orig_spatial_shape).reshape(-1)
        fn_coeffs = fn_coeffs[fn_mask]
    else:
        fn_mask = None
    azimuth = phi + torch.pi
    sh_transform, _, _ = get_torch_dipy_sh_transform(
        sh_order=sh_order, polar_coord=theta, azimuth_coord=azimuth
    )
    # Expand to have batch dim of 1.
    sh_transform = sh_transform.T[None]

    if fn_mask is not None:
        fn_samples = torch.zeros(
            fn_mask.shape[0],
            sh_transform.shape[2],
            dtype=sh_transform.dtype,
            device=sh_transform.device,
        )
        fn_samples[fn_mask] = torch.matmul(fn_coeffs, sh_transform)
    else:
        fn_samples = torch.matmul(fn_coeffs, sh_transform)

    if force_nonnegative:
        # Enforce non-negativity constraint.
        fn_samples.clamp_min_(0)
    fn_samples = fn_samples.reshape(*orig_spatial_shape, -1)
    fn_samples = fn_samples.movedim(-1, sh_order_dim)

    return fn_samples.contiguous()


def _quick_sphere_sample(fodf_coeffs: torch.Tensor, dipy_sphere=None):
    if dipy_sphere is None:
        dipy_sphere = dipy.data.get_sphere(name="repulsion724")
    fodf_coeffs = fodf_coeffs.flatten()[None]
    l_max = mrinr.vols._mrtrix3_spharm.max_sh_len_to_lmax(fodf_coeffs.numel())
    # assert fodf_coeffs.numel() == 45
    theta = torch.from_numpy(dipy_sphere.theta).to(fodf_coeffs)
    phi = torch.from_numpy(dipy_sphere.phi).to(fodf_coeffs)
    s = sample_sphere_coords(
        fodf_coeffs,
        theta=theta,
        phi=phi,
        sh_order=l_max,
        sh_order_dim=1,
        force_nonnegative=True,
    )

    return s, theta, phi


@torch.no_grad()
def odf_peaks_mrtrix(
    odf: torch.Tensor,
    affine_vox2real: torch.Tensor,
    mask: torch.Tensor,
    n_peaks: int,
    min_amp: float,
    match_peaks_vol: Optional[torch.Tensor] = None,
    mrtrix_nthreads: Optional[int] = None,
) -> tuple[torch.Tensor, nib.spatialimages.DataobjImage]:
    batch_size = odf.shape[0]
    if batch_size != 1:
        raise NotImplementedError("ERROR: Batch size != 1 not implemented")

    n_peak_channels = n_peaks * 3

    affine = affine_vox2real.squeeze(0).detach().cpu().numpy()
    odf_im = nib.Nifti1Image(
        einops.rearrange(
            odf.detach().cpu().squeeze(0).numpy(), "coeff x y z -> x y z coeff"
        ),
        affine=affine,
    )
    mask_im = nib.Nifti1Image(
        mask.detach().cpu().squeeze(0).squeeze(0).numpy().astype(np.uint8),
        affine=affine,
    )
    if match_peaks_vol is not None:
        match_vol = (
            match_peaks_vol.detach().cpu().squeeze(0).numpy()[..., :n_peak_channels]
        )
        match_im = nib.Nifti1Image(
            match_vol,
            affine=affine,
        )

    with tempfile.TemporaryDirectory(prefix="_odf_peaks_mrtrix_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        mask_f = tmp_dir / "mask.nii"
        nib.save(mask_im, mask_f)
        src_odf_f = tmp_dir / "src_odf.nii"
        nib.save(odf_im, src_odf_f)
        if match_peaks_vol is not None:
            match_peaks_f = tmp_dir / "match_peaks.nii"
            nib.save(match_im, match_peaks_f)
        out_peaks_f = tmp_dir / "odf_peaks.nii"
        cmd = f"sh2peaks -quiet -num {n_peaks} -threshold {min_amp} -mask {mask_f}"
        if mrtrix_nthreads is not None:
            cmd = " ".join([cmd, "-nthreads", str(mrtrix_nthreads)])
        if match_peaks_vol is not None:
            cmd = " ".join([cmd, "-peaks", str(match_peaks_f)])
        cmd = " ".join([cmd, str(src_odf_f), str(out_peaks_f)])

        call_result = mrinr.sh_cmds.call_shell_exec(
            cmd=cmd, args="", cwd=tmp_dir, popen_args_override=shlex.split(cmd)
        )
        assert out_peaks_f.exists()
        out_peaks_im = nib.load(out_peaks_f)
        # Make a copy of the image data, as the temporary directory will be deleted
        # before exiting the function.
        out_peaks_im = nib.Nifti1Image(
            out_peaks_im.get_fdata().astype(odf_im.get_data_dtype()),
            affine=out_peaks_im.affine,
        )

    out_peaks = out_peaks_im.get_fdata()
    out_peaks = torch.from_numpy(out_peaks).unsqueeze(0).to(odf)
    # Nans are either outside of the mask or in CSF locations with no peak, so just
    # assign a 0 vector as the peak direction.
    out_peaks.nan_to_num_(nan=0.0)

    return out_peaks, out_peaks_im


def gfa(sh_coeffs: torch.Tensor) -> torch.Tensor:
    # Generalized fractional anisotropy.
    # See Equation 2 in
    # Cohen-Adad, et. al., "Detection of multiple pathways in the spinal cord using q-ball imaging,"

    num = sh_coeffs[..., 0] ** 2
    denom = (sh_coeffs**2).sum(-1)
    empty_odfs = torch.isclose(denom, denom.new_zeros(1))
    denom[empty_odfs] = 1.0
    gen_fa = torch.sqrt(1 - (num / denom))
    gen_fa[empty_odfs] = 0.0

    return gen_fa


@functools.lru_cache(maxsize=10)
def get_torch_dipy_sh_transform(
    sh_order: int, polar_coord: torch.Tensor, azimuth_coord: torch.Tensor
) -> _SHOrderDegreeResult:
    polar = polar_coord.detach().flatten().cpu().numpy()
    azimuth = azimuth_coord.detach().flatten().cpu().numpy()
    # The azimuth in dipy/scipy is set to be between (0, 2*pi], rather than (-pi, pi],
    # so the azimuth coordinate must be set to that compatible range.
    assert azimuth.min() >= mrinr.coords.MIN_PHI
    assert azimuth.max() <= mrinr.coords.MAX_PHI
    # The Tournier basis is what Mrtrix uses, so we'll default to that for now.
    # <https://mrtrix.readthedocs.io/en/3.0.4/concepts/spherical_harmonics.html#formulation-used-in-mrtrix3>
    # <https://github.com/dipy/dipy/blob/13af40fec09fb23a3692cb0bfdcb91d08acfd766/dipy/reconst/shm.py#L363>
    sh, order, degree = dipy.reconst.shm.real_sh_tournier(
        sh_order=sh_order, theta=polar, phi=azimuth, full_basis=False, legacy=False
    )
    sh = torch.from_numpy(sh).to(polar_coord)
    order = torch.from_numpy(order).to(polar_coord)
    degree = torch.from_numpy(degree).to(polar_coord)

    return _SHOrderDegreeResult(sh, order, degree)


def thresh_odf_samples_by_pdf(
    sphere_samples: torch.Tensor, pdf_thresh_min: float
) -> torch.Tensor:
    s_pdf = sphere_samples - sphere_samples.min(1, keepdim=True).values
    s_pdf = s_pdf / s_pdf.sum(1, keepdim=True)
    s = torch.where(s_pdf < pdf_thresh_min, 0, sphere_samples)

    return s


def thresh_odf_samples_by_value(
    sphere_samples: torch.Tensor, value_min: float
) -> torch.Tensor:
    s = torch.where(sphere_samples < value_min, 0, sphere_samples)
    return s


def thresh_odf_samples_by_quantile(
    sphere_samples: torch.Tensor, q_low: float
) -> torch.Tensor:
    q_low_val = torch.nanquantile(sphere_samples, q=q_low, dim=1, keepdim=True)
    s = torch.where(sphere_samples < q_low_val, 0, sphere_samples)

    return s


@functools.lru_cache(maxsize=10)
def _get_degree_order_vecs(l_max: int, batch_size: int, device="cpu"):
    unique_l = np.arange(0, l_max + 2, 2).astype(int)

    l_degrees = list()
    m_orders = list()
    for l in unique_l:
        for m in np.arange(-l, l + 1):
            l_degrees.append(l)
            m_orders.append(m)
    l_degrees = einops.repeat(
        torch.Tensor(l_degrees).to(torch.int32), "d -> b d", b=batch_size
    )
    m_orders = einops.repeat(
        torch.Tensor(m_orders).to(torch.int32), "o -> b o", b=batch_size
    )
    return l_degrees.to(device), m_orders.to(device)


def _broad_sh_mask(l_all: torch.Tensor, m_all: torch.Tensor, l=None, m=None):
    # Assume that for l_all and m_all, the first N-1 dimensions are "batch" dimensions
    # that store "batch" number of repeats of the same array, with unique values in dim
    # -1.
    l_all = l_all.view(-1, l_all.shape[-1])[0]
    m_all = m_all.view(-1, m_all.shape[-1])[0]

    if l is None:
        l = torch.unique(l_all)
    elif np.isscalar(l):
        l = torch.as_tensor([l]).reshape(-1)
    elif torch.is_tensor(l):
        l = l.reshape(-1)
    else:
        # General iterable
        l = torch.as_tensor(list(l)).reshape(-1)
    l = l.to(l_all)
    if m is None:
        m = torch.unique(m_all)
    elif np.isscalar(m):
        m = torch.as_tensor([m]).reshape(-1)
    elif torch.is_tensor(m):
        m = m.reshape(-1)
    else:
        # General iterable
        m = torch.as_tensor(list(m)).reshape(-1)
    m = m.to(m_all)

    return torch.isin(l_all, l) & torch.isin(m_all, m)


def _sh_idx(l: torch.Tensor, m: torch.Tensor, invalid_idx_replace=None):
    # Assume l and m are copied across their respective n-1 first dimensions.
    l_ = l.view(-1, l.shape[-1])[0]
    m_ = m.view(-1, m.shape[-1])[0]
    idx = ((l_ // 2) * (l_ + 1) + m_).to(torch.long)

    if invalid_idx_replace is not None:
        l_max = l.max().cpu().item()
        max_idx = (((l_max + 1) * (l_max + 2)) // 2) - 1
        idx.masked_fill_(idx > max_idx, int(invalid_idx_replace))
    return idx


# def __jax_spherical_harmonic(
#     degree: "jax.Array",
#     order: "jax.Array",
#     theta: "jax.Array",
#     phi: "jax.Array",
#     l_max: int,
# ) -> "jax.Array":
#     # Scipy/JAX use the opposite convention for theta and phi as used throughout the
#     # rest of this module, so swap them for this function.
#     scipy_theta = phi
#     scipy_phi = theta
#     return jax.scipy.special.sph_harm(
#         m=order, n=degree, theta=scipy_theta, phi=scipy_phi, n_max=l_max
#     )


# def __jax_norm_legendre_poly(
#     degree: "jax.Array",
#     order: "jax.Array",
#     theta: "jax.Array",
#     l_max: int,
# ) -> "jax.Array":
#     # We don't care about phi, so long as it doesn't 0-out the sh value.
#     dummy_phi = jnp.zeros_like(theta)
#     sh = __jax_spherical_harmonic(
#         degree=degree, order=order, theta=theta, phi=dummy_phi, l_max=l_max
#     )
#     exp_theta = jnp.exp(1j * order * dummy_phi)
#     # Remove the e^(i m phi) to just get N_lm * Y_lm(cos theta)
#     # Only return real component, imaginary component should be 0 anyway.
#     return (sh / exp_theta).real


# def _norm_legendre_poly(
#     degree: torch.Tensor,
#     order: torch.Tensor,
#     theta: torch.Tensor,
# ) -> torch.Tensor:
#     theta = theta.squeeze(-1)
#     degree = degree.squeeze(0)
#     order = order.squeeze(0)
#     l_max = int(degree.max().cpu().item())

#     assert tuple(degree.shape) == tuple(order.shape)
#     if (theta.ndim > 2) or (degree.ndim > 2):
#         raise RuntimeError(
#             "ERROR: Will only accept 1D or 2D shapes, "
#             + f"got {tuple(theta.shape)}, {tuple(degree.shape)}"
#         )
#     elif (theta.ndim == 2) and (degree.ndim == 2):
#         assert tuple(theta.shape) == tuple(degree.shape)
#     elif theta.ndim == 2:
#         target_shape = tuple(theta.shape)
#     elif degree.ndim == 2:
#         target_shape = tuple(degree.shape)
#     else:
#         target_shape = (theta.shape[0], degree.shape[0])
#     batch_size_angle = target_shape[0]
#     batch_size_lm = target_shape[1]
#     l_max = int(degree.max().cpu().item())

#     if tuple(degree.shape) != target_shape:
#         broad_degree = einops.repeat(
#             degree, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
#         )
#         broad_order = einops.repeat(
#             order, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
#         )
#     else:
#         broad_degree = degree
#         broad_order = order
#     if tuple(theta.shape) != target_shape:
#         broad_theta = einops.repeat(
#             theta, "b_angle -> b_angle b_lm", b_lm=batch_size_lm
#         )
#     else:
#         broad_theta = theta

#     broad_degree = einops.rearrange(broad_degree, "b_angle b_lm -> (b_angle b_lm)")
#     broad_order = einops.rearrange(broad_order, "b_angle b_lm -> (b_angle b_lm)")
#     broad_theta = einops.rearrange(broad_theta, "b_angle b_lm -> (b_angle b_lm)")

#     jax_norm_P_lm = __jax_norm_legendre_poly(
#         degree=mrinr.utils.t2j(broad_degree),
#         order=mrinr.utils.t2j(broad_order),
#         theta=mrinr.utils.t2j(broad_theta),
#         l_max=l_max,
#     )
#     norm_P_lm = einops.rearrange(
#         mrinr.utils.j2t(jax_norm_P_lm),
#         "(b_angle b_lm) -> b_angle b_lm",
#         b_angle=batch_size_angle,
#     )

#     return norm_P_lm


# def _sph_harm(
#     degree: torch.Tensor,
#     order: torch.Tensor,
#     theta: torch.Tensor,
#     phi: torch.Tensor,
# ) -> torch.Tensor:
#     theta = theta.squeeze(-1)
#     phi = phi.squeeze(-1)
#     degree = degree.squeeze(0)
#     order = order.squeeze(0)

#     assert tuple(degree.shape) == tuple(order.shape)
#     assert tuple(theta.shape) == tuple(phi.shape)
#     if (theta.ndim > 2) or (degree.ndim > 2):
#         raise RuntimeError(
#             "ERROR: Will only accept 1D or 2D shapes, "
#             + f"got {tuple(theta.shape)}, {tuple(degree.shape)}"
#         )
#     elif (theta.ndim == 2) and (degree.ndim == 2):
#         assert tuple(theta.shape) == tuple(degree.shape)
#     elif theta.ndim == 2:
#         target_shape = tuple(theta.shape)
#     elif degree.ndim == 2:
#         target_shape = tuple(degree.shape)
#     else:
#         target_shape = (theta.shape[0], degree.shape[0])
#     batch_size_angle = target_shape[0]
#     batch_size_lm = target_shape[1]
#     l_max = int(degree.max().cpu().item())

#     if tuple(degree.shape) != target_shape:
#         broad_degree = einops.repeat(
#             degree, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
#         )
#         broad_order = einops.repeat(
#             order, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
#         )
#     else:
#         broad_degree = degree
#         broad_order = order
#     if tuple(theta.shape) != target_shape:
#         broad_theta = einops.repeat(
#             theta, "b_angle -> b_angle b_lm", b_lm=batch_size_lm
#         )
#         broad_phi = einops.repeat(phi, "b_angle -> b_angle b_lm", b_lm=batch_size_lm)
#     else:
#         broad_theta = theta
#         broad_phi = phi

#     broad_degree = einops.rearrange(broad_degree, "b_angle b_lm -> (b_angle b_lm)")
#     broad_order = einops.rearrange(broad_order, "b_angle b_lm -> (b_angle b_lm)")
#     broad_theta = einops.rearrange(broad_theta, "b_angle b_lm -> (b_angle b_lm)")
#     broad_phi = einops.rearrange(broad_phi, "b_angle b_lm -> (b_angle b_lm)")

#     jax_sph_harm = __jax_spherical_harmonic(
#         degree=mrinr.utils.t2j(broad_degree),
#         order=mrinr.utils.t2j(broad_order),
#         theta=mrinr.utils.t2j(broad_theta),
#         phi=mrinr.utils.t2j(broad_phi),
#         l_max=l_max,
#     )
#     sph_harm_vals = einops.rearrange(
#         mrinr.utils.j2t(jax_sph_harm),
#         "(b_angle b_lm) -> b_angle b_lm",
#         b_angle=batch_size_angle,
#     )

#     return sph_harm_vals


# def sh_basis_mrtrix3(
#     theta: torch.Tensor,
#     phi: torch.Tensor,
#     degree: torch.Tensor,
#     order: torch.Tensor,
# ) -> torch.Tensor:
#     Y_m_abs_l = _sph_harm(order=torch.abs(order), degree=degree, theta=theta, phi=phi)
#     Y_m_abs_l = torch.where(order < 0, np.sqrt(2) * Y_m_abs_l.imag, Y_m_abs_l)
#     Y_m_abs_l = torch.where(order > 0, np.sqrt(2) * Y_m_abs_l.real, Y_m_abs_l)

#     return Y_m_abs_l.real
