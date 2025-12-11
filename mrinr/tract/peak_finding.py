# -*- coding: utf-8 -*-
# ruff: noqa: F722
import math
from functools import partial
from typing import Union

import einops
import numpy as np
import torch
from jaxtyping import Bool, Float, Int, Shaped

import mrinr


def __get_mrtrix_lm_from_sh_idx(
    x: Shaped[torch.Tensor, " sh_lm"],
) -> tuple[Int[torch.Tensor, " sh_lm"], Int[torch.Tensor, " sh_lm"]]:
    """Calculate degree and order vectors from the length of a spharm-ordered vector.

    Assumes the MRTrix3 convention of spherical harmonics.

    Parameters
    ----------
    x: Tensor, shape (sh_lm,)
        Vector ordered by l and m, as per MRtrix3's convention.

    Returns
    -------
    tuple of Tensors, shape (sh_lm,) and (sh_lm,)
        Degree and order vectors, respectively.
    """
    # Get l (i.e. degree) from the index with a hacked up formula.
    # idx = x.new_tensor(torch.arange(x.shape[0]), dtype=torch.float)
    idx = torch.arange(x.shape[0], device=x.device, dtype=torch.float)
    # Formula for l_max given size of SH vector.
    l = (-3 + torch.sqrt(1 + 8 * (idx + 1))) / 2
    # Some numerical gymnastics to get to l.
    l = (torch.ceil(((l / 2) - 1e-4)) * 2).round().abs().to(torch.int32)
    # Now find m (i.e. order) from the index and l.
    m = -((l * (l + 1)) // 2) + idx.to(l)

    return l, m


def __get_scalar_with_mrtrix_sh_idx(
    x: Shaped[torch.Tensor, " sh_lm"],
    l: Int[torch.Tensor, "1"],
    m: Int[torch.Tensor, "1"],
) -> Shaped[torch.Tensor, "1"]:
    """Samples a spharm-ordered vector at the given degree and order.

    Assumes the MRTrix3 ordering of spherical harmonics. Compatible with 'torch.vmap'
    and 'torch.compile' without using explicit integer indexing by using masking with
    the given l and m.

    Parameters
    ----------
    x: Tensor, shape (sh_lm,)
        Vector ordered by l and m, as per MRtrix3's convention.
    l: Tensor, shape (1,)
        Degree of the spherical harmonic.
    m: Tensor, shape (1,)
        Order of the spherical harmonic.

    Returns
    -------
    Tensor, shape (1,)
        Value of the vector at the given l and m indices.
    """
    l_vec, m_vec = __get_mrtrix_lm_from_sh_idx(x)
    # Zero out all elements that do not match the given l and m.
    y = x * ((l_vec == l) & (m_vec == m))
    y = y.sum()
    return y


def __scalar_sh_grad_k(
    l: Int[torch.Tensor, "1"],
    m: Int[torch.Tensor, "1"],
    norm_P_l_m: Float[torch.Tensor, " sh_lm"],
) -> Float[torch.Tensor, "1"]:
    select_lm = lambda x, l_, m_: __get_scalar_with_mrtrix_sh_idx(x, l_, m_)
    k = torch.where(
        m > 0,
        torch.sqrt((l + m) * (l - m + 1)) * select_lm(norm_P_l_m, l, m - 1),
        norm_P_l_m.new_zeros(l.shape),
    )
    k = k - torch.where(
        (m > 0) & (l > m),
        torch.sqrt((l - m) * (l + m + 1)) * select_lm(norm_P_l_m, l, m + 1),
        norm_P_l_m.new_zeros(l.shape),
    )
    k = k * -0.5
    return k


def _k(
    l: Int[torch.Tensor, "b spharm_lm"],
    m: Int[torch.Tensor, "b spharm_lm"],
    norm_P_l_m: Float[torch.Tensor, " sh_lm"],
) -> Float[torch.Tensor, "b spharm_lm"]:
    fn = torch.vmap(
        torch.vmap(
            __scalar_sh_grad_k,
            in_dims=(0, 0, None),
            out_dims=0,
        ),
        in_dims=0,
        out_dims=0,
        randomness="error",
    )
    return fn(l, m, norm_P_l_m)


def _norm_legendre_poly_cos_theta(
    l: Int[torch.Tensor, "b spharm_lm"],
    m: Int[torch.Tensor, "b spharm_lm"],
    elevation: Float[torch.Tensor, "b spharm_lm"],
) -> Float[torch.Tensor, "b spharm_lm"]:
    fn = torch.vmap(
        mrinr.vols._mrtrix3_spharm.__scalar_torch_norm_legendre_poly_cos_theta,
        in_dims=0,
        out_dims=0,
        randomness="error",
    )
    l_ = einops.rearrange(l, "b spharm_lm -> (b spharm_lm)")
    m_ = einops.rearrange(m, "b spharm_lm -> (b spharm_lm)")
    elevation_ = einops.rearrange(elevation, "b spharm_lm -> (b spharm_lm)")
    NP_lm_ = fn(l_, m_, elevation_)
    NP_lm = einops.rearrange(
        NP_lm_, "(b spharm_lm) -> b spharm_lm", b=l.shape[0], spharm_lm=l.shape[1]
    )
    return NP_lm


def __scalar_ds_delev(
    azimuth: Float[torch.Tensor, "n_polar=1"],
    l: Int[torch.Tensor, "n_spharm=1"],
    m: Int[torch.Tensor, "n_spharm=1"],
    norm_P_l_m: Float[torch.Tensor, " sh_lm_total"],
    k: Float[torch.Tensor, " sh_lm_total"],
    spharm_coeffs: Float[torch.Tensor, " sh_lm_total"],
) -> Float[torch.Tensor, "1"]:
    # Convenience function for indexing into vectors by l and m.
    select_lm = lambda x, l_, m_: __get_scalar_with_mrtrix_sh_idx(x, l_, m_)
    zonal = torch.where(
        (l > 0) & (m == 0),
        torch.sqrt(l * (l + 1))
        * select_lm(norm_P_l_m, l, m + 1)  # m+1 exists for all l > 0
        * select_lm(spharm_coeffs, l, m),
        azimuth.new_zeros(azimuth.shape),
    )

    sqrt_2 = math.sqrt(2)
    non_zonal = torch.where(
        m > 0,
        select_lm(k, l, m)
        * (
            (sqrt_2 * torch.cos(m * azimuth)) * select_lm(spharm_coeffs, l, m)
            + (sqrt_2 * torch.sin(m * azimuth)) * select_lm(spharm_coeffs, l, -m)
        ),
        azimuth.new_zeros(azimuth.shape),
    )

    ds_delev = zonal + non_zonal
    return ds_delev


def _ds_delev(
    azimuth: Float[torch.Tensor, "b spharm_lm"],
    l: Int[torch.Tensor, "b spharm_lm"],
    m: Int[torch.Tensor, "b spharm_lm"],
    norm_P_l_m: Float[torch.Tensor, "b spharm_lm"],
    k: Float[torch.Tensor, "b spharm_lm"],
    spharm_coeffs: Float[torch.Tensor, "b spharm_lm"],
) -> Float[torch.Tensor, " b"]:
    fn = torch.vmap(
        torch.vmap(__scalar_ds_delev, in_dims=(0, 0, 0, None, None, None), out_dims=0),
        in_dims=0,
        out_dims=0,
        randomness="error",
    )

    ds_delev_lm = fn(azimuth, l, m, norm_P_l_m, k, spharm_coeffs)
    ds_delev = ds_delev_lm.sum(-1)
    return ds_delev


def __scalar_ds_dazim(
    azimuth: Float[torch.Tensor, "n_polar=1"],
    elevation: Float[torch.Tensor, "n_polar=1"],
    at_pole: Bool[torch.Tensor, "n_polar=1"],
    l: Int[torch.Tensor, "n_spharm=1"],
    m: Int[torch.Tensor, "n_spharm=1"],
    norm_P_l_m: Float[torch.Tensor, " sh_lm_total"],
    k: Float[torch.Tensor, " sh_lm_total"],
    spharm_coeffs: Float[torch.Tensor, " sh_lm_total"],
) -> Float[torch.Tensor, "n_spharm=1"]:
    select_lm = lambda x, l_, m_: __get_scalar_with_mrtrix_sh_idx(x, l_, m_)
    cos_azim = math.sqrt(2) * torch.cos(m * azimuth)
    sin_azim = math.sqrt(2) * torch.sin(m * azimuth)

    ds_dazim = torch.where(
        m <= 0,
        cos_azim.new_zeros(cos_azim.shape),
        torch.where(
            at_pole,
            select_lm(k, l, m)
            * (
                cos_azim * select_lm(spharm_coeffs, l, -m)
                - sin_azim * select_lm(spharm_coeffs, l, m)
            ),
            m
            * select_lm(norm_P_l_m, l, m)
            * (
                cos_azim * select_lm(spharm_coeffs, l, -m)
                - sin_azim * select_lm(spharm_coeffs, l, m)
            ),
        ),
    )
    return ds_dazim


def _ds_dazim(
    azimuth: Float[torch.Tensor, "b spharm_lm"],
    elevation: Float[torch.Tensor, "b spharm_lm"],
    at_pole: Bool[torch.Tensor, "b spharm_lm"],
    l: Int[torch.Tensor, "b spharm_lm"],
    m: Int[torch.Tensor, "b spharm_lm"],
    norm_P_l_m: Float[torch.Tensor, "b spharm_lm"],
    k: Float[torch.Tensor, "b spharm_lm"],
    spharm_coeffs: Float[torch.Tensor, "b spharm_lm"],
) -> Float[torch.Tensor, " b"]:
    # Vectorize over the degree-order coefficients, then over the batch/polar axis.
    fn = torch.vmap(
        torch.vmap(
            __scalar_ds_dazim,
            in_dims=(0, 0, 0, 0, 0, None, None, None),
            out_dims=0,
        ),
        in_dims=0,
        out_dims=0,
        randomness="error",
    )

    ds_dazim_lm = fn(
        azimuth,
        elevation,
        at_pole,
        l,
        m,
        norm_P_l_m,
        k,
        spharm_coeffs,
    )
    ds_dazim = ds_dazim_lm.sum(-1)

    return ds_dazim


def _batch_sh_grad(
    theta: Float[torch.Tensor, " batch"],
    phi: Float[torch.Tensor, " batch"],
    l: Int[torch.Tensor, " sh_lm"],
    m: Int[torch.Tensor, " sh_lm"],
    sh_coeffs: Float[torch.Tensor, "batch sh_lm"],
    pole_angle_tol: float = 1e-5,
) -> Float[torch.Tensor, "batch delev_dazim=2"]:
    """Gradient of spherical harmonics coefficients with respect to elevation and azimuth.

    Computes the gradient of 'batch_size' spherical functions, as represented by
    spherical harmonics, with respect to the elevation and azimuth polar angles. This
    function uses functions that are vmapped over batches and spherical harmonic
    indices, and is compatible with `torch.compile` under a single computational graph.
    The maximum supported spharm degree is determined by the optimized associated
    Legendre polynomial function implementation, which is currently set to 10.

    Parameters
    ----------
    theta: Tensor, shape (batch_size,)
        Elevation angles in radians, broadcasted over all spherical harmonic orders/degrees.
    phi: Tensor, shape (batch_size,)
        Azimuth angles in radians, repeated broadcasted over all spherical harmonic orders/degrees.
    l: Tensor, shape (spharm_orders_degrees,)
        Degree vector of spherical harmonics coefficients, broadcasted across all batches.
    m: Tensor, shape (spharm_orders_degrees,)
        Order vector of spherical harmonics coefficients, broadcasted across all batches.
    sh_coeffs: Tensor, shape (batch_size, sh_lm)
        Spherical harmonic coefficients.
    pole_angle_tol: float
        Tolerance for considering an angle to be at the pole, by default 1e-5.

    Returns
    -------
    Tensor, shape (batch_size, elevation_azimuth=2)
        A tensor containing the gradients with respect to elevation and azimuth angles.
    """
    cart_coords = mrinr.coords.unit_sphere2xyz(theta=theta, phi=phi)
    x = cart_coords[..., 0]
    y = cart_coords[..., 1]
    # Convert to elevation and azimuth angles.
    elev_mrtrix = theta
    azim_mrtrix = torch.arctan2(y, x)
    at_pole = torch.sin(elev_mrtrix) < pole_angle_tol
    # Duplicate polar angles over the order-degree coefficients.
    batch_size = theta.shape[0]
    len_lm = l.shape[0]
    # Store elevation and at_pole mask with shape [batch_size,], no repeating.
    no_repeat_elev_mrtrix = elev_mrtrix.clone()
    no_repeat_at_pole = at_pole.clone()
    elev_mrtrix = einops.repeat(elev_mrtrix, "b -> b sh_lm", sh_lm=len_lm)
    azim_mrtrix = einops.repeat(azim_mrtrix, "b -> b sh_lm", sh_lm=len_lm)
    at_pole = einops.repeat(at_pole, "b -> b sh_lm", sh_lm=len_lm)
    # Duplicate l and m over the batch/polar axis.
    l = einops.repeat(l, "sh_lm -> b sh_lm", b=batch_size, sh_lm=len_lm)
    m = einops.repeat(m, "sh_lm -> b sh_lm", b=batch_size, sh_lm=len_lm)

    # Pre-compute vectors used in each derivative.
    norm_P_l_m = _norm_legendre_poly_cos_theta(l=l, m=m, elevation=elev_mrtrix)
    k = _k(l=l, m=m, norm_P_l_m=norm_P_l_m)

    # Gradient of elevation, shape '(batch_size,)'.
    ds_delev = _ds_delev(
        azimuth=azim_mrtrix,
        l=l,
        m=m,
        norm_P_l_m=norm_P_l_m,
        k=k,
        spharm_coeffs=sh_coeffs,
    )
    # Derivative of azimuth, shape '(batch_size,)'.
    ds_dazim = _ds_dazim(
        azimuth=azim_mrtrix,
        elevation=elev_mrtrix,
        at_pole=at_pole,
        l=l,
        m=m,
        norm_P_l_m=norm_P_l_m,
        k=k,
        spharm_coeffs=sh_coeffs,
    )
    # Scale ds/d azimuth to make this derivative into a gradient.
    # <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates>
    ds_dazim = ds_dazim / torch.where(
        ~no_repeat_at_pole, torch.sin(no_repeat_elev_mrtrix), 1.0
    )

    # Add a dimension back to the derivatives to make them a 1D vector of length 2.
    jac = torch.stack([ds_delev, ds_dazim], dim=-1)
    return jac


def _subset_mask_at_decreasing_thresh(
    real_mask: torch.Tensor,
    chunk_mask: torch.Tensor,
    thresholds: np.ndarray,
):
    if thresholds.shape[0] == 0:
        r = chunk_mask, thresholds
    else:
        real_m_sum = real_mask.sum().item()
        # Negate the batch sizes to use searchsorted.
        search_idx = np.searchsorted(-thresholds, -real_m_sum, side="left")
        if search_idx > 0:
            new_thresh = thresholds[search_idx - 1]
            # print(f"Below new threshold {new_thresh}, creating new chunk mask.")
            popped_thresholds = thresholds[search_idx:]
            new_chunk_mask = real_mask.clone()
            n_to_fill = int(new_thresh - real_m_sum)
            # print("Num elements to fill beyond true mask: ", n_to_fill)
            chunk_mask_to_fill_idx = torch.nonzero(~real_mask)[:n_to_fill, 0]
            new_chunk_mask[(chunk_mask_to_fill_idx,)] = True
            # print("New chunk mask sum: ", new_chunk_mask.sum())
            r = new_chunk_mask, popped_thresholds
        else:
            r = chunk_mask, thresholds
    return r


@torch.no_grad()
def find_peak_grad_ascent_variable_decreasing_batch_size(
    sh_coeffs: Float[torch.Tensor, "batch sh_lm"],
    seed_theta: Float[torch.Tensor, " batch"],
    seed_phi: Float[torch.Tensor, " batch"],
    lr: float,
    momentum: float,
    max_epochs: int,
    l_max: int,
    tol_angular: float = 0.01 * (math.pi / 180),
    compile: bool = False,
    compile_kwargs: dict = dict(),
) -> tuple[Float[torch.Tensor, " batch"], Float[torch.Tensor, " batch"]]:
    """Finds the peak of a spherical function using gradient ascent.

    This function performs gradient ascent on a spherical function represented by
    spherical harmonic coefficients. This implementation reduces the batch size in the
    optimization loop immediately as functions converge, allowing for lower overhead
    as smaller batches are processed the further into the optimization loop.

    If using 'compile=True', it is recommended to use the 'dynamic=True' option in
    'compile_kwargs' so the compiled gradient function is optimized for arbitrary batch
    size.

    Parameters
    ----------
    sh_coeffs: Tensor, shape (batch_size, sh_lm)
        Spherical harmonic coefficients representing the spherical function.
    seed_theta: Tensor, shape (batch_size,)
        Initial elevation angles in radians for each batch.
    seed_phi: Tensor, shape (batch_size,)
        Initial azimuth angles in radians for each batch.
    lr: float
        Learning rate for the gradient ascent.
    momentum: float
        Momentum factor for the gradient ascent.
    max_epochs: int
        Maximum number of epochs to run the gradient ascent.
    l_max: int
        Maximum degree of the spherical harmonics to consider.
    tol_angular: float, optional
        Tolerance for the angular distance between consecutive iterations, by default
        0.01 degrees converted to radians.
    compile: bool, optional
        Whether to compile the function using `torch.compile`, by default False.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to `torch.compile`, by default an empty dict.

    Returns
    -------
    tuple of Tensors, shape (batch_size,) and (batch_size,)
        Polar coordinates of each spherical function's peak.
    """
    l, m = mrinr.vols.odf._get_degree_order_vecs(
        l_max=l_max, batch_size=1, device=sh_coeffs.device
    )
    l = l.squeeze(0)
    m = m.squeeze(0)

    params = torch.stack([seed_theta, seed_phi], -1)

    POLE_ANGLE_TOL = 1e-6
    if compile:
        grad_fn = partial(
            torch.compile(_batch_sh_grad, **compile_kwargs),
            pole_angle_tol=POLE_ANGLE_TOL,
        )
        arc_len_fn = torch.compile(mrinr.coords.unit_sphere_arc_len, **compile_kwargs)
    else:
        grad_fn = partial(
            _batch_sh_grad,
            pole_angle_tol=POLE_ANGLE_TOL,
        )
        arc_len_fn = mrinr.coords.unit_sphere_arc_len
    # If a batch has all zero phi angles or all zero sh coefficients, then that
    # batch should not be optimized over, and should already be considered "converged".
    converged = (seed_theta == 0.0) | (sh_coeffs == 0.0).all(-1)
    not_converged = ~converged

    nu_t = torch.zeros_like(params)
    grad_t = torch.zeros_like(nu_t)
    epoch = 0
    # Main gradient ascent loop.
    if converged.all():
        max_epochs = 0
    for epoch in range(max_epochs):
        nu_tm1 = nu_t
        # if dynamic_loop_batch_size:
        grad_t[not_converged] = grad_fn(
            sh_coeffs=sh_coeffs[not_converged],
            l=l,
            m=m,
            theta=params[not_converged, 0],
            phi=params[not_converged, 1],
        )

        nu_t = momentum * nu_tm1 + lr * grad_t
        # Batches that have converged should no longer be updated.
        # Add the gradient to perform gradient ascent, rather than gradient descent.
        params_tp1 = params + (nu_t * (not_converged.unsqueeze(-1)))
        # Theta does not "cycle back" between 0 and pi, it "bounces back" such as in
        # a sequence 0.01 -> 0.001 -> 0.0 -> 0.001 -> 0.01. This is unlike phi which
        # does cycle back: 2pi - 2eps -> 2pi - eps -> 0 -> 0 + eps ...
        theta_tp1 = params_tp1[..., 0]
        # The where() handles theta > pi, and the abs() handles theta < pi.
        params_tp1[..., 0] = torch.where(
            theta_tp1 > torch.pi,
            torch.pi - (theta_tp1 % torch.pi),
            torch.abs(theta_tp1),
        )
        # Phi just cycles back.
        params_tp1[..., 1] %= 2 * torch.pi
        arc_len_t_to_tp1 = arc_len_fn(
            theta_1=params[..., 0],
            phi_1=params[..., 1],
            theta_2=params_tp1[..., 0],
            phi_2=params_tp1[..., 1],
        )

        converged |= arc_len_t_to_tp1 < tol_angular
        if converged.all():
            break
        # Update the continue batch mask.
        not_converged = ~converged
        params = params_tp1

    return params[..., 0], params[..., 1]


@torch.no_grad()
def find_peak_grad_ascent_log_decreasing_batch_size(
    sh_coeffs: Float[torch.Tensor, "batch sh_lm"],
    seed_theta: Float[torch.Tensor, " batch"],
    seed_phi: Float[torch.Tensor, " batch"],
    lr: float,
    momentum: float,
    max_epochs: int,
    l_max: int,
    min_batch_size: int = 2**6,
    tol_angular: float = 0.01 * (math.pi / 180),
    compile: bool = False,
    compile_kwargs: dict = dict(),
) -> tuple[Float[torch.Tensor, " batch"], Float[torch.Tensor, " batch"]]:
    """Finds the peak of a spherical function using gradient ascent.

    This function performs gradient ascent on a spherical function represented by
    spherical harmonic coefficients.

    This implementation reduces the batch size in the optimization loop in discrete
    log-spaced steps, allowing for lower overhead as smaller batches are processed the
    further into the optimization loop while also allowing batch size specialization,
    i.e. allowing 'fullgraph=True' in 'compile_kwargs'.

    It may be necessary to increase the allowed function recompiles with torch dynamo
    with settings such as
    ```python
    import torch._dynamo
    torch._dynamo.config.recompile_limit = TORCHDYNAMO_CACHE_SIZE_LIMIT
    torch._dynamo.config.cache_size_limit = TORCHDYNAMO_CACHE_SIZE_LIMIT
    ```
    See
    <https://docs.pytorch.org/docs/2.8/torch.compiler_troubleshooting.html#changing-the-cache-size-limit>
    and
    <https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html#compile-time-caching-configuration>

    Parameters
    ----------
    sh_coeffs: Tensor, shape (batch_size, sh_lm)
        Spherical harmonic coefficients representing the spherical function.
    seed_theta: Tensor, shape (batch_size,)
        Initial elevation angles in radians for each batch.
    seed_phi: Tensor, shape (batch_size,)
        Initial azimuth angles in radians for each batch.
    lr: float
        Learning rate for the gradient ascent.
    momentum: float
        Momentum factor for the gradient ascent.
    max_epochs: int
        Maximum number of epochs to run the gradient ascent.
    l_max: int
        Maximum degree of the spherical harmonics to consider.
    min_batch_size: int
        Minimum batch size of the log-reduced loop batch sizes, by default 64.

        Batch sizes start at the batch size in 'sh_coeffs', and are reduced by powers
        of 2 until the batch size is at 'min_batch_size'.
    tol_angular: float, optional
        Tolerance for the angular distance between consecutive iterations, by default
        0.01 degrees converted to radians.
    compile: bool, optional
        Whether to compile the function using `torch.compile`, by default False.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to `torch.compile`, by default an empty dict.

    Returns
    -------
    tuple of Tensors, shape (batch_size,) and (batch_size,)
        Polar coordinates of each spherical function's peak.
    """
    l, m = mrinr.vols.odf._get_degree_order_vecs(
        l_max=l_max, batch_size=1, device=sh_coeffs.device
    )
    l = l.squeeze(0)
    m = m.squeeze(0)

    params = torch.stack([seed_theta, seed_phi], -1)

    POLE_ANGLE_TOL = 1e-6
    if compile:
        grad_fn = partial(
            torch.compile(_batch_sh_grad, **compile_kwargs),
            pole_angle_tol=POLE_ANGLE_TOL,
        )
        arc_len_fn = torch.compile(mrinr.coords.unit_sphere_arc_len, **compile_kwargs)
    else:
        grad_fn = partial(
            _batch_sh_grad,
            pole_angle_tol=POLE_ANGLE_TOL,
        )
        arc_len_fn = mrinr.coords.unit_sphere_arc_len
    # If a batch has all zero phi angles or all zero sh coefficients, then that
    # batch should not be optimized over, and should already be considered "converged".
    converged = (seed_theta == 0.0) | (sh_coeffs == 0.0).all(-1)

    # Set up masks for discrete batch size reduction.
    batch_size = params.shape[0]
    # Create set of discrete batch sizes that exponentially decrease by powers of 2.
    # Start at the power of 2 just below the batch size, and stop at the power of 2
    # just above or at the minimum batch size.
    start = np.floor(np.log2(batch_size - 1))
    stop = np.floor(np.log2(min_batch_size)) - 1
    if (batch_size <= min_batch_size) or (start <= stop):
        # No batch size reduction.
        batch_size_thresholds = np.array([])
    else:
        batch_size_thresholds = 2 ** np.arange(start, stop, step=-1)
    not_converged = ~converged
    continue_batch_mask = torch.ones_like(not_converged)

    nu_t = torch.zeros_like(params)
    grad_t = torch.zeros_like(nu_t)
    epoch = 0
    # Main gradient ascent loop.
    if converged.all():
        max_epochs = 0
    for epoch in range(max_epochs):
        nu_tm1 = nu_t
        # if dynamic_loop_batch_size:
        grad_t[continue_batch_mask] = grad_fn(
            sh_coeffs=sh_coeffs[continue_batch_mask],
            l=l,
            m=m,
            theta=params[continue_batch_mask, 0],
            phi=params[continue_batch_mask, 1],
        )

        nu_t = momentum * nu_tm1 + lr * grad_t
        # Batches that have converged should no longer be updated.
        # Add the gradient to perform gradient ascent, rather than gradient descent.
        params_tp1 = params + (nu_t * (not_converged.unsqueeze(-1)))
        # Theta does not "cycle back" between 0 and pi, it "bounces back" such as in
        # a sequence 0.01 -> 0.001 -> 0.0 -> 0.001 -> 0.01. This is unlike phi which
        # does cycle back: 2pi - 2eps -> 2pi - eps -> 0 -> 0 + eps ...
        theta_tp1 = params_tp1[..., 0]
        # The where() handles theta > pi, and the abs() handles theta < pi.
        params_tp1[..., 0] = torch.where(
            theta_tp1 > torch.pi,
            torch.pi - (theta_tp1 % torch.pi),
            torch.abs(theta_tp1),
        )
        # Phi just cycles back.
        params_tp1[..., 1] %= 2 * torch.pi
        arc_len_t_to_tp1 = arc_len_fn(
            theta_1=params[..., 0],
            phi_1=params[..., 1],
            theta_2=params_tp1[..., 0],
            phi_2=params_tp1[..., 1],
        )

        converged |= arc_len_t_to_tp1 < tol_angular
        if converged.all():
            break
        # Update the continue batch mask.
        not_converged = ~converged
        continue_batch_mask, batch_size_thresholds = _subset_mask_at_decreasing_thresh(
            real_mask=not_converged,
            chunk_mask=continue_batch_mask,
            thresholds=batch_size_thresholds,
        )

        params = params_tp1

    return params[..., 0], params[..., 1]


def __no_compile_sh_grad(
    sh_coeffs: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    pole_angle_tol: float = 1e-5,
) -> torch.Tensor:
    l_max = int(l.max().cpu().item())

    cart_coords = mrinr.coords.unit_sphere2xyz(theta=theta, phi=phi)
    x = cart_coords[..., 0]
    y = cart_coords[..., 1]

    elev_mrtrix = theta
    azim_mrtrix = torch.arctan2(y, x)
    at_pole = torch.sin(elev_mrtrix) < pole_angle_tol

    # Convenience function for indexing into l/m indexable matrices.
    bmask_l_m = lambda l_, m_: (
        Ellipsis,
        mrinr.vols.odf._broad_sh_mask(l_all=l, m_all=m, l=l_, m=m_),
    )

    norm_P_l_m = mrinr.vols.odf.norm_legendre_poly_cos_theta(
        degree=l, order=m, theta=elev_mrtrix
    )

    nonzero_degrees = list(range(2, l_max + 1, 2))

    ds_delev_zonal = (
        torch.sqrt(l * (l + 1))[bmask_l_m(nonzero_degrees, 0)]
        * norm_P_l_m[bmask_l_m(nonzero_degrees, 1)]
        * sh_coeffs[bmask_l_m(nonzero_degrees, 0)]
    ).sum(-1)

    m_pos_mask = bmask_l_m(None, range(1, l_max + 1))
    # Shape [batch_size, (n_sh_coeffs - ((l_max / 2) + 1))/2], for both pos m and neg m.
    m_pos = m[m_pos_mask]
    m_neg = -m_pos
    l_pos_m = l[m_pos_mask]
    cos_azim = np.sqrt(2) * torch.cos(m_pos * azim_mrtrix.unsqueeze(-1))
    sin_azim = np.sqrt(2) * torch.sin(m_pos * azim_mrtrix.unsqueeze(-1))
    coeff_m_pos = sh_coeffs[m_pos_mask]
    coeff_m_neg = sh_coeffs[..., mrinr.vols.odf._sh_idx(l_pos_m, m_neg)]

    k = (
        torch.sqrt((l_pos_m + m_pos) * (l_pos_m - m_pos + 1))
        * norm_P_l_m[..., mrinr.vols.odf._sh_idx(l_pos_m, m_pos - 1)]
    )
    # Only add the second term when m + 1 exists, aka not greater than m's respective
    # degree.
    k -= torch.where(
        l_pos_m > m_pos,
        torch.sqrt((l_pos_m - m_pos) * (l_pos_m + m_pos + 1))
        * norm_P_l_m[
            ..., mrinr.vols.odf._sh_idx(l_pos_m, m_pos + 1, invalid_idx_replace=0)
        ],
        0,
    )
    k *= -0.5
    ds_delev_nonzone = k * (cos_azim * coeff_m_pos + sin_azim * coeff_m_neg)
    ds_delev_nonzone = ds_delev_nonzone.sum(-1)

    # Numerically, derivative of the azimuth depends on whether or not we're at a pole.
    ds_dazim = torch.where(
        at_pole.unsqueeze(-1),
        k * (cos_azim * coeff_m_neg - sin_azim * coeff_m_pos),
        m_pos
        * norm_P_l_m[m_pos_mask]
        * (cos_azim * coeff_m_neg - sin_azim * coeff_m_pos),
    )
    ds_dazim = ds_dazim.sum(-1)

    # Scale ds/d azimuth to make this derivative into a gradient.
    # <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates>
    ds_dazim /= torch.where(~at_pole, torch.sin(elev_mrtrix), 1.0)

    ds_delev = ds_delev_zonal + ds_delev_nonzone

    jac = torch.stack([ds_delev, ds_dazim], dim=-1)

    return jac


@torch.no_grad()
def find_peak_grad_ascent_fixed_batch(
    sh_coeffs: torch.Tensor,
    seed_theta: torch.Tensor,
    seed_phi: torch.Tensor,
    lr: float,
    momentum: float,
    max_epochs: int,
    l_max: int,
    dynamic_loop_batch_size: bool = False,
    tol_angular: float = 0.01 * (torch.pi / 180),
    return_all_steps: bool = False,
    compile: bool = False,
    compile_kwargs: dict = dict(),
) -> Union[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
]:
    l, m = mrinr.vols.odf._get_degree_order_vecs(
        l_max=l_max, batch_size=1, device=sh_coeffs.device
    )
    l = l.squeeze(0)
    m = m.squeeze(0)

    params = torch.stack([seed_theta, seed_phi], -1)
    param_steps = list() if return_all_steps else None

    POLE_ANGLE_TOL = 1e-6
    if compile:
        grad_fn = partial(
            torch.compile(_batch_sh_grad, **compile_kwargs),
            pole_angle_tol=POLE_ANGLE_TOL,
        )
        arc_len_fn = torch.compile(mrinr.coords.unit_sphere_arc_len, **compile_kwargs)
    else:
        grad_fn = partial(
            __no_compile_sh_grad,
            pole_angle_tol=POLE_ANGLE_TOL,
        )
        arc_len_fn = mrinr.coords.unit_sphere_arc_len
    # If a batch has all zero phi angles or all zero sh coefficients, then that
    # batch should not be optimized over, and should already be considered "converged"
    converged = (seed_theta == 0.0) | (sh_coeffs == 0.0).all(-1)
    # converged = torch.zeros_like(seed_theta).bool()
    nu_t = torch.zeros_like(params)
    grad_t = torch.zeros_like(nu_t)
    epoch = 0
    if return_all_steps:
        param_steps.append(params)

    # Main gradient ascent loop.
    if converged.all():
        max_epochs = 0
    for epoch in range(max_epochs):
        nu_tm1 = nu_t
        if dynamic_loop_batch_size:
            grad_t[~converged] = grad_fn(
                sh_coeffs=sh_coeffs[~converged],
                l=l[~converged],
                m=m[~converged],
                theta=params[~converged, 0],
                phi=params[~converged, 1],
            )
        else:
            grad_t = grad_fn(
                sh_coeffs=sh_coeffs, l=l, m=m, theta=params[..., 0], phi=params[..., 1]
            )
        nu_t = momentum * nu_tm1 + lr * grad_t
        # Batches that have converged should no longer be updated.
        # Add the gradient to perform gradient ascent, rather than gradient descent.
        params_tp1 = params + (nu_t * (~converged.unsqueeze(-1)))
        # Theta does not "cycle back" between 0 and pi, it "bounces back" such as in
        # a sequence 0.01 -> 0.001 -> 0.0 -> 0.001 -> 0.01. This is unlike phi which
        # does cycle back: 2pi - 2eps -> 2pi - eps -> 0 -> 0 + eps ...
        theta_tp1 = params_tp1[..., 0]
        # The where() handles theta > pi, and the abs() handles theta < pi.
        params_tp1[..., 0] = torch.where(
            theta_tp1 > torch.pi,
            torch.pi - (theta_tp1 % torch.pi),
            torch.abs(theta_tp1),
        )
        # Phi just cycles back.
        params_tp1[..., 1] %= 2 * torch.pi
        arc_len_t_to_tp1 = arc_len_fn(
            theta_1=params[..., 0],
            phi_1=params[..., 1],
            theta_2=params_tp1[..., 0],
            phi_2=params_tp1[..., 1],
        )

        converged |= arc_len_t_to_tp1 < tol_angular
        if converged.all():
            break

        params = params_tp1
        if dynamic_loop_batch_size:
            # Zero out gradients for converged batches.
            grad_t[converged] = 0.0
        if return_all_steps:
            param_steps.append(params)

    if not return_all_steps:
        ret = (params[..., 0], params[..., 1])
    else:
        ret = (params[..., 0], params[..., 1], param_steps)

    return ret
