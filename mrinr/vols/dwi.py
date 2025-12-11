# -*- coding: utf-8 -*-
# Functions and objects that can be applied to diffusion-weighted volumes.

from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch

import mrinr


def resample_dwi_directions(
    dwi: torch.Tensor,
    src_grad_mrtrix_table: torch.Tensor,
    target_grad_mrtrix_table: torch.Tensor,
    bval_round_decimals=-2,
    k_nearest_points=5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_g = src_grad_mrtrix_table[:, :-1].detach().cpu().numpy()
    src_bvals = src_grad_mrtrix_table[:, -1].detach().cpu().numpy()
    target_g = target_grad_mrtrix_table[:, :-1].detach().cpu().numpy()
    target_bvals = target_grad_mrtrix_table[:, -1].detach().cpu().numpy()

    src_shells = np.round(src_bvals, decimals=bval_round_decimals).astype(int)
    target_shells = np.round(target_bvals, decimals=bval_round_decimals).astype(int)
    # # Force all vectors to have unit norm.
    # src_g_norm = np.linalg.norm(src_g, ord=2, axis=-1, keepdims=True)
    # src_g_norm = np.where(np.isclose(src_g_norm, 0, atol=1e-4), 1.0, src_g_norm)
    # src_g = src_g / src_g_norm
    # target_g_norm = np.linalg.norm(target_g, ord=2, axis=-1, keepdims=True)
    # target_g_norm = np.where(
    #     np.isclose(target_g_norm, 0, atol=1e-4), 1.0, target_g_norm
    # )
    # target_g = target_g / target_g_norm
    # # Project all vectors into the top hemisphere, as we have antipodal symmetry in dwi.
    # src_g = np.where(src_g[:, -1, None] < 0, -src_g, src_g)
    # target_g = np.where(target_g[:, -1, None] < 0, -target_g, target_g)
    # # src_g[:, -1] = np.abs(src_g[:, -1])
    # # target_g[:, -1] = np.abs(target_g[:, -1])

    # # Double precision floats are more numerically stable, so explicitly cast to that
    # # inside the arccos.
    # p_arc_len = np.arccos(
    #     einops.einsum(
    #         target_g.astype(np.float64), src_g.astype(np.float64), "b1 d, b2 d -> b1 b2"
    #     )
    # )
    p_arc_len = mrinr.coords._antipodal_sym_pairwise_arc_len(
        torch.from_numpy(target_g), torch.from_numpy(src_g)
    ).numpy()
    p_arc_len = p_arc_len.astype(target_g.dtype)
    p_arc_len = np.nan_to_num(p_arc_len, nan=0)
    p_arc_w = (np.pi / 2) - p_arc_len
    # Zero-out weights between dissimilar shells.
    p_arc_w[target_shells[:, None] != src_shells[None, :]] = 0.0

    # b0 volumes are just assigned a copy of the source b0 that is nearest to the
    # relative index of the target b0.
    # Zero-out all b0 weights
    p_arc_w[(target_shells[:, None] == 0) & (src_shells[None, :] == 0)] = 0.0
    src_b0_idx = np.where(np.isclose(src_shells, 0))[0]
    src_b0_relative_idx = src_b0_idx / len(src_shells)
    target_b0_idx = np.where(np.isclose(target_shells, 0))[0]
    target_b0_relative_idx = target_b0_idx / len(target_shells)

    for (
        i_target_b0,
        rel_target_b0_i_idx,
    ) in zip(target_b0_idx, target_b0_relative_idx):
        j_selected_src_b0 = src_b0_idx[
            np.argmin(np.abs(rel_target_b0_i_idx - src_b0_relative_idx))
        ]
        # pi/2 is the max weight that can be selected before normalization.
        p_arc_w[i_target_b0, j_selected_src_b0] = np.pi / 2

    # Zero-out any weights lower than the top k weights.
    p_arc_w[
        p_arc_w < (np.flip(np.sort(p_arc_w, -1), -1))[:, (k_nearest_points - 1), None]
    ] = 0.0
    # If any weights are close to pi/2, then zero-out all other weights and just produce
    # a copy of the DWI that is (nearly) angularly identical to the target.
    for i_target in range(p_arc_w.shape[0]):
        if np.isclose(p_arc_w[i_target], (np.pi / 2), atol=1e-5).any():
            pi_idx = np.argmax(p_arc_w[i_target])
            p_arc_w[i_target] = p_arc_w[i_target] * 0.0
            p_arc_w[i_target, pi_idx] = np.pi / 2

    # Normalize weights to sum to 1.0.
    norm_p_arc_w = p_arc_w / p_arc_w.sum(1, keepdims=True)
    # Weigh and combine the source DWIs to make the target DWIs.
    target_dwis = list()
    src_dwi = dwi.detach().cpu().numpy()
    for i_target in range(p_arc_w.shape[0]):
        norm_w_i = norm_p_arc_w[i_target]
        tmp_dwi_i = np.zeros_like(src_dwi[0])
        for j_src, w_j in enumerate(norm_w_i):
            if np.isclose(w_j, 0):
                continue
            tmp_dwi_i += src_dwi[j_src] * w_j
        target_dwis.append(tmp_dwi_i)

    target_dwis = torch.from_numpy(np.stack(target_dwis, 0)).to(dwi)

    return target_dwis


def add_rician_noise(
    dwi: torch.Tensor,
    grad_table: pd.DataFrame,
    snr: Union[float, torch.Tensor],
    S0: Union[float, torch.Tensor],
    rng: torch.Generator,
    dwi_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Generator]:
    """Adds Rician-sampled complex noise to DWI volumes based on given SNR and S0.

    Based on implementations found in dipy
    <https://dipy.org/documentation/1.7.0/reference/dipy.sims/#add-noise> and
    <https://github.com/bchimagine/fODF_deep_learning/blob/b93a40b2de841aec5c619bbf7bd7fc4edc626ea6/crl_aux.py#L41>,
    the implementation of:

        D. Karimi, L. Vasung, C. Jaimes, F. Machado-Rivas, S. K. Warfield, and A. Gholipour,
        "Learning to estimate the fiber orientation distribution function from
        diffusion-weighted MRI," NeuroImage, vol. 239, p. 118316, Oct. 2021,
        doi: 10.1016/j.neuroimage.2021.118316.

    Parameters
    ----------
    dwi : torch.Tensor
        DWI channel-first volume Tensor.
    grad_table : pd.DataFrame
        MRtrix-style gradient table as a DataFrame.
    snr : Union[float, torch.Tensor]
        Target signal-to-noise ratio
    S0 : Union[float, torch.Tensor]
        Give S0 reference value for calculating sigma
    rng : torch.Generator
        Pytorch random number generator
    dwi_mask : Optional[torch.Tensor]
        Channel-first spatial mask, used in b0 quantile if given, by default None

    Returns
    -------
    Tuple[torch.Tensor, torch.Generator]
        Returns noised DWI Tensor and the random number generator.

    """
    rng_fork = torch.Generator(device=rng.device)
    rng_fork.set_state(rng.get_state())
    b = torch.from_numpy(grad_table.b.to_numpy()).to(dwi)
    shells = torch.round(b, decimals=-2)

    if not torch.is_tensor(S0):
        S0 = torch.ones_like(dwi) * S0
    if not torch.is_tensor(snr):
        snr = torch.ones_like(dwi) * snr
    # SNR = S0 / sigma
    sigma = S0 / snr
    sigma = sigma.broadcast_to(dwi.shape)
    N_real = torch.normal(mean=0, std=sigma, generator=rng_fork)
    N_complex = torch.normal(mean=0, std=sigma, generator=rng_fork)

    S = torch.sqrt((dwi + N_real) ** 2 + N_complex**2)
    if dwi_mask is not None:
        S = S * dwi_mask

    return S, rng_fork


def degrade_dwi(
    dwi: torch.Tensor,
    affine_vox2real: torch.Tensor,
    grad_table: pd.DataFrame,
    brain_mask: torch.Tensor,
    downsample_factor: float,
    prefilter_sigma_scale_coeff: float = 2.5,
    prefilter_sigma_truncate: float = 4.0,
    manual_crop_lr_sides: Optional[Tuple[Tuple[int, int], ...]] = (
        (1, 1),
        (1, 1),
        (1, 1),
    ),
    noise_snr: Optional[float] = None,
    S0_noise_b0_quantile: Optional[float] = None,
    rng: Optional[torch.Generator] = None,
) -> dict:
    affine_fr = affine_vox2real.to(torch.float64)
    fr_spacing = np.array(nib.affines.voxel_sizes(affine_fr.detach().cpu().numpy()))
    lr_spacing = fr_spacing * downsample_factor

    # Set spacing for LR fov.
    orig_fr_bb_coords = mrinr.coords.fov_bb_coords_from_vox_shape(
        affine_fr, vox_vol=dwi
    )
    lr_fov_coords, affine_lr_vox2real = mrinr.coords.scale_fov_spacing(
        orig_fr_bb_coords,
        affine_fr,
        spacing_scale_factors=(downsample_factor,) * 3,
        set_affine_orig_to_fov_orig=True,
        new_fov_align_direction="interior",
    )
    # Prefilter/blur DWI before downsampling.
    blur_dwi = mrinr.vols.prefilter_gaussian_blur(
        dwi,
        src_spacing=tuple(fr_spacing),
        target_spacing=tuple(lr_spacing),
        sigma_scale_coeff=prefilter_sigma_scale_coeff,
        sigma_truncate=prefilter_sigma_truncate,
    )
    # Downsample DWI.
    lr_real_coord_grid = mrinr.coords._fov_coord_grid(lr_fov_coords, affine_lr_vox2real)
    lr_dwi = mrinr.coords.sample_vol(
        blur_dwi,
        coords_mm_xyz=lr_real_coord_grid,
        affine_vox2mm=affine_fr,
        mode="linear",
        align_corners=True,
        override_out_of_bounds_val=torch.nan,
    )
    lr_brain_mask = mrinr.coords.sample_vol(
        brain_mask,
        coords_mm_xyz=lr_real_coord_grid,
        affine_vox2mm=affine_fr,
        mode="nearest",
        align_corners=True,
        override_out_of_bounds_val=torch.nan,
    )

    # Allow for forced cropping of LR, useful for whole-volume processing with a fixed
    # downsampling factor, which would usually result in NaNs on the edge(s).
    if manual_crop_lr_sides is not None:
        crops_low_high = manual_crop_lr_sides
        lr_dwi, cropped_affine_lr_vox2real = mrinr.vols.crop_vox(
            lr_dwi, affine_lr_vox2real, *crops_low_high
        )
        lr_brain_mask, _ = mrinr.vols.crop_vox(
            lr_brain_mask, affine_lr_vox2real, *crops_low_high
        )
        affine_lr_vox2real = cropped_affine_lr_vox2real
        lr_fov_coords = mrinr.coords.fov_bb_coords_from_vox_shape(
            affine_lr_vox2real, vox_vol=lr_dwi
        )

    # Add Rician noise to downsampled dwi.
    if noise_snr is not None:
        if (rng is None) or (S0_noise_b0_quantile is None):
            raise ValueError(
                "ERROR: Cannot add noise, rng and S0_noise_b0_quantile must no be None"
            )
        rng_fork = torch.Generator()
        rng_fork.set_state(rng.get_state())
        # Calculate S0 value needed for noise injection.
        shells = grad_table.b.to_numpy().round(-2)
        # Take all b0s.
        b0_idx = (np.where(shells == 0)[0][:9],)
        b0_select = np.zeros_like(shells).astype(bool).copy()
        b0_select[b0_idx] = True
        b0s = dwi[shells == 0]
        b0s = b0s[brain_mask.broadcast_to(b0s.shape)].flatten()
        S0_noise = float(
            np.quantile(b0s.detach().cpu().numpy(), q=S0_noise_b0_quantile)
        )
        lr_dwi, rng_fork = add_rician_noise(
            lr_dwi,
            grad_table,
            snr=noise_snr,
            rng=rng_fork,
            S0=S0_noise,
            dwi_mask=lr_brain_mask,
        )

    return dict(
        dwi=lr_dwi,
        brain_mask=lr_brain_mask,
        grad_table=grad_table,
        affine=affine_lr_vox2real,
    )
