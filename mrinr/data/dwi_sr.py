# -*- coding: utf-8 -*-
# Functions and classes for loading and processing subject DWIs for super-resolution
# tasks.
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, TypedDict, Union

import einops
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import torch

import mrinr


class LoadedDWISuperResSampleDict(TypedDict):
    subj_id: str
    affine_vox2real: torch.Tensor
    dwi: torch.Tensor
    grad_table: pd.DataFrame
    odf: torch.Tensor
    brain_mask: torch.Tensor
    wm_mask: torch.Tensor
    gm_mask: torch.Tensor
    csf_mask: torch.Tensor


class PreprocedDWISuperResSubjDict(LoadedDWISuperResSampleDict):
    S0_noise: float
    patch_sampling_cumulative_weights: torch.Tensor


class DWISuperResLRFRSample(TypedDict):
    subj_id: str
    affine_lr_vox2real: torch.Tensor
    lr_real_coords: torch.Tensor
    lr_spacing: Tuple[float, ...]
    lr_fov_coords: torch.Tensor
    lr_dwi: torch.Tensor
    grad_table: np.ndarray
    affine_vox2real: torch.Tensor
    full_res_real_coords: torch.Tensor
    full_res_spacing: Tuple[float, ...]
    full_res_fov_coords: torch.Tensor
    odf: torch.Tensor
    brain_mask: torch.Tensor
    wm_mask: torch.Tensor
    gm_mask: torch.Tensor
    csf_mask: torch.Tensor


class _VoxRealAffineSpace(NamedTuple):
    vox_vol: torch.Tensor
    affine_vox2real: torch.Tensor
    fov_bb_real: torch.Tensor


class _BatchVoxRealAffineSpace(NamedTuple):
    vox_vols: Tuple[torch.Tensor, ...]
    affine_vox2real: torch.Tensor
    fov_bb_real: torch.Tensor


def load_dwi_super_res_subj_sample(
    subj_id: str,
    dwi_f: Path,
    grad_mrtrix_f: Path,
    odf_f: Path,
    fivett_seg_f: Path,
    brain_mask_f: Path,
) -> LoadedDWISuperResSampleDict:
    target_im_orient = "RAS"

    dwi_data = mrinr.data.io.load_dwi(
        dwi_f, grad_mrtrix_f=grad_mrtrix_f, reorient_im_to=target_im_orient
    )
    dwi = dwi_data["dwi"].to(torch.float32)
    affine_vox2real = dwi_data["affine"].to(torch.float32)
    grad_table = dwi_data["grad_table"]

    odf_data = mrinr.data.io.load_vol(odf_f, reorient_im_to=target_im_orient)
    odf = odf_data["vol"].to(dwi)

    brain_mask_data = mrinr.data.io.load_vol(
        brain_mask_f, reorient_im_to=target_im_orient, ensure_channel_dim=True
    )
    brain_mask = brain_mask_data["vol"].bool()

    fivett_data = mrinr.data.io.load_vol(fivett_seg_f, reorient_im_to=target_im_orient)
    fivett_seg = fivett_data["vol"].bool()

    # Ensure that all vox-to-real affines and image shapes are the same.
    for vol in (odf, brain_mask, fivett_seg):
        assert tuple(vol.shape[1:]) == tuple(dwi.shape[1:])
    for d in (odf_data, brain_mask_data, fivett_data):
        assert torch.isclose(affine_vox2real, d["affine"].to(affine_vox2real)).all()

    # Construct the 3 tissue masks from the fivett segmentation.
    wm_mask = fivett_seg[2].unsqueeze(0)
    gm_mask = (fivett_seg[0] | fivett_seg[1]).unsqueeze(0)
    csf_mask = fivett_seg[3].unsqueeze(0)

    return LoadedDWISuperResSampleDict(
        subj_id=subj_id,
        dwi=dwi,
        grad_table=grad_table,
        affine_vox2real=affine_vox2real,
        odf=odf,
        brain_mask=brain_mask,
        wm_mask=wm_mask,
        gm_mask=gm_mask,
        csf_mask=csf_mask,
    )


def lazy_sample_patch_from_super_res_sample(
    sample_dict: PreprocedDWISuperResSubjDict,
    patch_size: Tuple[int, ...],
    num_samples: int,
    rng: Union[str, torch.Generator] = "default",
    skip_crop_keys: Optional[Tuple[str]] = None,
) -> "mrinr.data._CallablePromisesList[PreprocedDWISuperResSubjDict]":
    def sample_single_patch(
        sample_dict: PreprocedDWISuperResSubjDict,
        patch_size,
        rng,
        skip_crop_keys: set,
        drop_keys: set,
    ) -> PreprocedDWISuperResSubjDict:
        cumul_w = sample_dict["patch_sampling_cumulative_weights"][0]

        rng_fork = torch.Generator(device=rng.device)
        rng_fork.set_state(rng.get_state())
        r_v = torch.rand(
            1, generator=rng_fork, dtype=cumul_w.dtype, device=cumul_w.device
        )

        patch_center_flat_idx = torch.searchsorted(cumul_w.view(-1), r_v, side="right")
        patch_center_idx = np.unravel_index(
            patch_center_flat_idx.cpu().item(), shape=tuple(cumul_w.shape)
        )
        patch_center_idx = np.array(patch_center_idx)
        p_s = np.array(patch_size)
        bb_vox_high = np.array(tuple(cumul_w.shape)) - 1
        patch_size_lower = np.ceil(p_s / 2).astype(int)
        patch_size_upper = np.floor(p_s / 2).astype(int)
        patch_crop_low = patch_center_idx - patch_size_lower
        patch_crop_high = bb_vox_high - (patch_center_idx + patch_size_upper - 1)

        affine_key = "affine_vox2real"
        affine_vox2real = sample_dict[affine_key]
        out_dict = dict()
        for k, v in sample_dict.items():
            if k == affine_key or k in drop_keys:
                continue

            if k in skip_crop_keys or not mrinr.utils.is_vol(v, ch=True, batch=False):
                out_dict[k] = v
            else:
                crop_vol, new_aff = mrinr.vols.crop_vox(
                    v,
                    affine_vox2real,
                    *[
                        (patch_crop_low[i], patch_crop_high[i])
                        for i in range(len(patch_crop_low))
                    ],
                )

                out_dict[k] = crop_vol
        out_dict[affine_key] = new_aff

        return PreprocedDWISuperResSubjDict(**out_dict)

    if rng == "default":
        state_rng = torch.default_generator.get_state()
    else:
        state_rng = rng.get_state()
    rng_fork = torch.Generator()
    rng_fork.set_state(state_rng)

    if skip_crop_keys is None:
        skip_crop_keys = set()
    else:
        skip_crop_keys = set(skip_crop_keys)
    drop_keys = set()
    lazy_samples = mrinr.data._CallablePromisesList()
    for i in range(num_samples):
        # Iterate the rng state.
        torch.randint(10, size=(1,), generator=rng_fork)
        # Create a new "forked" rng
        rng_i = torch.Generator(device=rng_fork.device)
        rng_i.set_state(rng_fork.get_state())
        lazy_samples.append(
            partial(
                sample_single_patch,
                sample_dict=sample_dict,
                patch_size=patch_size,
                rng=rng_i,
                skip_crop_keys=skip_crop_keys,
                drop_keys=drop_keys,
            )
        )

    if rng == "default":
        torch.default_generator.set_state(rng_fork.get_state())

    return lazy_samples


def preproc_loaded_super_res_subj(
    loaded_super_res_subj: LoadedDWISuperResSampleDict,
    S0_noise_b0_quantile: float,
    resample_target_grad_table: Optional[pd.DataFrame] = None,
    patch_sampling_w_erosion: Optional[int] = None,
) -> PreprocedDWISuperResSubjDict:
    grad_table = loaded_super_res_subj["grad_table"]
    dwi = loaded_super_res_subj["dwi"]
    brain_mask = loaded_super_res_subj["brain_mask"]
    update_subj_dict = dict()

    # Calculate S0 value needed for noise injection later on.
    shells = grad_table.b.to_numpy().round(-2)
    # Take all b0s.
    b0_idx = (np.where(shells == 0)[0][:9],)
    b0_select = np.zeros_like(shells).astype(bool).copy()
    b0_select[b0_idx] = True
    b0s = dwi[shells == 0]
    b0s = b0s[brain_mask.broadcast_to(b0s.shape)].flatten()
    S0_noise = float(np.quantile(b0s.detach().cpu().numpy(), q=S0_noise_b0_quantile))
    del b0s

    # Resample the DWIs according to the given gradient table.
    if resample_target_grad_table is not None:
        src_grad = torch.from_numpy(grad_table.to_numpy()).to(dwi)
        target_grad = torch.from_numpy(resample_target_grad_table.to_numpy()).to(
            src_grad
        )
        dwi = mrinr.vols.dwi.resample_dwi_directions(
            dwi, src_grad_mrtrix_table=src_grad, target_grad_mrtrix_table=target_grad
        )
        grad_table = resample_target_grad_table.copy(deep=True)
        update_subj_dict["dwi"] = dwi
        update_subj_dict["grad_table"] = grad_table

    # Create patch sampling weights volume according to the brain mask.
    spacing = nib.affines.voxel_sizes(
        loaded_super_res_subj["affine_vox2real"].detach().cpu().numpy()
    )
    sample_w = mrinr.vols.distance_transform_mask(brain_mask, spacing=spacing)
    mask_w = sample_w > 0
    if patch_sampling_w_erosion is not None:
        m = mask_w.detach().cpu().clone().numpy().astype(bool)
        # Take out channel dim.
        m = m[0]
        m = scipy.ndimage.binary_fill_holes(m)
        m = scipy.ndimage.binary_erosion(m, iterations=patch_sampling_w_erosion)
        m = m[None]
        mask_w = torch.from_numpy(m).to(mask_w) & mask_w
    sample_w = sample_w * mask_w
    norm_sample_w = sample_w / sample_w.sum()

    cumul_norm_sample_w = torch.cumsum(norm_sample_w.view(-1), dim=0).view(
        norm_sample_w.shape
    )

    return PreprocedDWISuperResSubjDict(
        **(loaded_super_res_subj | update_subj_dict),
        patch_sampling_cumulative_weights=cumul_norm_sample_w,
        S0_noise=S0_noise,
    )


def preproc_super_res_sample(
    super_res_sample_dict: PreprocedDWISuperResSubjDict,
    downsample_factor_range: Tuple[float, float],
    noise_snr_range: Optional[Tuple[float, float]],
    rng: Union[str, torch.Generator] = "default",
    prefilter_sigma_scale_coeff: float = 2.5,
    prefilter_sigma_truncate: float = 4.0,
    manual_crop_lr_sides: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> DWISuperResLRFRSample:
    if rng == "default":
        state_rng = torch.default_generator.get_state()
    else:
        state_rng = rng.get_state()
    rng_fork = torch.Generator()
    rng_fork.set_state(state_rng)
    # print(
    #     f"{torch.utils.data.get_worker_info().id}: {rng_fork.get_state().float().mean()}",
    #     flush=True,
    # )

    affine_fr_vox2real = super_res_sample_dict["affine_vox2real"].to(torch.float64)
    fr_spacing = np.array(
        nib.affines.voxel_sizes(affine_fr_vox2real.detach().cpu().numpy())
    )
    dwi = super_res_sample_dict["dwi"]

    downsample_min = float(downsample_factor_range[0])
    downsample_max = float(downsample_factor_range[1])
    downsample_factor = torch.rand(1, generator=rng_fork, dtype=torch.float64).item()
    downsample_factor = (
        downsample_factor * (downsample_max - downsample_min) + downsample_min
    )
    lr_spacing = fr_spacing * downsample_factor

    # Set spacing for LR fov.
    orig_fr_bb_coords = mrinr.coords.fov_bb_coords_from_vox_shape(
        affine_fr_vox2real, vox_vol=dwi
    )
    lr_fov_coords, affine_lr_vox2real = mrinr.coords.scale_fov_spacing(
        orig_fr_bb_coords,
        affine_fr_vox2real,
        spacing_scale_factors=(downsample_factor,) * 3,
        set_affine_orig_to_fov_orig=True,
        new_fov_align_direction="interior",
    )
    # Prefilter/blur DWI patch before downsampling.
    blur_dwi = mrinr.vols.prefilter_gaussian_blur(
        dwi,
        src_spacing=tuple(fr_spacing),
        target_spacing=tuple(lr_spacing),
        sigma_scale_coeff=prefilter_sigma_scale_coeff,
        sigma_truncate=prefilter_sigma_truncate,
    )
    # Downsample DWI patch.
    lr_real_coord_grid = mrinr.coords.fov_coord_grid(lr_fov_coords, affine_lr_vox2real)
    lr_dwi = mrinr.vols.sample_vol(
        blur_dwi,
        coords_mm_xyz=lr_real_coord_grid,
        affine_vox2mm=affine_fr_vox2real,
        mode="linear",
        align_corners=True,
        override_out_of_bounds_val=torch.nan,
    )
    # Crop the LR fov s.t. the smallest possible LR shape is the same as all LR samples
    # (assuming the same FR input shape).
    if downsample_min != downsample_max:
        lr_space = mrinr.vols._crop_lr_inside_smallest_lr(
            lr_dwi,
            affine_lr_vox2real,
            fr_fov_bb_real=orig_fr_bb_coords,
            affine_fr_vox2real=affine_fr_vox2real,
            max_spacing_scale_factor=downsample_max,
        )

        lr_dwi = lr_space.vox_vol
        affine_lr_vox2real = lr_space.affine_vox2real
        lr_fov_coords = lr_space.fov_bb_real

    # Allow for forced cropping of LR, useful for whole-volume processing with a fixed
    # downsampling factor, which would usually result in NaNs on the edge(s).
    if manual_crop_lr_sides is not None:
        crops_low_high = manual_crop_lr_sides
        lr_dwi, affine_lr_vox2real = mrinr.vols.crop_vox(
            lr_dwi, affine_lr_vox2real, *crops_low_high
        )
        lr_fov_coords = mrinr.coords.fov_bb_coords_from_vox_shape(
            affine_lr_vox2real, vox_vol=lr_dwi
        )

    # Add Rician noise to downsampled patch.
    if noise_snr_range is not None:
        snr_min = float(noise_snr_range[0])
        snr_max = float(noise_snr_range[1])
        snr = torch.rand(1, generator=rng_fork, dtype=torch.float64).item()
        snr = snr * (snr_max - snr_min) + snr_min
        lr_dwi, rng_fork = mrinr.vols.add_rician_noise(
            lr_dwi,
            super_res_sample_dict["grad_table"],
            snr=snr,
            rng=rng_fork,
            S0=super_res_sample_dict["S0_noise"],
            # Just hack together an LR brain mask, not worth resampling.
            dwi_mask=torch.all(
                ~torch.isclose(lr_dwi, lr_dwi.new_zeros(1)), dim=0, keepdim=True
            ),
        )

    # Crop the input FR sample to be contained within the cropped LR space.
    fr_vol_keys = ("odf", "brain_mask", "wm_mask", "gm_mask", "csf_mask")
    fr_vols = tuple([super_res_sample_dict[k] for k in fr_vol_keys])
    crop_frs_space = mrinr.vols._crop_frs_inside_lr(
        *fr_vols,
        affine_fr_vox2real=affine_fr_vox2real,
        lr_fov_bb_coords=lr_fov_coords,
    )
    fr_vols = crop_frs_space.vox_vols
    affine_fr_vox2real = crop_frs_space.affine_vox2real
    fr_bb_coords = crop_frs_space.fov_bb_real

    # Generate coordinates for both full-res and low-res spaces.
    lr_real_coords = mrinr.coords.fov_coord_grid(lr_fov_coords, affine_lr_vox2real)
    fr_real_coords = mrinr.coords.fov_coord_grid(fr_bb_coords, affine_fr_vox2real)
    # Collate function needs coordinate/channel-first tensors. Reshaping to
    # coordinate-first tensors will need to be done in the training loop, after sampling
    # from the DataLoader.
    lr_real_coords = einops.rearrange(lr_real_coords, "i j k coord -> coord i j k")
    fr_real_coords = einops.rearrange(fr_real_coords, "i j k coord -> coord i j k")

    out_dict = DWISuperResLRFRSample(
        subj_id=super_res_sample_dict["subj_id"],
        affine_lr_vox2real=affine_lr_vox2real.to(torch.float32),
        lr_real_coords=lr_real_coords.to(torch.float32),
        lr_spacing=lr_spacing,
        lr_fov_coords=lr_fov_coords.to(torch.float32),
        lr_dwi=lr_dwi,
        grad_table=super_res_sample_dict["grad_table"].to_numpy(),  # must be ndarray
        affine_vox2real=affine_fr_vox2real.to(torch.float32),
        full_res_real_coords=fr_real_coords.to(torch.float32),
        full_res_spacing=fr_spacing,
        full_res_fov_coords=fr_bb_coords.to(torch.float32),
        **{fr_vol_keys[i]: fr_vols[i] for i in range(len(fr_vols))},
    )

    # If the generator was the default pytorch generator, then update the global rng
    # state.
    if rng == "default":
        torch.default_generator.set_state(rng_fork.get_state())

    return out_dict
