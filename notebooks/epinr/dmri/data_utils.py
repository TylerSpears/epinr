# -*- coding: utf-8 -*-
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import pandas as pd
import skimage
import skimage.filters
import torch

import mrinr

PE_DIR_ALIASES = collections.defaultdict(
    lambda: None,
    {
        k: "ap"
        for k in ("ap", "a-p", "y-", "-y", "-j", "j-", (0, -1, 0), (0.0, -1.0, 0.0))
    }
    | {
        k: "pa"
        for k in (
            "pa",
            "p-a",
            "y",
            "y+",
            "+y",
            "j",
            "+j",
            "j+",
            (0, 1, 0),
            (0.0, 1.0, 0.0),
        )
    },
)

EMPTY_SESSION_TOKEN = "EMPTY"
EMPTY_RUN_TOKEN = "EMPTY"
EMPTY_TASK_TOKEN = "EMPTY"


dataset_table_cols = (
    "dataset_name",
    "subj_id",
    "session_id",
    "run_id",
    "dwi_idx",
    "dwi",
    "pe_dir",
    "dwi_mask",
    "total_readout_time_s",
    "t1w_reg_dwi",
    "t1w_mask",
    "fs_label",
    "t1w_wm_mask",
    "t1w_gm_mask",
    "t1w_csf_mask",
    "t1w2acpc_affine",
    "mni2t1w_warp",
    "topup_displacement_hz",
    "topup_corrected_dwi",
    "suscept_atlas_mm_dir_ap",
)


@dataclass
class DWISubjectData:
    dataset_name: str
    subj_id: str
    session_id: str
    run_id: str
    # DWI/b0 data
    dwi_idx: int
    b0: mrinr.typing.SingleScalarVolume
    b0_affine: mrinr.typing.SingleHomogeneousAffine3D
    b0_scanner_coord_grid: mrinr.typing.SingleCoordGrid3D
    b0_fov: torch.Tensor
    b0_min_coord: torch.Tensor
    b0_mask: mrinr.typing.SingleScalarVolume
    pe_dir: str
    total_readout_time_s: float
    # T1w anatomical data
    t1w: mrinr.typing.SingleScalarVolume
    t1w_affine: mrinr.typing.SingleHomogeneousAffine3D
    t1w_coord_grid: mrinr.typing.SingleCoordGrid3D
    t1w_fov: torch.Tensor
    t1w_min_coord: torch.Tensor
    t1w_mask: mrinr.typing.SingleScalarVolume
    # Scaled T1w and b0 volumes for training.
    scaled_b0: Optional[mrinr.typing.SingleScalarVolume] = None
    scaled_t1w: Optional[mrinr.typing.SingleScalarVolume] = None
    # Topup-corrected ground truth data
    topup_displace_hz: Optional[mrinr.typing.SingleCoordGrid3D] = None
    topup_corrected_b0: Optional[mrinr.typing.SingleScalarVolume] = None
    # Anatomical labels and transforms
    fs_label: Optional[mrinr.typing.SingleScalarVolume] = None
    t1w_wm_mask: Optional[mrinr.typing.SingleScalarVolume] = None
    t1w_gm_mask: Optional[mrinr.typing.SingleScalarVolume] = None
    t1w_csf_mask: Optional[mrinr.typing.SingleScalarVolume] = None
    t1w2acpc_affine: Optional[mrinr.typing.SingleHomogeneousAffine3D] = None
    mni2t1w_warp: Optional[Path] = None
    suscept_atlas_mm: Optional[mrinr.typing.SingleScalarVolume] = None


def vols_in_same_space(
    x: torch.Tensor,
    x_affine: mrinr.typing.SingleHomogeneousAffine3D,
    y: torch.Tensor,
    y_affine: mrinr.typing.SingleHomogeneousAffine3D,
    atol: float = 1e-4,
) -> bool:
    match = (tuple(x.shape) == tuple(y.shape)) and torch.isclose(
        x_affine, y_affine, atol=atol
    ).all().bool().item()
    return match


def load_dwi_subject_data(
    dataset_table: pd.DataFrame,
    dataset_dirs: dict[str, Path] | None,
    device: torch.device,
) -> list[DWISubjectData]:
    subject_data_list = []
    for _, row in dataset_table.iterrows():
        subj_data = collections.defaultdict(lambda: None)
        dataset_name = str(row["dataset_name"])
        subj_data["dataset_name"] = dataset_name
        subj_data["subj_id"] = str(row["subj_id"])
        subj_data["session_id"] = str(row["session_id"])
        subj_data["run_id"] = str(row["run_id"])
        subj_data["dwi_idx"] = str(row["dwi_idx"])
        subj_data["pe_dir"] = PE_DIR_ALIASES[str(row["pe_dir"]).lower()]
        subj_data["total_readout_time_s"] = float(row["total_readout_time_s"])

        # If dataset_dirs is None, assume all paths in the dataset table are absolute.
        if dataset_dirs is None:
            dataset_dir = Path("/").resolve()
        else:
            dataset_dir = dataset_dirs[dataset_name]

        # Load b0 volume and associated data.
        b0_d = mrinr.data.io.load_vol(dataset_dir / row["dwi"], ensure_channel_dim=True)
        subj_data["b0"] = b0_d["vol"].to(torch.float32)
        subj_data["b0_affine"] = b0_d["affine"].to(torch.float32)
        subj_data["b0_scanner_coord_grid"] = mrinr.coords.affine_coord_grid(
            subj_data["b0_affine"], subj_data["b0"].shape[1:]
        ).to(torch.float32)
        b0_mask_d = mrinr.data.io.load_vol(
            dataset_dir / row["dwi_mask"], ensure_channel_dim=True
        )
        subj_data["b0_mask"] = b0_mask_d["vol"].bool()

        # Load T1w anatomical data.
        t1w_d = mrinr.data.io.load_vol(
            dataset_dir / row["t1w_reg_dwi"], ensure_channel_dim=True
        )
        subj_data["t1w"] = t1w_d["vol"].to(torch.float32)
        subj_data["t1w_affine"] = t1w_d["affine"].to(torch.float32)
        subj_data["t1w_coord_grid"] = mrinr.coords.affine_coord_grid(
            subj_data["t1w_affine"], subj_data["t1w"].shape[1:]
        ).to(torch.float32)
        t1w_mask_d = mrinr.data.io.load_vol(
            dataset_dir / row["t1w_mask"], ensure_channel_dim=True
        )
        subj_data["t1w_mask"] = t1w_mask_d["vol"].bool()

        # Load the topup displacement field for ground truth comparison, if given.
        if not pd.isna(row["topup_displacement_hz"]):
            topup_disp_d = mrinr.data.io.load_vol(
                dataset_dir / row["topup_displacement_hz"], ensure_channel_dim=True
            )
            subj_data["topup_displace_hz"] = topup_disp_d["vol"].to(torch.float32)
        else:
            print(
                f"Warning: topup_displacement_hz is missing for subject {subj_data['subj_id']}",
                f"in dataset {dataset_name}.",
            )
            subj_data["topup_displace_hz"] = None
        # Load the topup corrected b0 volume for ground truth comparison, if given.
        if not pd.isna(row["topup_corrected_dwi"]):
            topup_b0_d = mrinr.data.io.load_vol(
                dataset_dir / row["topup_corrected_dwi"], ensure_channel_dim=True
            )
            subj_data["topup_corrected_b0"] = topup_b0_d["vol"].to(torch.float32)
        else:
            print(
                f"Warning: topup_corrected_dwi is missing for subject {subj_data['subj_id']}",
                f"in dataset {dataset_name}.",
            )
            subj_data["topup_corrected_b0"] = None
            topup_b0_d = None
        # Load the susceptibility atlas warped to subject space, if available.
        if not pd.isna(row["suscept_atlas_mm_dir_ap"]):
            try:
                sd_atlas_d = mrinr.data.io.load_vol(
                    dataset_dir / row["suscept_atlas_mm_dir_ap"],
                    ensure_channel_dim=True,
                )
                subj_data["suscept_atlas_mm"] = sd_atlas_d["vol"].to(torch.float32)
                # Flip displacement field direction from ap to pa.
                if PE_DIR_ALIASES[row["pe_dir"]] == "pa":
                    subj_data["suscept_atlas_mm"] *= -1.0
            except Exception as e:
                print(
                    f"Warning: Could not load suscept_atlas_mm for subject "
                    f"{subj_data['subj_id']} in dataset {dataset_name}."
                )
                print(f"  Exception: {e}")
                subj_data["suscept_atlas_mm"] = None
                sd_atlas_d = None
        else:
            print(
                f"Warning: suscept_atlas_mm is missing for subject "
                f"{subj_data['subj_id']} in dataset {dataset_name}."
            )
            subj_data["suscept_atlas_mm"] = None
            sd_atlas_d = None

        # Perform assert checks on loaded data.
        assert vols_in_same_space(
            subj_data["b0"],
            subj_data["b0_affine"],
            subj_data["b0_mask"],
            b0_mask_d["affine"],
        ), "b0 and b0_mask are not in the same space."

        if not vols_in_same_space(
            subj_data["t1w"],
            subj_data["t1w_affine"],
            subj_data["t1w_mask"],
            t1w_mask_d["affine"],
        ):
            # Resample t1w_mask to t1w space.
            print("Resampling t1w_mask to t1w space...")
            subj_data["t1w_mask"] = mrinr.grid_resample_scipy(
                subj_data["t1w_mask"],
                affine_x_el2coords=t1w_mask_d["affine"],
                sample_coords=mrinr.coords.affine_coord_grid(
                    subj_data["t1w_affine"].squeeze(0), subj_data["t1w"].shape[1:]
                ),
                order=0,
                padding_mode="nearest",
            )

        # Resample susceptibility atlas, if available.
        if subj_data["suscept_atlas_mm"] is not None:
            if not vols_in_same_space(
                subj_data["suscept_atlas_mm"],
                sd_atlas_d["affine"],
                subj_data["b0"],
                subj_data["b0_affine"],
            ):
                print("Resampling suscept_atlas_mm to b0 space...")
                subj_data["suscept_atlas_mm"] = mrinr.grid_resample_scipy(
                    subj_data["suscept_atlas_mm"],
                    affine_x_el2coords=sd_atlas_d["affine"],
                    sample_coords=mrinr.coords.affine_coord_grid(
                        subj_data["b0_affine"].squeeze(0), subj_data["b0"].shape[1:]
                    ),
                    order=1,
                    padding_mode="nearest",
                )

        if subj_data["topup_corrected_b0"] is not None:
            assert vols_in_same_space(
                subj_data["b0"],
                subj_data["b0_affine"],
                subj_data["topup_corrected_b0"],
                topup_b0_d["affine"],
            ), "b0 and topup_corrected_b0 are not in the same space."

        # Validate that all affines are in RAS orientation.
        for n, affine in [
            ("b0", subj_data["b0_affine"]),
            ("t1w", subj_data["t1w_affine"]),
        ]:
            ax_orient = mrinr.coords.get_neuro_affine_orientation_code(affine)
            assert (
                ax_orient == "RAS"
            ), f"{n} affine is not in RAS orientation: {ax_orient}"

        # Create grid fov and min coordinate tensors for normalizing coordinates to
        # [0, 1] range.
        subj_data["b0_fov"] = torch.abs(
            torch.diff(
                mrinr.coords.el_bb_from_shape(
                    subj_data["b0_affine"], subj_data["b0"].shape[1:]
                ),
                dim=0,
            )
            .float()
            .flatten()
        )
        # Min coordinate in scanner space, should be in 0 index for RAS volumes.
        subj_data["b0_min_coord"] = (
            subj_data["b0_scanner_coord_grid"][0, 0, 0]
            .to(subj_data["b0_fov"])
            .flatten()
        )
        subj_data["t1w_fov"] = torch.abs(
            torch.diff(
                mrinr.coords.el_bb_from_shape(
                    subj_data["t1w_affine"], subj_data["t1w"].shape[1:]
                ),
                dim=0,
            )
            .float()
            .flatten()
        )
        # Min coordinate in scanner space, should be in 0 index for RAS volumes.
        subj_data["t1w_min_coord"] = (
            subj_data["t1w_coord_grid"][0, 0, 0].to(subj_data["t1w_fov"]).flatten()
        )

        # Move all loaded data to the target device.
        subj_data = mrinr.utils.dict_tensor_pin_to(subj_data, device, pin=True)
        subject_data_list.append(DWISubjectData(**subj_data))
    return subject_data_list


def central_diff_det_j(
    displacement_field_mm: mrinr.typing.CoordGrid3D, spacing: torch.Tensor, pe_dir
) -> mrinr.typing.ScalarVolume:
    pe_dir = PE_DIR_ALIASES[pe_dir]
    if pe_dir in ("ap", "pa"):
        phi = displacement_field_mm
        # Only need the component of the displacement field in the PE direction.
        if phi.ndim == 5:
            phi = phi[..., 1]
        if spacing.ndim == 2:
            s = spacing[:, 1:2]
        else:
            s = spacing
        # Pad in the phase encoding direction (y) and use central differences to
        # compute the derivative.
        p = (0, 0, 1, 1, 0, 0)
        phi = torch.nn.functional.pad(phi, p, mode="reflect")
        dphi_da = (phi[..., 2:, :] - phi[..., :-2, :]) / (2 * s.reshape(1, 1, -1, 1))
        # Add a channel dimension.
        dphi_da.unsqueeze_(1)
    else:
        raise NotImplementedError(
            "central_diff_det_j only supports 'ap' and 'pa' phase encoding directions."
        )

    # When displacement is only in one direction, the Jacobian determinant of the
    # displacement field is equal to 1 + the derivative of the displacement in that
    # direction. See
    # Liu et al., "Improving distortion correction for isotropic high-resolution 3D
    # diffusion MRI by optimizing Jacobian modulation," 2021.
    return 1 + dphi_da


# Data preprocessing utilities.
def scale_vol(
    x: mrinr.typing.SingleScalarVolume,
    winsorize_quantiles: tuple[float, float] = (0.005, 0.995),
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> mrinr.typing.SingleScalarVolume:
    x_ = einops.rearrange(x, "c x y z -> (x y z) c")
    q = torch.quantile(
        x_,
        torch.as_tensor(winsorize_quantiles).to(x.device),
        dim=0,
        keepdim=True,
    )
    x_scaled = torch.clamp(x_, min=q[0], max=q[1])
    min_, max_ = feature_range

    x_std = (x_ - x_.min(dim=0, keepdim=True).values) / (
        x_.max(dim=0, keepdim=True).values - x_.min(dim=0, keepdim=True).values
    )
    x_scaled = x_std * (max_ - min_) + min_
    x_scaled = einops.rearrange(
        x_scaled, "(x y z) c-> c x y z", x=x.shape[-3], y=x.shape[-2], z=x.shape[-1]
    )
    return x_scaled


@torch.no_grad()
def blur_mask(
    mask: mrinr.typing.SingleMaskVolume,
    sigma_mm: torch.Tensor,
    spacing: torch.Tensor,
    mode="nearest",
    cval=0,
    truncate=4.0,
) -> mrinr.typing.SingleScalarVolume:
    if (torch.as_tensor(sigma_mm) <= 0.0).all():
        # No blurring needed.
        return mask.to(torch.float32)
    m = torch.as_tensor(mask).cpu().bool().numpy().astype(np.float32)
    if mask.ndim == 4:
        m = np.squeeze(m, axis=0)
    sigma_vox = (sigma_mm / spacing).cpu().numpy().astype(float)
    m_blurred = skimage.filters.gaussian(
        m, sigma=sigma_vox, mode=mode, cval=cval, truncate=truncate, preserve_range=True
    )

    m_blurred = torch.as_tensor(m_blurred, device=mask.device)
    if mask.ndim == 4:
        m_blurred = m_blurred.unsqueeze(0)
    # Scale the blurred mask such that the lowest value in the original mask is 1.0.
    edge_max = torch.maximum(
        m_blurred[~mask.bool()].max(), torch.tensor(1e-6, device=mask.device)
    )
    m_blurred = m_blurred / edge_max
    m_blurred[mask.bool()] = 1.0

    return m_blurred


@torch.no_grad()
def downsample_to_target_antialiased(
    x: mrinr.typing.SingleScalarVolume,
    affine_x_el2coords: mrinr.typing.SingleHomogeneousAffine3D,
    target_spatial_shape: tuple[int, ...],
    target_affine_el2coords: mrinr.typing.SingleHomogeneousAffine3D,
    # Scale sigma beyond Nyquist, in voxels; typically in [1.0, 1.6].
    sigma_vox_scale: float = 1.0,
    truncate: float = 3.0,
    **grid_resample_kwargs,
) -> mrinr.typing.SingleScalarVolume:
    orig_shape = tuple(x.shape)
    if affine_x_el2coords.ndim != target_affine_el2coords.ndim:
        raise ValueError(
            f"Input affine_x_el2coords and target_affine_el2coords must have the "
            f"same number of dimensions, got {affine_x_el2coords.ndim} and "
            f"{target_affine_el2coords.ndim}"
        )
    if x.ndim == 4:
        x = x.unsqueeze(0)  # Add batch dim.
    if affine_x_el2coords.ndim == 2:
        affine_x_el2coords = affine_x_el2coords.unsqueeze(0)
        target_affine_el2coords = target_affine_el2coords.unsqueeze(0)
    if x.shape[0] != 1:
        raise ValueError(
            f"Input volume x must have a batch dimension of size 1 if 4D, got {tuple(x.shape)}"
        )
    elif affine_x_el2coords.shape[0] != 1:
        raise ValueError(
            f"Input affine_x_el2coords must have a batch dimension of size 1 if 3D, got {tuple(affine_x_el2coords.shape)}"
        )
    elif target_affine_el2coords.shape[0] != 1:
        raise ValueError(
            f"Input target_affine_el2coords must have a batch dimension of size 1 if 3D, got {tuple(target_affine_el2coords.shape)}"
        )

    target_spacing = (
        mrinr.coords.spacing(target_affine_el2coords).flatten().cpu().numpy()
    )
    orig_spacing = mrinr.coords.spacing(affine_x_el2coords).flatten().cpu().numpy()

    # Compute sigma for Gaussian antialiasing filter.
    sigma_vox = sigma_vox_scale * 0.5 * (target_spacing / orig_spacing)
    print(f"{orig_spacing=}, {target_spacing=}")
    print(f"Downsampling with sigma_vox: {sigma_vox}")
    x_blurred = skimage.filters.gaussian(
        image=x.cpu().squeeze(0).numpy(),
        sigma=sigma_vox,
        mode="reflect",
        truncate=truncate,
        preserve_range=True,
        channel_axis=0,
    )
    x_blurred = torch.from_numpy(x_blurred).to(x).unsqueeze(0)

    # Resample to target coordinates.
    y = mrinr.grid_resample(
        x_blurred,
        affine_x_el2coords=affine_x_el2coords,
        sample_coords=mrinr.coords.affine_coord_grid(
            target_affine_el2coords, target_spatial_shape[-3:]
        ),
        **grid_resample_kwargs,
    )

    if len(orig_shape) == 4:
        y = y.squeeze(0)  # Remove batch dim.
    return y


@torch.no_grad()
def resize_antialiased(
    x: mrinr.typing.SingleScalarVolume,
    affine_x_el2coords: mrinr.typing.SingleHomogeneousAffine3D,
    target_spatial_shape: tuple[int, ...],
    # Scale sigma beyond Nyquist, in voxels; typically in [1.0, 1.6].
    sigma_vox_scale: float = 1.0,
    truncate: float = 3.0,
    **grid_resample_kwargs,
) -> tuple[mrinr.typing.SingleScalarVolume, mrinr.typing.SingleHomogeneousAffine3D]:
    orig_shape = tuple(x.shape)
    if x.ndim == 4:
        x = x.unsqueeze(0)  # Add batch dim.
    if affine_x_el2coords.ndim == 2:
        affine_x_el2coords = affine_x_el2coords.unsqueeze(0)
    if x.shape[0] != 1:
        raise ValueError(
            f"Input volume x must have a batch dimension of size 1 if 4D, got {tuple(x.shape)}"
        )
    elif affine_x_el2coords.shape[0] != 1:
        raise ValueError(
            f"Input affine_x_el2coords must have a batch dimension of size 1 if 3D, got {tuple(affine_x_el2coords.shape)}"
        )
    orig_spatial_shape = tuple(x.shape[-3:])

    # Compute sigma for Gaussian antialiasing filter.
    sigma_vox = (
        sigma_vox_scale
        * 0.5
        * (np.array(target_spatial_shape) / np.array(orig_spatial_shape))
    )
    print(f"Resizing with sigma_vox: {sigma_vox}")
    x_blurred = skimage.filters.gaussian(
        image=x.cpu().squeeze(0).numpy(),
        sigma=sigma_vox,
        mode="reflect",
        truncate=truncate,
        preserve_range=True,
        channel_axis=0,
    )
    x_blurred = torch.from_numpy(x_blurred).to(x).unsqueeze(0)

    # Resample to target spatial shape.
    y, affine_y = mrinr.resize(
        x_blurred,
        affine_x_el2coords=affine_x_el2coords,
        target_spatial_shape=target_spatial_shape,
        centered=True,
        **grid_resample_kwargs,
    )

    if len(orig_shape) == 4:
        y = y.squeeze(0)  # Remove batch dim.
        affine_y = affine_y.squeeze(0)
    return y, affine_y


def ras_displacement_field_vox2susceptibility_field_hz(
    displacement_field_vox,
    readout_time_s: float,
    pe_dir: str,
):
    pe_dir = pe_dir.lower()
    # Displacement occurs in the axis-aligned direction, so an AP PE direction (which
    # goes from + to - in scanner space) will be negated.
    if PE_DIR_ALIASES[pe_dir] == "ap":
        displacement_field_vox = -displacement_field_vox
    elif PE_DIR_ALIASES[pe_dir] == "pa":
        pass
    else:
        raise ValueError(f"Invalid pe_dir: {pe_dir}")
    return displacement_field_vox / readout_time_s


def susceptibility_field_hz2ras_displacement_field_vox(
    displacement_field_hz,
    readout_time_s: float,
    pe_dir: str,
):
    pe_dir = pe_dir.lower()
    d = displacement_field_hz * readout_time_s
    # Displacement occurs in the axis-aligned direction, so an AP PE direction (which
    # goes from + to - in scanner space) will be negated.
    if PE_DIR_ALIASES[pe_dir] == "ap":
        d = -d
    elif PE_DIR_ALIASES[pe_dir] == "pa":
        pass
    else:
        raise ValueError(f"Invalid pe_dir: {pe_dir}")
    return d
