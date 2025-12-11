# -*- coding: utf-8 -*-
import collections
import math
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
    # Topup-corrected ground truth data
    topup_displace_hz: mrinr.typing.SingleCoordGrid3D
    topup_corrected_b0: mrinr.typing.SingleScalarVolume
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
    dataset_dirs: dict[str, Path],
    device: torch.device,
) -> list[DWISubjectData]:
    subject_data_list = []
    for _, row in dataset_table.iterrows():
        subj_data = collections.defaultdict(lambda: None)
        dataset_name = row["dataset_name"]
        subj_data["dataset_name"] = dataset_name
        subj_data["subj_id"] = row["subj_id"]
        subj_data["session_id"] = row["session_id"]
        subj_data["run_id"] = row["run_id"]
        subj_data["dwi_idx"] = row["dwi_idx"]
        subj_data["pe_dir"] = PE_DIR_ALIASES[row["pe_dir"]]
        subj_data["total_readout_time_s"] = float(row["total_readout_time_s"])

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

        # Load the topup displacement field for ground truth comparison.
        topup_disp_d = mrinr.data.io.load_vol(
            dataset_dir / row["topup_displacement_hz"], ensure_channel_dim=True
        )
        subj_data["topup_displace_hz"] = topup_disp_d["vol"].to(torch.float32)
        # Load the topup corrected b0 volume for ground truth comparison.
        topup_b0_d = mrinr.data.io.load_vol(
            dataset_dir / row["topup_corrected_dwi"], ensure_channel_dim=True
        )
        subj_data["topup_corrected_b0"] = topup_b0_d["vol"].to(torch.float32)
        # Load the susceptibility atlas warped to subject space, if available.
        if row["suscept_atlas_mm_dir_ap"] is not None:
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


class WeightedNMIParzenLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        num_bins: int = 32,
        sigma_ratio: float = 0.5,
        reduction: str = "mean",
        eps: float = 1e-7,
        norm_mi: bool = True,
        norm_images: bool = True,
    ):
        super().__init__(reduction=reduction)
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        self.spatial_dims = 3
        self.num_bins = int(num_bins)
        # shape (num_bins)
        bin_centers = torch.linspace(0.0, 1.0, self.num_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_centers: torch.Tensor

        sigma = torch.mean(self.bin_centers[1:] - self.bin_centers[:-1]) * sigma_ratio
        self.register_buffer("sigma", sigma)
        self.sigma: torch.Tensor
        self.eps = eps
        self.norm_mi = norm_mi
        self.norm_images = norm_images

    @staticmethod
    def spatial_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Min-max normalize x to [0, 1] along spatial dimensions."""
        x_min = einops.reduce(x, "b c x y z -> b c 1 1 1", "min")
        x_max = einops.reduce(x, "b c x y z -> b c 1 1 1", "max")
        x_normalized = (x - x_min) / (x_max - x_min + eps)
        return x_normalized

    def parzen_windowing_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """Parzen Gaussian weighting function to approximate histogram differentiably.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape B x C x X x Y x Z.

            *NOTE* Input volume must have intensities normalized to [0, 1]. Intensities
            will be clamped to [0, 1] internally.
        Returns
        -------
        torch.Tensor
            Discrete probability distributions for each voxel, shape
            (B*C) x (X*Y*Z) x num_bins. Distributions are not averaged over spatial
            dimensions.
        """
        y = torch.clamp(x, 0.0, 1.0)
        # Move independent dims to the front, and merge spatial dims into a sampling
        # dimension, plus a singleton dimension for broadcasting hist. bins.
        y = einops.rearrange(y, "b c ... -> (b c) (...) 1")
        w_parzen = (1 / (self.sigma * math.sqrt(2 * math.pi))) * torch.exp(
            -0.5 * ((y - self.bin_centers[None, None, :]) / self.sigma) ** 2
        )
        # Normalize over bins.
        p = w_parzen / torch.maximum(
            torch.sum(w_parzen, dim=-1, keepdim=True),
            w_parzen.new_tensor([self.eps]),
        )
        # Wait to average over sampling dimensions until estimating the joint histogram.
        return p

    def weighted_nmi(
        self,
        pred: mrinr.typing.ScalarVolume,
        target: mrinr.typing.ScalarVolume,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
        norm_mi: bool = True,
        normalize_images: bool = True,
    ) -> torch.Tensor:
        """Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
            weight_mask: the shape should be B[1DHW] or B[NDHW], optional.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(
                f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})"
            )

        if normalize_images:
            # Normalize pred and target to [0, 1] along spatial dims, keeping batch and
            # channel dims independent.
            x = self.spatial_normalize(pred, eps=self.eps)
            y = self.spatial_normalize(target, eps=self.eps)
        else:
            x = pred
            y = target

        # Parzen windowing, without averaging over samples. Outputs are
        # shape (B*C)(D*H*W)(num_bins).
        p_pred = self.parzen_windowing_gaussian(x)
        p_target = self.parzen_windowing_gaussian(y)

        # Estimate joint histogram P_pred,target weighted by the weight mask if
        # provided.
        if weight_mask is not None:
            w = einops.rearrange(
                weight_mask.expand_as(pred), "b c x y z -> (b c) (x y z) 1 1"
            )
            # Normalize the weight mask to be in the range [0, 1].
            w = w / torch.maximum(
                w.max(dim=1, keepdim=True).values, w.new_tensor([self.eps])
            )
        else:
            w = 1.0
        p_joint = (
            w * einops.einsum(p_pred, p_target, "bc xyz i, bc xyz j -> bc xyz i j")
        ).sum(1)
        # Normalize joint histogram.
        p_joint = p_joint / torch.maximum(
            p_joint.sum(dim=(-2, -1), keepdim=True),
            p_joint.new_tensor([self.eps]),
        )
        # Estimate marginal histograms.
        p_pred_marginal = p_joint.sum(dim=-1)
        p_target_marginal = p_joint.sum(dim=-2)

        # Compute entropies.
        H_pred = -torch.sum(
            p_pred_marginal
            * torch.log(
                torch.maximum(p_pred_marginal, p_pred_marginal.new_tensor([self.eps]))
            ),
            dim=-1,
        )
        H_target = -torch.sum(
            p_target_marginal
            * torch.log(
                torch.maximum(
                    p_target_marginal, p_target_marginal.new_tensor([self.eps])
                )
            ),
            dim=-1,
        )
        H_joint = -torch.sum(
            p_joint * torch.log(torch.maximum(p_joint, p_joint.new_tensor([self.eps]))),
            dim=(-2, -1),
        )

        # NMI or plain MI.
        if norm_mi:
            mi = (H_pred + H_target) / torch.maximum(
                H_joint, H_joint.new_tensor([self.eps])
            )
        else:
            mi = H_pred + H_target - H_joint

        if self.reduction == "sum":
            # sum over the batch and channel ndims
            r = torch.sum(mi)
        elif self.reduction == "none":
            # No reduction of independent dims.
            r = einops.rearrange(mi, "(b c) -> b c", b=pred.shape[0], c=pred.shape[1])
        elif self.reduction == "mean":
            # average over the batch and channel ndims
            r = torch.mean(mi)
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction}, "
                'available options are ["mean", "sum", "none"].'
            )
        return r

    def forward(
        self,
        pred: mrinr.typing.ScalarVolume,
        target: mrinr.typing.ScalarVolume,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
    ) -> torch.Tensor:
        # Loss is negative NMI.
        return -self.weighted_nmi(
            pred,
            target,
            weight_mask=weight_mask,
            norm_mi=self.norm_mi,
            normalize_images=self.norm_images,
        )


class NCC(torch.nn.modules.loss._Loss):
    # Normalized Cross Correlation
    # Taken from <https://github.com/MIAGroupUT/IDIR/blob/main/objectives/ncc.py>,
    # which itself was taken from <https://github.com/BDdeVos/TorchIR>
    class _StableStd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            assert tensor.numel() > 1
            ctx.tensor = tensor.detach()
            res = torch.std(tensor).detach()
            ctx.result = res.detach()
            return res

        @staticmethod
        def backward(ctx, grad_output):
            tensor = ctx.tensor.detach()
            result = ctx.result.detach()
            e = 1e-6
            assert tensor.numel() > 1
            return (
                (2.0 / (tensor.numel() - 1.0))
                * (grad_output.detach() / (result.detach() * 2 + e))
                * (tensor.detach() - tensor.mean().detach())
            )

    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.forward = self.metric

    def ncc(self, x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        stablestd = self._StableStd.apply
        std = stablestd(x1) * stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    def metric(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return -self.ncc(fixed, warped)


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


# Data preprocessing
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
