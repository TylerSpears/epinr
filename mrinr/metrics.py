# -*- coding: utf-8 -*-
# Functions for measuring task performance.
import shlex
import tempfile
from pathlib import Path
from typing import Optional

import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import mrinr
from mrinr.coords import MAX_UNIT_ARC_LEN, MIN_UNIT_ARC_LEN

# Median normalized ODF peak amplitude, empirically determined from several HCP DWIs.
# Useful for getting a sense of what an "average" normalized ODF peak value should be.
HCP_MEDIAN_NORM_PEAK = 0.0073


def mse_batchwise_masked(y_pred, y, mask):
    masked_y_pred = y_pred.clone()
    masked_y = y.clone()
    m = mask.expand_as(masked_y)
    masked_y_pred[~m] = torch.nan
    masked_y[~m] = torch.nan
    se = F.mse_loss(masked_y_pred, masked_y, reduction="none")
    mse = torch.nanmean(se, dim=1, keepdim=True)
    return mse


@torch.no_grad()
def sphere_jensen_shannon_distance(
    input_odf_coeffs, target_odf_coeffs, mask, theta, phi, sh_order=8
):
    epsilon = 1e-5
    batch_size = input_odf_coeffs.shape[0]
    if batch_size != 1:
        raise NotImplementedError("ERROR: Batch size != 1 not implemented or tested")
    input_sphere_samples = mrinr.vols.odf.sample_sphere_coords(
        input_odf_coeffs * mask,
        theta=theta,
        phi=phi,
        sh_order=sh_order,
        sh_order_dim=1,
        mask=mask,
        force_nonnegative=True,
    )
    n_sphere_samples = input_sphere_samples.shape[1]
    sphere_mask = mask.expand_as(input_sphere_samples)
    # Mask and reshape to (n_vox x batch_size) x n_prob_samples
    input_sphere_samples = einops.rearrange(
        input_sphere_samples[sphere_mask],
        "(b s v) -> (b v) s",
        b=batch_size,
        s=n_sphere_samples,
    )
    # Normalize to sum to 1.0, as a probability density.
    input_sphere_samples /= torch.maximum(
        torch.sum(input_sphere_samples, dim=1, keepdim=True),
        input_odf_coeffs.new_zeros(1) + epsilon,
    )
    target_sphere_samples = mrinr.vols.odf.sample_sphere_coords(
        target_odf_coeffs * mask,
        theta=theta,
        phi=phi,
        sh_order=sh_order,
        sh_order_dim=1,
        mask=mask,
    )
    target_sphere_samples = einops.rearrange(
        target_sphere_samples[sphere_mask],
        "(b s v) -> (b v) s",
        b=batch_size,
        s=n_sphere_samples,
    )
    # Normalize to sum to 1.0, as a probability density.
    target_sphere_samples /= torch.maximum(
        torch.sum(target_sphere_samples, dim=1, keepdim=True),
        target_odf_coeffs.new_zeros(1) + epsilon,
    )

    Q_log_in = torch.log(input_sphere_samples.to(torch.float64))
    P_log_target = torch.log(target_sphere_samples.to(torch.float64))
    M_log = torch.log(
        (input_sphere_samples + target_sphere_samples).to(torch.float64) / 2
    )
    del input_sphere_samples, target_sphere_samples
    d_P_M = F.kl_div(M_log, P_log_target, reduction="none", log_target=True)
    # Implement batchmean per-voxel.
    # nan values from the kl divergence occur when the expected density is 0.0 and the
    # log is -inf. The 'contribution' of that element is 0 as the limit approaches 0,
    # so just adding the non-nan values should be valid.
    d_P_M = d_P_M.nansum(1, keepdim=True) / d_P_M.shape[1]

    d_Q_M = F.kl_div(M_log, Q_log_in, reduction="none", log_target=True)
    d_Q_M = d_Q_M.nansum(1, keepdim=True) / d_Q_M.shape[1]

    js_div = d_P_M / 2 + d_Q_M / 2
    js_div = einops.rearrange(js_div, "(b v) s -> (b s v)", b=batch_size, s=1)
    js_dist = torch.zeros_like(mask).to(input_odf_coeffs)
    js_dist.masked_scatter_(mask, torch.sqrt(js_div).to(torch.float32)).to(
        input_odf_coeffs
    )

    return js_dist


@torch.no_grad()
def odf_peaks_mrtrix(
    odf: torch.Tensor,
    affine: torch.Tensor,
    mask: torch.Tensor,
    n_peaks: int,
    min_amp: float,
    match_peaks_vol: Optional[torch.Tensor] = None,
    mrtrix_nthreads: Optional[int] = None,
) -> torch.Tensor:
    batch_size = odf.shape[0]
    if batch_size != 1:
        raise NotImplementedError("ERROR: Batch size != 1 not implemented")

    n_peak_channels = n_peaks * 3

    affine = affine.squeeze(0).detach().cpu().numpy()
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

    with tempfile.TemporaryDirectory(prefix="mrinr_odf_peaks_mrtrix_") as tmp_dir_name:
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

        out_peaks = out_peaks_im.get_fdata()
        out_peaks = torch.from_numpy(out_peaks).unsqueeze(0).to(odf)
    # Nans are either outside of the mask or in CSF locations with no peak, so just
    # assign a 0 vector as the peak direction.
    out_peaks.nan_to_num_(nan=0.0)

    return out_peaks


@torch.no_grad()
def waae(
    odf_pred: torch.Tensor,
    odf_gt: torch.Tensor,
    mask: torch.Tensor,
    peaks_pred: torch.Tensor,
    peaks_gt: torch.Tensor,
    n_peaks: int,
    odf_integral_theta: torch.Tensor,
    odf_integral_phi: torch.Tensor,
    fp_fn_w: float,
    sh_order: int = 8,
) -> torch.Tensor:
    batch_size = odf_pred.shape[0]
    if batch_size != 1:
        raise NotImplementedError("ERROR: Batch size != 1 not implemented")

    # Assume that all peaks are ordered by amplitude.
    # Only select the first N x 3 peaks for the ground truth, but keep all available
    # peaks in the prediction.
    max_coord_idx = 3 * n_peaks
    peaks_gt = peaks_gt[..., :max_coord_idx]
    assert peaks_pred.shape[-1] % 3 == 0
    n_peaks_pred = peaks_pred.shape[-1] // 3
    # Only select voxels where either the ground truth or the prediction has a peak. All
    # others are 0s.
    gt_has_peaks_mask = (
        ~torch.isclose(peaks_gt, peaks_gt.new_zeros(1)).all(-1)
    ) & mask.squeeze(1)
    pred_has_peaks_mask = (
        ~torch.isclose(peaks_pred, peaks_pred.new_zeros(1)).all(-1)
    ) & mask.squeeze(1)
    peaks_gt = peaks_gt[gt_has_peaks_mask | pred_has_peaks_mask]
    peaks_pred = peaks_pred[gt_has_peaks_mask | pred_has_peaks_mask]

    # Integral across the sphere for the GT odfs, for finding the "W" in "WAAE".
    # Only select voxels where the ground truth has a peak. All others are 0s.
    gt_sphere_samples = mrinr.vols.odf.sample_sphere_coords(
        odf_gt,
        theta=odf_integral_theta,
        phi=odf_integral_phi,
        mask=gt_has_peaks_mask.unsqueeze(1),
        sh_order=sh_order,
        force_nonnegative=True,
    )
    gt_sphere_samples = einops.rearrange(
        gt_sphere_samples, "b dirs x y z -> b x y z dirs"
    )
    gt_sphere_integral = gt_sphere_samples[gt_has_peaks_mask | pred_has_peaks_mask].sum(
        dim=-1, keepdims=True
    )
    gt_sphere_integral.clamp_(min=torch.finfo(gt_sphere_integral.dtype).eps)
    del gt_sphere_samples

    # Integral across the sphere for the pred odfs, for finding the "W" in "WAAE" in
    # cases of a false positive peak.
    # Only select voxels where the prediction has a peak. All others are 0s.
    # pred_sphere_samples = pitn.odf.sample_sphere_coords(
    #     odf_pred,
    #     theta=odf_integral_theta,
    #     phi=odf_integral_phi,
    #     mask=pred_has_peaks_mask.unsqueeze(1),
    #     sh_order=8,
    #     force_nonnegative=True,
    # )
    # pred_sphere_samples = einops.rearrange(
    #     pred_sphere_samples, "b dirs x y z -> b x y z dirs"
    # )
    # pred_sphere_integral = pred_sphere_samples[
    #     gt_has_peaks_mask | pred_has_peaks_mask
    # ].sum(dim=-1, keepdims=True)
    # pred_sphere_integral.clamp_(min=torch.finfo(pred_sphere_integral.dtype).eps)
    # del pred_sphere_samples

    # Reshape peaks to split along each peak.
    peaks_gt = einops.rearrange(
        peaks_gt, "vox (n_peaks coord) -> n_peaks vox coord", coord=3
    )
    peaks_pred = einops.rearrange(
        peaks_pred, "vox (n_peaks coord) -> n_peaks vox coord", coord=3
    )
    # Project all peaks into the z+ hemisphere/quadrant, as there is anitipodal
    # symmetry.
    peaks_gt = torch.where(
        (peaks_gt[..., -1, None] < 0).expand_as(peaks_gt), -peaks_gt, peaks_gt
    )
    peaks_pred = torch.where(
        (peaks_pred[..., -1, None] < 0).expand_as(peaks_pred), -peaks_pred, peaks_pred
    )

    FALSE_ARC_LEN_INDICATOR = torch.inf
    # Keep a running sum of each GT peak's waae.
    # (1, 1, x, y, z)
    running_waae = torch.zeros_like(odf_gt[:, 0:1])
    # Iterate over all peaks/"fixels" in the GT.
    for i_peak in range(n_peaks):
        peaks_gt_i = peaks_gt[i_peak]
        gt_has_peak_i_mask = ~(
            torch.isclose(peaks_gt_i, peaks_gt_i.new_zeros(1)).all(-1, keepdim=True)
        )
        # Determine the prediction peak based on the minimum arc len from the gt peak.
        arc_lens_pred_i = list()
        for j_pred_peak in range(n_peaks_pred):
            peaks_pred_ij = peaks_pred[j_pred_peak]
            pred_gt_arc_len_ij = mrinr.coords._antipodal_sym_arc_len(
                peaks_pred_ij, peaks_gt_i
            )
            pred_has_peak_ij_mask = ~(
                torch.isclose(peaks_pred_ij, peaks_pred_ij.new_zeros(1)).all(
                    -1, keepdim=True
                )
            )
            fp = (~gt_has_peak_i_mask) & pred_has_peak_ij_mask
            fn = gt_has_peak_i_mask & (~pred_has_peak_ij_mask)
            pred_gt_arc_len_ij.masked_fill_(
                fp.squeeze(1) | fn.squeeze(1), FALSE_ARC_LEN_INDICATOR
            )
            arc_lens_pred_i.append(pred_gt_arc_len_ij)
        arc_lens_pred_i = torch.stack(arc_lens_pred_i, dim=0)
        peaks_pred_i = torch.take_along_dim(
            peaks_pred,
            torch.argmin(arc_lens_pred_i.unsqueeze(-1), dim=0, keepdim=True),
            dim=0,
        )
        peaks_pred_i.squeeze_(0)
        # peaks_pred_i = torch.take_along_dim(peaks_pred, torch.argmin(arc_lens_pred_i, dim=0), dim=0)
        # peaks_pred_i = peaks_pred[torch.argmin(arc_lens_pred_i, dim=0)]
        del (
            arc_lens_pred_i,
            pred_gt_arc_len_ij,
            fp,
            fn,
            peaks_pred_ij,
        )

        pred_has_peak_i_mask = ~(
            torch.isclose(peaks_pred_i, peaks_pred_i.new_zeros(1)).all(-1, keepdim=True)
        )
        # False negatives
        pred_fn_mask = gt_has_peak_i_mask & (~pred_has_peak_i_mask)
        # False positives
        pred_fp_mask = (~gt_has_peak_i_mask) & pred_has_peak_i_mask

        peak_height_gt_i = torch.linalg.norm(peaks_gt_i, ord=2, dim=-1, keepdims=True)
        # peak_height_pred_i = torch.linalg.norm(
        #     peaks_pred_i, ord=2, dim=-1, keepdims=True
        # )

        gt_pred_arc_len = (
            mrinr.coords._antipodal_sym_arc_len(
                peaks_pred_i.to(torch.float64), peaks_gt_i.to(torch.float64)
            )
            .unsqueeze(-1)
            .to(peaks_pred_i)
        )
        # False positives and false negatives both get maximum arc length penalty.
        gt_pred_arc_len.masked_fill_(pred_fn_mask | pred_fp_mask, MAX_UNIT_ARC_LEN)
        gt_pred_arc_len.clamp_(min=MIN_UNIT_ARC_LEN, max=MAX_UNIT_ARC_LEN)

        # Amplitude of GT fixel normalized by the entire GT's volume.
        w_i = peak_height_gt_i / gt_sphere_integral
        # Zero out the weight of this fixel if this is a true negative fixel.
        w_i[(~gt_has_peak_i_mask) & (~pred_has_peak_i_mask)] = 0.0
        # # Handle false positives by assigning W to be the predicted peak height.
        # w_fp_i = peak_height_pred_i / pred_sphere_integral
        w_i = torch.where(pred_fp_mask | pred_fn_mask, fp_fn_w, w_i)

        running_waae[(gt_has_peaks_mask | pred_has_peaks_mask).unsqueeze(1)] += (
            gt_pred_arc_len * w_i
        ).squeeze(-1)
        del w_i, gt_pred_arc_len

    return running_waae
