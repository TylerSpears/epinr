# -*- coding: utf-8 -*-

import einops
import torch
import torch.nn.functional as F

import mrinr

# Constants for streamline status
# Streamline is still being tracked.
CONTINUE = 1
# Streamline terminates at this point, i.e. this is the final point of the streamline.
# A streamline may only have one STOP point in its trajectory.
STOP = 0
# Streamline was stopped in a previous iteration.
PREV_STOP = 3
# Streamline is too short, but may be combined with its antipodal counterpart. Indicates
# the streamline validity is to be determined.
STOP_SHORT = 4
PREV_STOP_SHORT = 5
# Generic valid streamline; it is not invalid.
VALID = 2
# Streamline is wholly invalid, e.g. it should be discarded in its entirety.
INVALID = -1
# Streamline is invalid, and its antipodal counterpart should also be removed.
INVALID_ANTIPODAL = -2
# Undefined status, may be necessary to throw an exception.
UNDEFINED = -10


def to_continue_mask(streamline_status: torch.Tensor) -> torch.Tensor:
    return streamline_status == CONTINUE


def merge_status(
    status_tm1: torch.Tensor,
    *partial_statuses_t,
) -> torch.Tensor:
    # b x num_stop_conditions
    statuses_t = torch.stack(tuple(partial_statuses_t), dim=1)
    tmp_status = torch.ones_like(status_tm1) * UNDEFINED
    # If all partial statues are CONTINUE, then the current status is CONTINUE.
    tmp_status[(statuses_t == CONTINUE).all(dim=1)] = CONTINUE
    # If any partial statues are STOP, then the current status is STOP.
    tmp_status[(statuses_t == STOP).any(dim=1)] = STOP
    # If the previous status was STOP or PREV_STOP, then the current status is PREV_STOP.
    tmp_status[(status_tm1 == STOP) | (status_tm1 == PREV_STOP)] = PREV_STOP
    # Similar for STOP_SHORT and PREV_STOP_SHORT.
    tmp_status[(status_tm1 == STOP_SHORT) | (status_tm1 == PREV_STOP_SHORT)] = (
        PREV_STOP_SHORT
    )
    # If any statuses are INVALID, then the current status is INVALID.
    tmp_status[(statuses_t == INVALID).any(dim=1) | (status_tm1 == INVALID)] = INVALID
    tmp_status[
        (statuses_t == INVALID_ANTIPODAL).any(dim=1) | (status_tm1 == INVALID_ANTIPODAL)
    ] = INVALID_ANTIPODAL

    # Use an assert so that the statement may be skipped in python's `-O` mode.
    assert not (tmp_status == UNDEFINED).any(), (
        "Some streamlines have an UNDEFINED status after merging. This may indicate an "
        + "error in stopping condition or merging logic."
    )

    return tmp_status


def streamline_min_len_mm(
    current_status: torch.Tensor,
    streamline_len_mm_t: torch.Tensor,
    min_len_mm: float,
    invalidate_short: bool = True,
) -> torch.Tensor:
    """Ensure streamlines still being tracked do not fall below a minimum length.

    If the streamline is still being tracked and its length is below the minimum length,
    it is marked as STOP_SHORT or INVALID.

    Parameters
    ----------
    current_status : torch.Tensor
        Current status Tensor of the streamlines at timepoint t.
    streamline_len_mm_t : torch.Tensor
        Length of the streamlines at timepoint t in mm.
    min_len_mm : float
        Minimum inclusive length of the streamlines in mm.
    invalidate_short : bool, optional
        If True, streamlines that are stopped and below the minimum length are marked as
        INVALID. If False, they are marked as STOP_SHORT. Default is True.

    Returns
    -------
    torch.Tensor
        New status Tensor of the streamlines at timepoint t, where streamlines that are
        stopped and below the minimum length are marked as STOP_SHORT or INVALID.
    """
    status = torch.where(
        (current_status == STOP) & (streamline_len_mm_t < min_len_mm),
        INVALID if invalidate_short else STOP_SHORT,
        current_status,
    )
    # Always invalidate streamlines with negative lengths.
    status = torch.where(
        streamline_len_mm_t < 0.0,
        INVALID,
        status,
    )
    return status


def streamline_max_len_mm(
    current_status: torch.Tensor,
    streamline_len_mm_tp1: torch.Tensor,
    max_len_mm: float,
    truncate_long: bool = True,
) -> torch.Tensor:
    """Ensure streamlines are truncated or invalidated if they exceed a maximum length.

    Parameters
    ----------
    current_status : torch.Tensor
        Current status Tensor of the streamlines at timepoint t.
    streamline_len_mm_tp1 : torch.Tensor
        Length of the streamlines at timepoint t+1 in mm.
    max_len_mm : float
        Maximum inclusive length of the streamlines in mm.
    truncate_long : bool, optional
        If True, streamlines that exceed the maximum length are marked as STOP. If False,
        they are marked as INVALID. Default is True.

    Returns
    -------
    torch.Tensor
        New status Tensor of the streamlines at timepoint t, where streamlines are
        marked as STOP or INVALID if they exceed the maximum length.
    """

    status = torch.where(
        (current_status == CONTINUE) & (streamline_len_mm_tp1 > max_len_mm),
        STOP if truncate_long else INVALID,
        current_status,
    )
    # Always invalidate streamlines with negative lengths.
    status = torch.where(streamline_len_mm_tp1 < 0.0, INVALID, status)
    return status


def gfa_threshold(
    gfa_min_threshold: float,
    sh_coeff: torch.Tensor,
) -> torch.Tensor:
    gen_fa = mrinr.vols.odf.gfa(sh_coeff)
    status = torch.where(
        gen_fa < gfa_min_threshold,
        STOP,
        CONTINUE,
    )
    return status


def vol_sample_threshold(
    vol: torch.Tensor,
    affine_vox2real: torch.Tensor,
    sample_coords: torch.Tensor,
    sample_min: float = -torch.inf,
    sample_max: float = torch.inf,
    failure_status: int = STOP,
    **sample_vol_kwargs,
) -> torch.Tensor:
    c = einops.rearrange(sample_coords, "b coord -> 1 b 1 1 coord")
    affine = affine_vox2real.unsqueeze(0)
    v = vol
    if v.ndim == 3:
        v = v.unsqueeze(0)
    if v.ndim == 4:
        v = v.unsqueeze(0)
    samples = mrinr.grid_resample(
        v, sample_coords=c, affine_x_el2coords=affine, **sample_vol_kwargs
    )
    # Reshape to be batch x n_channels.
    samples = einops.rearrange(samples, "1 c b 1 1 -> b c")
    # s = torch.ones(samples.shape[0], dtype=torch.int, device=samples.device)
    status = torch.where(
        ((samples < sample_min) | (samples > sample_max)).all(dim=-1),
        failure_status,
        CONTINUE,
    )
    return status


def scalar_threshold(
    x: torch.Tensor,
    min_thresh: float = -torch.inf,
    max_thresh: float = torch.inf,
) -> torch.Tensor:
    # status = torch.ones(x.shape[0], dtype=torch.int, device=x.device)
    status = torch.where(
        (x < min_thresh) | (x > max_thresh) | x.isnan(), STOP, CONTINUE
    )
    return status


def angular_threshold(
    angle_x: torch.Tensor,
    angle_y: torch.Tensor,
    max_radians: float,
) -> torch.Tensor:
    cos_sim = F.cosine_similarity(angle_x, angle_y, dim=-1)
    cos_sim.clamp_(min=mrinr.coords.MIN_COS_SIM, max=mrinr.coords.MAX_COS_SIM)
    arc_len = torch.arccos(cos_sim)
    # Replace arc lengths with zero if the cosine similarity is close to 1.0.
    arc_len = torch.where(
        torch.isclose(cos_sim, torch.ones_like(cos_sim) * mrinr.coords.MAX_COS_SIM),
        torch.zeros_like(arc_len),
        arc_len,
    )

    # status = torch.ones(arc_len.shape[0], dtype=torch.int, device=arc_len.device)
    status = torch.where(arc_len > max_radians, STOP, CONTINUE)
    return status
