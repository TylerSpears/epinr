# -*- coding: utf-8 -*-
# Functions and objects to transform spherical coordinates.
import collections
import functools
from typing import Tuple

import dipy
import dipy.reconst.csdeconv
import dipy.reconst.shm
import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import mrinr

MIN_COS_SIM = -1.0 + torch.finfo(torch.float32).eps
MAX_COS_SIM = 1.0 - torch.finfo(torch.float32).eps
MAX_UNIT_ARC_LEN = (torch.pi / 2) - torch.finfo(torch.float32).eps
MIN_UNIT_ARC_LEN = 0.0 + torch.finfo(torch.float32).eps
MIN_THETA = 0.0
MAX_THETA = torch.pi
MIN_PHI = 0.0
MAX_PHI = (2 * torch.pi) - torch.finfo(torch.float32).eps
AT_POLE_EPS = torch.finfo(torch.float32).eps

_ThetaPhiResult = collections.namedtuple("_ThetaPhiResult", ("theta", "phi"))


def xyz2unit_sphere_theta_phi(
    coords_xyz: torch.Tensor,
) -> _ThetaPhiResult:
    #! Inputs to this function should generally be 64-bit floats! Precision is poor for
    #! 32-bit floats.

    x = coords_xyz[..., 0]
    y = coords_xyz[..., 1]
    z = coords_xyz[..., 2]
    r = torch.linalg.norm(coords_xyz, ord=2, axis=-1)
    r[r == 0] = 1.0
    theta = torch.arccos(z / r)
    # The discontinuities of atan2 mean we have to shift and cycle some values.
    phi = torch.arctan2(y, x) % (2 * torch.pi)
    at_pole = torch.sin(theta) < AT_POLE_EPS
    # At N and S poles, y = x = 0, which would make phi undefined. However, phi is
    # arbitrary at poles in spherical coordinates, so just set to a small non-zero value
    # for avoiding potential numerical issues.
    phi = torch.where(at_pole, AT_POLE_EPS, phi)
    return _ThetaPhiResult(theta=theta, phi=phi)


def unit_sphere2xyz(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    #! Consider using 64-bit floats for function input. Precision is poor for 32-bit.
    # r = 1 on the unit sphere.
    # Theta does not "cycle back" between 0 and pi, it "bounces back" such as in
    # a sequence 0.01 -> 0.001 -> 0.0 -> 0.001 -> 0.01. This is unlike phi which
    # does cycle back: 2pi - 2eps -> 2pi - eps -> 0 -> 0 + eps ...
    # The where() handles theta > pi, and the abs() handles theta < pi.
    # theta = torch.where(
    #     theta > torch.pi,
    #     torch.pi - (theta % torch.pi),
    #     torch.abs(theta),
    # )
    # # Phi just cycles back.
    # phi = phi % (2 * torch.pi)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)


def unit_sphere_arc_len(
    theta_1: torch.Tensor,
    phi_1: torch.Tensor,
    theta_2: torch.Tensor,
    phi_2: torch.Tensor,
):
    coords_1 = unit_sphere2xyz(theta_1, phi_1)
    coords_2 = unit_sphere2xyz(theta_2, phi_2)
    cos_sim = torch.nn.functional.cosine_similarity(coords_1, coords_2, dim=-1)
    cos_sim.clamp_(min=MIN_COS_SIM, max=MAX_COS_SIM)
    arc_len = torch.arccos(cos_sim)
    arc_len.masked_fill_(torch.isclose(cos_sim, cos_sim.new_tensor([MAX_COS_SIM])), 0.0)
    return arc_len


@functools.lru_cache(maxsize=10)
def get_torch_sample_sphere_coords(
    sphere, device: torch.DeviceObjType, dtype: torch.dtype
) -> _ThetaPhiResult:
    theta = sphere.theta
    phi = sphere.phi
    theta = torch.from_numpy(theta).to(device=device, dtype=dtype, copy=True)
    phi = torch.from_numpy(phi).to(device=device, dtype=dtype, copy=True)

    return _ThetaPhiResult(theta, phi)


# def unit_sphere_arc_length(
#     theta_1: torch.Tensor,
#     phi_1: torch.Tensor,
#     theta_2: torch.Tensor,
#     phi_2: torch.Tensor,
# ) -> torch.Tensor:
#     r = 1
#     dist_squared = (
#         r**2
#         + r**2
#         - 2
#         * r
#         * r
#         * (
#             torch.sin(theta_1) * torch.sin(theta_2) * torch.cos(phi_1 - phi_2)
#             + torch.cos(theta_1) * torch.cos(theta_2)
#         )
#     )
#     dist = torch.sqrt(dist_squared)

#     return dist


@functools.lru_cache(maxsize=10)
def _adjacent_sphere_points_idx(
    theta: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert theta.shape == phi.shape
    assert theta.ndim == 1 and phi.ndim == 1
    # Get the full pairwise arc length between all points on the sphere. This matrix is
    # symmetric.
    pairwise_arc_len = unit_sphere_arc_len(
        theta[:, None], phi[:, None], theta[None], phi[None]
    )

    # Find the arc length that will determine adjacency to each point.
    # Find the closest pole idx.
    pole_idx = torch.sort(theta).indices[0]
    pole_adjacent = pairwise_arc_len[pole_idx, :]
    pole_adjacent_sorted = torch.sort(pole_adjacent)
    # Grab the arc length halfway between the length of the "closest 6" and the next
    # closest set.
    sphere_surface_point_radius = (
        pole_adjacent_sorted.values[6]
        + (pole_adjacent_sorted.values[7] - pole_adjacent_sorted.values[6]) / 2
    )

    arc_len_sorted = pairwise_arc_len.sort(1)
    # Grab indices 1-7 because we don't care about index 0 (same point, arc len ~= 0.0), and
    # we only want *up to* the closest 6 points.
    nearest_point_idx = arc_len_sorted.indices[:, 1:7]
    # Now we want only those points within the pre-determined radius. Points near the bottom
    # of the hemisphere will have fewer than 6 adjacent points.
    # !Make sure to use the mask to avoid silent indexing errors in the future!
    # We could provide invalid indices at the non-adjacent points, but that makes
    # function-writing more difficult down the line.
    nearest_point_idx_mask = arc_len_sorted.values[:, 1:7] < sphere_surface_point_radius

    return nearest_point_idx, nearest_point_idx_mask


def antipodal_xyz_coords(coords_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return -coords_xyz


def antipodal_sphere_coords(theta: torch.Tensor, phi: torch.Tensor) -> _ThetaPhiResult:
    return _ThetaPhiResult(
        theta=torch.pi - theta, phi=(phi + torch.pi) % (2 * torch.pi)
    )


def antipodal_unit_sphere_arc_len(
    theta_1: torch.Tensor,
    phi_1: torch.Tensor,
    theta_2: torch.Tensor,
    phi_2: torch.Tensor,
) -> torch.Tensor:
    theta_1p, phi_1p = antipodal_sphere_coords(theta_1, phi_1)
    arc_len_1_2 = unit_sphere_arc_len(
        theta_1=theta_1, phi_1=phi_1, theta_2=theta_2, phi_2=phi_2
    )
    arc_len_1p_2 = unit_sphere_arc_len(
        theta_1=theta_1p, phi_1=phi_1p, theta_2=theta_2, phi_2=phi_2
    )

    return torch.minimum(arc_len_1_2, arc_len_1p_2)


def _antipodal_sym_arc_len(
    cart_dir_x: torch.Tensor, cart_dir_y: torch.Tensor
) -> torch.Tensor:
    x = cart_dir_x
    y = cart_dir_y
    x = einops.rearrange(x, "... coord -> (...) coord")
    y = einops.rearrange(y, "... coord -> (...) coord")
    x_norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    x_norm = torch.where(
        torch.isclose(x_norm, x_norm.new_zeros(1), atol=1e-4),
        torch.ones_like(x_norm),
        x_norm,
    )
    x = x / x_norm
    y_norm = torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)
    y_norm = torch.where(
        torch.isclose(y_norm, y_norm.new_zeros(1), atol=1e-4),
        torch.ones_like(y_norm),
        y_norm,
    )
    y = y / y_norm

    # Project vectors into the top hemisphere, as we have antipodal symmetry in dwi.
    y = torch.where(y[:, -1, None] < 0, -y, y)
    # Find arc length between x and y when x is in both the northern and southern
    # hemspheres.
    # Orig x angles
    x_dot_y = (x.to(torch.float64) * y.to(torch.float64)).sum(-1, keepdims=True)
    x_dot_y.clamp_(min=MIN_COS_SIM, max=MAX_COS_SIM)
    arc_l_orig = torch.arccos(x_dot_y).to(y)
    # x vectors flipped across all axes
    x_dot_y = ((-x).to(torch.float64) * y.to(torch.float64)).sum(-1, keepdims=True)
    x_dot_y.clamp_(min=MIN_COS_SIM, max=MAX_COS_SIM)
    arc_l_flip = torch.arccos(x_dot_y).to(y)
    # Take the nearest arc length between the original and the flipped vectors.
    arc_len = torch.minimum(arc_l_orig, arc_l_flip)

    arc_len = arc_len.reshape(cart_dir_x.shape[:-1])

    return arc_len


def _antipodal_sym_pairwise_arc_len(
    cart_dir_x: torch.Tensor, cart_dir_y: torch.Tensor
) -> torch.Tensor:
    x = cart_dir_x
    y = cart_dir_y
    x = einops.rearrange(x, "... coord -> (...) coord")
    y = einops.rearrange(y, "... coord -> (...) coord")
    x_norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    x_norm = torch.where(
        torch.isclose(x_norm, x_norm.new_zeros(1), atol=1e-4),
        torch.ones_like(x_norm),
        x_norm,
    )
    x = x / x_norm
    y_norm = torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)
    y_norm = torch.where(
        torch.isclose(y_norm, y_norm.new_zeros(1), atol=1e-4),
        torch.ones_like(y_norm),
        y_norm,
    )
    y = y / y_norm

    # Project vectors into the top hemisphere, as we have antipodal symmetry in dwi.
    y = torch.where(y[:, -1, None] < 0, -y, y)
    # Find arc length between x and y when x is in both the northern and southern
    # hemspheres.
    # Orig x angles
    x_dot_y = einops.einsum(
        x.to(torch.float64), y.to(torch.float64), "b1 d, b2 d -> b1 b2"
    )
    x_dot_y.clamp_(min=MIN_COS_SIM, max=MAX_COS_SIM)
    arc_l_orig = torch.arccos(x_dot_y).to(y)
    # x vectors flipped across all axes
    x_dot_y = einops.einsum(
        (-x).to(torch.float64), y.to(torch.float64), "b1 d, b2 d -> b1 b2"
    )
    x_dot_y.clamp_(min=MIN_COS_SIM, max=MAX_COS_SIM)
    arc_l_flip = torch.arccos(x_dot_y).to(y)
    # Take the nearest arc length between the original and the flipped vectors.
    arc_len = torch.minimum(arc_l_orig, arc_l_flip)

    return arc_len
