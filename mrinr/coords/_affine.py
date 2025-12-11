# -*- coding: utf-8 -*-
# Functions and objects that handle coordinates in affine spaces.

import functools
from typing import NamedTuple, Optional, Tuple

import einops
import matplotlib.pyplot as plt
import monai
import monai.data.meta_tensor
import monai.data.utils
import monai.transforms.spatial.functional
import monai.transforms.utils
import nibabel as nib
import numpy as np
import torch

import mrinr

__all__ = [
    "inv_affine",
    "spacing",
    "transform_coords",
    "_canonicalize_coords_3d_affine",
    "_canonicalize_coords_2d_affine",
    "el_coord_grid",
    "affine_coord_grid",
    "affine_el2normalized_grid",
    "combine_affines",
    "enumerate_bb_to_corners",
    "exscribe_corners_to_bb",
    "bb_to_affine_grid",
    "el_bb_from_shape",
    "pad_affine",
    "center_scale_affine",
    "resize_affine",
    "get_neuro_affine_orientation_code",
    "zoom",
    "scale_coord_grid_by_bb",
    #!To be deprecated or updated:
    # "enumerate_bb_to_corners_3d",
    # "exscribe_corners_to_bb_3d",
    # "bb_to_affine_grid_3d",
    # "affine_vox2normalized_grid",
    # "affine_coordinate_grid",
    "_BatchVoxRealAffineSpace",
    # "_fov_coord_grid",
    # "_reorient_affine_space",
    # "_transform_affine_space",
    "_VoxRealAffineSpace",
    "RealAffineSpaceVol",
    # "fov_bb_coords_from_vox_shape",
    # "scale_fov_spacing",
    "vox_shape_from_fov",
]


#!DEPRECATED
class RealAffineSpaceVol(NamedTuple):
    vol: mrinr.typing.SingleVolume
    affine_vox2real: mrinr.typing.SingleHomogeneousAffine3D
    fov_bb_real: mrinr.typing.SingleFOV3D


#!DEPRECATED
class _VoxRealAffineSpace(NamedTuple):
    vox_vol: torch.Tensor
    affine_vox2real: torch.Tensor
    fov_bb_real: torch.Tensor


#!DEPRECATED
class _BatchVoxRealAffineSpace(NamedTuple):
    vox_vols: Tuple[torch.Tensor, ...]
    affine_vox2real: torch.Tensor
    fov_bb_real: torch.Tensor


def inv_affine(
    homog_aff: mrinr.typing.AnyHomogeneousAffineSD,
    rounding_decimals: Optional[int] = 8,
) -> mrinr.typing.AnyHomogeneousAffineSD:
    a = homog_aff
    a_inv: torch.Tensor = torch.linalg.inv(a)
    # Clean up any numerical instability artifacts.
    if rounding_decimals is not None:
        a_inv.round_(decimals=rounding_decimals)
    a_inv[..., -1, -1] = 1.0
    a_inv[..., -1, :-1] = 0.0

    return a_inv


def combine_affines(
    *affines: mrinr.typing.AnyHomogeneousAffineSD,
    transform_order_left_to_right: bool = True,
) -> mrinr.typing.AnyHomogeneousAffineSD:
    """Combine two affine homogeneous matrices in the specified order.

    To combine the affines in the order given in the function arguments (left to right),
    the order of matrix multiplication must be flipped. For example, if transformation A
    should be applied before transformation B, then the matrix multiplication would be
    `B @ A`.

    Parameters
    ----------
    affines : mrinr.typing.AnyHomogeneousAffineSD
        Affine matrices to combine.
    transform_order_left_to_right : bool, optional
        Indicate the desired order of transformations, by default True.

        If True (default), then the transforms will be applied in the order given in
        the function arguments, e.g. left to right, resulting in a matrix
        multiplication of `affine_2 @ affine_1`. If False, then the transforms will
        be applied from right-to-left, e.g. `affine_1 @ affine_2`. When False, the
        transform combination is the same as nibabel's `dot_reduce` function, See
        <https://nipy.org/nibabel/reference/nibabel.affines.html#dot-reduce>.

    Returns
    -------
    AnyHomogeneousAffineSD
        Merged affine matrix.
    """
    # Apply transforms in the order given in the function arguments.
    if transform_order_left_to_right:
        affs = reversed(list(affines))
    # Apply transforms in the order that would be given to a matrix multiplication.
    else:
        affs = list(affines)

    combined_affine = functools.reduce(
        lambda x, y: einops.einsum(x, y, "... i j, ... j k -> ... i k"), affs
    )

    return combined_affine


def spacing(
    affine: mrinr.typing.AnyHomogeneousAffineSD,
) -> torch.Tensor:
    # See <https://nipy.org/nibabel/reference/nibabel.affines.html#voxel-sizes>
    # and
    # <https://github.com/nipy/nibabel/blob/33c6721e564a999b510d83d2f7a3fbb2f4deb88b/nibabel/affines.py#L272>
    spacing = torch.linalg.norm(affine[..., :-1, :-1], ord=2, dim=-2)

    return spacing


def _canonicalize_coords_3d_affine(
    coords_3d: torch.Tensor,
    affine: torch.Tensor,
    broadcast_batch: bool = True,
) -> Tuple[mrinr.typing.CoordGrid3D, mrinr.typing.HomogeneousAffine3D]:
    """Reshape and/or repeat tensor elements to a compatible shape.

    coords_3d : torch.Tensor
        Coordinates in a coordinate-last format with shape `[B] x [s_1, s2, s3] x 3`.

        The last dimension must always be size 3.
        If coords_3d is 1D, then it will be expanded according to the common batch
            size.
        If coords_3d is 2D, then the first dimension is assumed to be a batch size.
        If coords_3d is 3D, then the first dimension is assumed to be the batch size,
            and the second is assumed to be a `within-batch` sample dimension.
        If coords_3d is 4D, then the first 3 dimensions are assumed to be spatial
            dimensions, and the batch size is assumed to be 1.
        If coords_3D is 5D, then the first dimension is assumed to be the batch size,
            and the next 3 dimensions are assumed to be spatial dimensions that
            correspond to their respective batch indices.
        Otherwise, a RuntimeError is raised.

    affine : torch.Tensor
        Affine matrix (or matrices) of shape `[B] x 4 x 4`.
    broadcast_batch: bool, optional
        Indicate whether the batches sizes will be broadcast to match, by default True

        Disabling this can be useful when handling the ambiguous case of a 4-dimensional
        coordinate tensor which is assumed to have a batch size of 1. If this is
        disabled, then this function will raise a RuntimeError when the affine's
        batch size is not 1, potentially catching false assumptions about the input
        data shape(s).

    Batch size will be determined as the broadcasted size of the batch sizes given
        in the shapes of `coords_3d` and `affine`. If the determined batch sizes are
        incompatible, a ValueError is raised. For example if coords_3d is `4 x 3`, and
        `affine` is `4 x 4`, then the batch size is set to `broadcast_shape(4, 1) = 4`.
        If `coords_3d` is `4 x 3` and `affine` is `2 x 4 x 4`, the batch sizes are
        incompatible, and an error is raised.
    """

    c = coords_3d
    if c.ndim < 1 or c.ndim > 5:
        raise RuntimeError(
            "ERROR: Expected coords_3d to have shape `[B] x [s_1, s2, s3] x 3`, but",
            f"got shape {tuple(c.shape)}.",
        )
    if c.shape[-1] != 3:
        raise RuntimeError(
            "ERROR: Expected coords_3d to have the last dimension be a coordinate "
            f"dimension of size 3, got '{c.shape[-1]}' "
            f"with full shape {tuple(c.shape)}."
        )

    # Add a batch dimension.
    if c.ndim == 1 or c.ndim == 4:
        c = torch.unsqueeze(c, 0)

    # Add or expand spatial dimensions, if necessary.
    if c.ndim == 2:
        c_spatial_shape = (1, 1, 1)
    elif c.ndim == 3:
        c_spatial_shape = (c.shape[1], 1, 1)
    else:
        c_spatial_shape = tuple(c.shape[1:-1])
    c = c.reshape(c.shape[0], *c_spatial_shape, c.shape[-1])
    batch_c = c.shape[0]

    a = affine
    if a.ndim < 2 or a.ndim > 3 or tuple(a.shape[-2:]) != (4, 4):
        raise RuntimeError(
            f"ERROR: Expected affine of shape `[B] x 4 x 4`, got {tuple(a.shape)}"
        )
    if a.ndim == 2:
        a = torch.unsqueeze(a, 0)

    batch_a = a.shape[0]

    if batch_c != 1 and batch_a != 1 and batch_c != batch_a:
        raise RuntimeError("ERROR: Batch sizes are not broadcastable")

    common_batch_size = max(batch_c, batch_a)
    if batch_c != batch_a:
        if not broadcast_batch:
            raise RuntimeError(
                "ERROR: Batch sizes broadcasting is not allowed, "
                + "but broadcasting is required for transforming "
                + f"coordinate shape '{coords_3d.shape}' -> '{c.shape} "
                + f"and affine shape {affine.shape} -> '{a.shape}'."
            )
        if batch_c == 1:
            c = einops.repeat(
                c, "1 s1 s2 s3 coord -> b s1 s2 s3 coord", b=common_batch_size
            )
        elif batch_a == 1:
            a = einops.repeat(
                a,
                "1 homog_aff_1 homog_aff_2 -> b homog_aff_1 homog_aff_2",
                b=common_batch_size,
            )

    c = c.to(torch.result_type(c, a))
    a = a.to(torch.result_type(c, a))

    return c, a


def _canonicalize_coords_2d_affine(
    coords_2d: torch.Tensor,
    affine: torch.Tensor,
    broadcast_batch: bool = True,
) -> tuple[mrinr.typing.CoordGrid2D, mrinr.typing.HomogeneousAffine2D]:
    """Reshape and/or repeat tensor elements to a compatible shape.

    coords_2d : torch.Tensor
        Coordinates in a coordinate-last format with shape `[B] x [s_1, s2] x 2`.

        The last dimension must always be size 2.
        If coords_2d is 1D, then it will be expanded according to the common batch
            size.
        If coords_2d is 2D, then the first dimension is assumed to be a batch size.
        If coords_2d is 3D, then the first 2 dimensions are assumed to be spatial
            dimensions, and the batch size is assumed to be 1.
        If coords_2d is 4D, then the first dimension is assumed to be the batch size,
            and the next 2 dimensions are assumed to be spatial dimensions that
            correspond to their respective batch indices.
        Otherwise, a RuntimeError is raised.

    affine : torch.Tensor
        Affine matrix (or matrices) of shape `[B] x 3 x 3`.
    broadcast_batch: bool, optional
        Indicate whether the batches sizes will be broadcast to match, by default True

        Disabling this can be useful when handling the ambiguous case of a 3-dimensional
        coordinate tensor which is assumed to have a batch size of 1. If this is
        disabled, then this function will raise a RuntimeError when the affine's
        batch size is not 1, potentially catching false assumptions about the input
        data shape(s).

    Batch size will be determined as the broadcasted size of the batch sizes given
        in the shapes of `coords_2d` and `affine`. If the determined batch sizes are
        incompatible, a ValueError is raised. For example if coords_2d is `4 x 2`, and
        `affine` is `3 x 3`, then the batch size is set to `broadcast_shape(4, 1) = 4`.
        If `coords_2d` is `4 x 2` and `affine` is `2 x 3 x 3`, the batch sizes are
        incompatible, and an error is raised.
    """

    c = coords_2d
    if c.ndim < 1 or c.ndim > 4:
        raise RuntimeError(
            "ERROR: Expected coords_2d to have shape `[B] x [s_1, s2] x 2`, but",
            f"got shape {tuple(c.shape)}.",
        )
    if c.shape[-1] != 2:
        raise RuntimeError(
            "ERROR: Expected coords_2d to have the last dimension be a coordinate "
            f"dimension of size 2, got '{c.shape[-1]}' "
            f"with full shape {tuple(c.shape)}."
        )

    # Add a batch dimension.
    if c.ndim == 1 or c.ndim == 3:
        c = torch.unsqueeze(c, 0)

    # Add or expand spatial dimensions, if necessary.
    if c.ndim == 2:
        c_spatial_shape = (1, 1)
    elif c.ndim == 3:
        c_spatial_shape = (c.shape[1], 1)
    else:
        c_spatial_shape = tuple(c.shape[1:-1])
    c = c.reshape(c.shape[0], *c_spatial_shape, c.shape[-1])
    batch_c = c.shape[0]

    a = affine
    if a.ndim < 2 or a.ndim > 3 or tuple(a.shape[-2:]) != (3, 3):
        raise RuntimeError(
            f"ERROR: Expected affine of shape `[B] x 3 x 3`, got {tuple(a.shape)}"
        )
    if a.ndim == 2:
        a = torch.unsqueeze(a, 0)

    batch_a = a.shape[0]

    if batch_c != 1 and batch_a != 1 and batch_c != batch_a:
        raise RuntimeError("ERROR: Batch sizes are not broadcastable")

    common_batch_size = max(batch_c, batch_a)
    if batch_c != batch_a:
        # Either the coord batch size or the affine batch size is 1, so expand that
        # singleton dimension.
        if not broadcast_batch:
            raise RuntimeError(
                "ERROR: Batch sizes broadcasting is not allowed, "
                + "but broadcasting is required for transforming "
                + f"coordinate shape '{coords_2d.shape}' -> '{c.shape} "
                + f"and affine shape {affine.shape} -> '{a.shape}'."
            )
        if batch_c == 1:
            c = einops.repeat(c, "1 s1 s2 coord -> b s1 s2 coord", b=common_batch_size)
        elif batch_a == 1:
            a = einops.repeat(
                a,
                "1 homog_aff_1 homog_aff_2 -> b homog_aff_1 homog_aff_2",
                b=common_batch_size,
            )

    c = c.to(torch.result_type(c, a))
    a = a.to(torch.result_type(c, a))

    return c, a


def transform_coords(
    coords: mrinr.typing.AnyCoordSD,
    affine_a2b: mrinr.typing.AnyHomogeneousAffineSD,
    broadcast_batch: bool = True,
) -> mrinr.typing.AnyCoordSD:
    # 2D
    if coords.shape[-1] == 2:
        c, a = _canonicalize_coords_2d_affine(
            coords, affine=affine_a2b, broadcast_batch=broadcast_batch
        )
        # Expand the affine matrices to be broadcastable over the coordinate spatial
        # dimensions.
        a = einops.rearrange(a, "b i j -> b 1 1 i j")
    # 3D
    elif coords.shape[-1] == 3:
        c, a = _canonicalize_coords_3d_affine(
            coords, affine=affine_a2b, broadcast_batch=broadcast_batch
        )
        # Expand the affine matrices to be broadcastable over the coordinate spatial
        # dimensions.
        a = einops.rearrange(a, "b i j -> b 1 1 1 i j")

    # Split the homogeneous affine matrix to avoid appending a 1 to the coordinates.
    # Rotation, scaling, and shearing.
    p = einops.einsum(a[..., :-1, :-1], c, "... i j, ... j -> ... i")
    # Translation.
    p += a[..., :-1, -1]

    # If the coordinates were repeated over a batch dimension, then reshape to include
    # that batch dimension.
    if p.numel() != coords.numel():
        b_out = p.shape[0]
        b_in = coords.shape[0] if coords.ndim == p.ndim else None
        # If input coordinates had no batch, then reshape p to be a batched version of
        # the input spatial size.
        if b_in is None:
            p = p.reshape(b_out, *tuple(coords.shape))
        # Otherwise, reshape p to override the batch size in the input shape, but keep
        # the same spatial size.
        else:
            p = p.reshape(b_out, *tuple(coords.shape[1:]))
        # p = p.reshape(-1, *tuple(coords.shape))
    # Otherwise, reshape p to match the input shape.
    else:
        p = p.reshape(coords.shape)
    if torch.is_floating_point(coords) and p.dtype != coords.dtype:
        p = p.to(dtype=coords.dtype)

    return p


def get_neuro_affine_orientation_code(
    affine: mrinr.typing.SingleHomogeneousAffine3D,
) -> str:
    a = affine.detach().cpu().numpy()
    ax_code = nib.orientations.aff2axcodes(a)
    return "".join([str(c).upper() for c in ax_code])


def el_coord_grid(
    spatial_shape: tuple[int, ...],
    to_torch_tensor: Optional[torch.Tensor] = None,
) -> mrinr.typing.SingleCoordGrid2D | mrinr.typing.SingleCoordGrid3D:
    if to_torch_tensor is not None:
        to_t = to_torch_tensor
    else:
        to_t = torch.ones(8, dtype=torch.float32)

    coord_grid = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(s, dtype=to_t.dtype, device=to_t.device)
                for s in spatial_shape
            ],
            indexing="ij",
        ),
        dim=-1,
    )
    return coord_grid


def affine_el2normalized_grid(
    spatial_fov: tuple, lower: float, upper: float, to_tensor=None
) -> torch.Tensor:
    """Construct an affine transformation that maps from element space to [lower, upper]."""
    diff = upper - lower

    if to_tensor is not None:
        aff_el2grid = torch.eye(
            len(spatial_fov) + 1, dtype=to_tensor.dtype, device=to_tensor.device
        )
    else:
        aff_el2grid = torch.eye(len(spatial_fov) + 1, dtype=torch.float)
    # Scale all coordinates to be in range [0, diff].
    aff_diag = diff / (aff_el2grid.new_tensor(spatial_fov) - 1)
    aff_diag = torch.cat([aff_diag, aff_diag.new_ones(1)], 0)
    aff_el2grid = aff_el2grid.diagonal_scatter(aff_diag)
    # Translate coords "back" by `lower` arbitrary unit(s) to make all coords within
    # range [lower, upper], rather than [0, diff].
    aff_el2grid[:-1, -1] = lower

    return aff_el2grid


def affine_coord_grid(
    affine_el2coord: mrinr.typing.AnyHomogeneousAffineSD,
    spatial_shape: tuple[int, ...],
) -> (
    mrinr.typing.SingleCoordGrid2D
    | mrinr.typing.SingleCoordGrid3D
    | mrinr.typing.CoordGrid2D
    | mrinr.typing.CoordGrid3D
):
    coord_grid = el_coord_grid(spatial_shape, to_torch_tensor=affine_el2coord)
    # If the affine matrix is batched, then the coordinate grid must also be batched.
    # Broadcasting rules should ensure that the batch size of 1 is expanded to match
    # the affine batch size.
    if affine_el2coord.ndim == 3:
        coord_grid = coord_grid.unsqueeze(0)

    return transform_coords(coord_grid, affine_a2b=affine_el2coord)


def bb_to_affine_grid(
    bb: mrinr.typing.SingleBoundingBox3D
    | mrinr.typing.BoundingBox3D
    | mrinr.typing.SingleBoundingBox2D
    | mrinr.typing.BoundingBox2D,
    spatial_shape: tuple[int, ...],
    affine: Optional[mrinr.typing.AnyHomogeneousAffineSD] = None,
):
    if bb.ndim == 2:
        b = bb.unsqueeze(0)
    else:
        b = bb
    # Check spatial dimension compatibility.
    if b.shape[-1] != len(spatial_shape):
        raise ValueError(
            "ERROR: Spatial dimension mismatch between bounding box and spatial shape. "
            + f"Spatial shape is '{spatial_shape}', "
            + f"while bounding box has '{b.shape[-1]}' spatial dimensions."
        )
    if affine is not None:
        if b.shape[-1] != affine.shape[-1] - 1:
            raise ValueError(
                "ERROR: Spatial dimension mismatch between bounding box and affine."
            )

    grids = list()
    # Meshgrid cannot be batched, so iterate over each bounding box.
    for b_i in b.unbind(0):
        sides = [
            torch.linspace(
                start=b_i_d_j[0],
                end=b_i_d_j[1],
                steps=d_j,
                dtype=b_i.dtype,
                device=b_i.device,
            )
            for d_j, b_i_d_j in zip(spatial_shape, b_i.unbind(-1), strict=True)
        ]
        grid = torch.stack(torch.meshgrid(*sides, indexing="ij"), dim=-1)
        grids.append(grid)
    grids = torch.stack(grids, dim=0)

    # If affine is None, assume an identity transformation.
    if affine is not None:
        if affine.ndim == 2:
            aff = affine.unsqueeze(0)
        else:
            aff = affine
        ret = transform_coords(grids, aff)
    else:
        ret = grids

    if bb.ndim == 2:
        ret = ret.squeeze(0)
    return ret


def enumerate_bb_to_corners(
    bb: mrinr.typing.SingleBoundingBox3D
    | mrinr.typing.BoundingBox3D
    | mrinr.typing.SingleBoundingBox2D
    | mrinr.typing.BoundingBox2D,
) -> (
    mrinr.typing.SingleCorners3D
    | mrinr.typing.Corners3D
    | mrinr.typing.SingleCorners2D
    | mrinr.typing.Corners2D
):
    if bb.ndim == 2:
        b = bb.unsqueeze(0)
    else:
        b = bb
    c = einops.rearrange(b, "batch lower_upper coord -> batch coord lower_upper")
    c = torch.stack(
        [torch.cartesian_prod(*(b_i.unbind(0))) for b_i in c.unbind(0)], dim=0
    )
    if bb.ndim == 2:
        c = c.squeeze(0)
    return c


def exscribe_corners_to_bb(
    corners: mrinr.typing.SingleCorners3D
    | mrinr.typing.Corners3D
    | mrinr.typing.SingleCorners2D
    | mrinr.typing.Corners2D,
    round_bb: bool = False,
    expansion_buffer: float = 0.0,
) -> (
    mrinr.typing.SingleBoundingBox3D
    | mrinr.typing.BoundingBox3D
    | mrinr.typing.SingleBoundingBox2D
    | mrinr.typing.BoundingBox2D
):
    if corners.ndim == 2:
        c = corners.unsqueeze(0)
    else:
        c = corners
    bb = torch.cat(
        [
            torch.amin(c, dim=1, keepdim=True) - expansion_buffer,
            torch.amax(c, dim=1, keepdim=True) + expansion_buffer,
        ],
        1,
    )
    if round_bb:
        # Snap bounding box to integer values (usually voxel indices).
        # Translate lower bound "down"
        bb[:, 0] = bb[:, 0].floor()
        # Translate upper bound "up"
        bb[:, 1] = bb[:, 1].ceil()

    if corners.ndim == 2:
        bb = bb.squeeze(0)
    return bb


def el_bb_from_shape(
    affine: torch.Tensor,
    spatial_shape: tuple[int, ...],
) -> mrinr.typing.SingleBoundingBox2D | mrinr.typing.SingleBoundingBox3D:
    spatial_dims = affine.shape[-1] - 1
    fov_shape = tuple(spatial_shape[-spatial_dims:])

    # Convert to 0-based pixel/voxel indexing.
    fov_end_points = torch.Tensor(fov_shape).to(affine) - 1
    edge_points = torch.stack([torch.zeros_like(fov_end_points), fov_end_points], dim=0)

    bb = transform_coords(edge_points, affine)

    return bb


def _coord_scale_factors_by_bb(
    bb: mrinr.typing.SingleBoundingBox2D
    | mrinr.typing.SingleBoundingBox3D
    | mrinr.typing.BoundingBox2D
    | mrinr.typing.BoundingBox3D,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_dims = bb.shape[-1]
    space_size = torch.abs(torch.diff(bb, dim=-2).float().flatten())
    if spatial_dims == 2:
        space_size = einops.rearrange(space_size, "... ndim -> ... 1 1 1 ndim")
        min_space_coord = einops.rearrange(
            bb[..., 0].to(space_size), "... ndim -> ... 1 1 1 ndim"
        )
    elif spatial_dims == 3:
        space_size = einops.rearrange(space_size, "... ndim -> ... 1 1 1 1 ndim")
        min_space_coord = einops.rearrange(
            bb[..., 0].to(space_size), "... ndim -> ... 1 1 1 1 ndim"
        )
    else:
        raise ValueError(
            "ERROR: Expected spatial dimensions to be 2 or 3, got "
            f"'{spatial_dims}' spatial dimensions."
        )

    return space_size, min_space_coord


def scale_coord_grid_by_bb(
    coords: mrinr.typing.SingleCoordGrid2D
    | mrinr.typing.SingleCoordGrid3D
    | mrinr.typing.CoordGrid2D
    | mrinr.typing.CoordGrid3D,
    bb: mrinr.typing.SingleBoundingBox2D
    | mrinr.typing.SingleBoundingBox3D
    | mrinr.typing.BoundingBox2D
    | mrinr.typing.BoundingBox3D,
) -> (
    mrinr.typing.SingleCoordGrid2D
    | mrinr.typing.SingleCoordGrid3D
    | mrinr.typing.CoordGrid2D
    | mrinr.typing.CoordGrid3D
):
    # Find space physical size and min coordinate for normalizing all template
    # coordinates to [0, 1].
    space_size, min_space_coord = _coord_scale_factors_by_bb(bb)
    return (coords - min_space_coord.to(coords)) / space_size.to(coords)


def pad_affine(
    affine: mrinr.typing.AnyHomogeneousAffineSD,
    pad_low: tuple[int, ...],
    return_transform_mat: bool = False,
) -> (
    mrinr.typing.AnyHomogeneousAffineSD
    | tuple[mrinr.typing.AnyHomogeneousAffineSD, mrinr.typing.AnyHomogeneousAffineSD]
):
    # Low pads require a translation of the affine matrix to maintain vox->real
    # mappings.
    batch_size = affine.shape[0] if affine.ndim == 3 else None
    if batch_size is None:
        affine = affine.unsqueeze(0)
    b = batch_size if batch_size is not None else 1
    d = affine.shape[-1] - 1
    pad_low_el_aff = (
        torch.eye(d + 1, dtype=affine.dtype, device=affine.device)
        .repeat(b, 1, 1)
        .contiguous()
    )

    # Translate the affine by the padding size in elements.
    pad_low_el_aff[..., :-1, -1] = -torch.Tensor(pad_low, device=affine.device).to(
        affine
    )
    # Transform the affine to be in the new padded space.
    padded_affine = einops.einsum(affine, pad_low_el_aff, "... i j, ... j k -> ... i k")

    if batch_size is None:
        pad_low_el_aff.squeeze_(0)
        padded_affine.squeeze_(0)
    if return_transform_mat:
        # Create the translation matrix that is equivalent to the translation
        # previously performed, but applied to the left side of the affine matrix.
        t = pad_low_el_aff.clone()
        t_diff = (padded_affine - affine)[..., :-1, -1]
        t[..., :-1, -1] = t_diff
        r = (padded_affine, t)
    else:
        r = padded_affine
    return r


def center_scale_affine(
    affine: mrinr.typing.AnyHomogeneousAffineSD,
    in_spatial_shape: tuple[int, ...],
    scale_factor: tuple[float, ...],
    out_spatial_shape: tuple[int, ...],
    return_transform_mat: bool = False,
) -> (
    mrinr.typing.AnyHomogeneousAffineSD
    | tuple[mrinr.typing.AnyHomogeneousAffineSD, mrinr.typing.AnyHomogeneousAffineSD]
):
    batch_size = affine.shape[0] if affine.ndim == 3 else None
    b = batch_size if batch_size is not None else 1
    d = affine.shape[-1] - 1
    if batch_size is None:
        affine = affine.unsqueeze(0)
    orig_a = mrinr.coords.transform_coords(
        ((torch.as_tensor(in_spatial_shape).repeat(b, 1)) - 1) / 2,
        affine,
        broadcast_batch=False,
    )
    Z = torch.as_tensor(
        monai.transforms.utils.create_scale(
            spatial_dims=d, scaling_factor=scale_factor, device=affine.device
        )
    ).to(affine)
    Z = einops.repeat(Z, "m n -> b m n", b=b)
    orig_b = mrinr.coords.transform_coords(
        ((torch.as_tensor(out_spatial_shape).repeat(b, 1)) - 1) / 2,
        mrinr.coords.combine_affines(Z, affine, transform_order_left_to_right=False),
        broadcast_batch=False,
    )
    t = orig_b - orig_a
    T = torch.eye(d + 1, dtype=affine.dtype, device=affine.device).repeat(b, 1, 1)
    T[..., :-1, -1] = -t
    M = mrinr.coords.combine_affines(T, Z, transform_order_left_to_right=False)
    new_affine = mrinr.coords.combine_affines(
        M, affine, transform_order_left_to_right=False
    )

    if batch_size is None:
        M.squeeze_(0)
        new_affine.squeeze_(0)
    if return_transform_mat:
        r = (new_affine, M)
    else:
        r = new_affine
    return r


def resize_affine(
    affine_x_el2coords: mrinr.typing.AnyHomogeneousAffineSD,
    in_spatial_shape: tuple[int, ...],
    target_spatial_shape: tuple[int, ...],
    centered: bool = True,
    return_coord_grid: bool = True,
    return_transform_mat: bool = False,
) -> (
    tuple[
        mrinr.typing.AnyHomogeneousAffineSD,
        mrinr.typing.SingleCoordGrid2D
        | mrinr.typing.SingleCoordGrid3D
        | mrinr.typing.CoordGrid2D
        | mrinr.typing.CoordGrid3D,
    ]
    | mrinr.typing.AnyHomogeneousAffineSD
):
    spatial_dims = affine_x_el2coords.shape[-1] - 1
    in_shape = tuple(in_spatial_shape)[-spatial_dims:]
    target_shape = tuple(target_spatial_shape)[-spatial_dims:]
    M = monai.transforms.utils.scale_affine(
        spatial_size=in_shape, new_spatial_size=target_shape, centered=centered
    )
    M = torch.as_tensor(
        M, dtype=affine_x_el2coords.dtype, device=affine_x_el2coords.device
    )
    # A_resized = A @ M
    resized_affine = combine_affines(
        affine_x_el2coords, M, transform_order_left_to_right=False
    )

    if return_coord_grid:
        target_coord_grid = affine_coord_grid(
            resized_affine, spatial_shape=target_shape
        )
        if return_transform_mat:
            r = resized_affine, target_coord_grid, M
        else:
            r = resized_affine, target_coord_grid
    else:
        if return_transform_mat:
            r = resized_affine, M
        else:
            r = resized_affine

    return r


@torch.no_grad()
def zoom(
    x: mrinr.typing.SingleImage | mrinr.typing.SingleVolume,
    affine: mrinr.typing.SingleHomogeneousAffine2D
    | mrinr.typing.SingleHomogeneousAffine3D,
    zoom_factors: float | tuple[float],
    mode: str,
    padding_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_orig_dtype = x.dtype
    if x.dtype in {torch.bool, torch.uint8}:
        x = x.to(torch.float32)

    x_meta = monai.data.meta_tensor.MetaTensor(x, affine=affine)
    zoomed_x: monai.data.meta_tensor.MetaTensor = (
        monai.transforms.spatial.functional.zoom(
            x_meta,
            scale_factor=zoom_factors,
            mode=mode,
            padding_mode=padding_mode,
            keep_size=False,
            align_corners=True
            if mode in {"linear", "bilinear", "bicubic", "trilinear"}
            else None,
            dtype=x.dtype,
            lazy=False,
            transform_info=dict(),
        )
    )

    y, y_affine = zoomed_x, zoomed_x.affine
    y_affine = y_affine.to(affine)
    y = y.as_tensor().to(x_orig_dtype)
    return y, y_affine


# #!DEPRECATED
# def affine_coordinate_grid(
#     affine_vox2b: torch.Tensor, vox_fov_shape: Union[torch.Tensor, int, Tuple[int]]
# ) -> torch.Tensor:
#     if torch.is_tensor(vox_fov_shape):
#         fov_shape = tuple(vox_fov_shape.shape[-3:])
#     elif isinstance(vox_fov_shape, int):
#         fov_shape = (vox_fov_shape,) * 3
#     else:
#         fov_shape = tuple(vox_fov_shape)

#     vox_coords = torch.stack(
#         torch.meshgrid(
#             *[
#                 torch.arange(d, dtype=affine_vox2b.dtype, device=affine_vox2b.device)
#                 for d in fov_shape
#             ],
#             indexing="ij",
#         ),
#         dim=-1,
#     )

#     return transform_coords(vox_coords, affine_vox2b)


# #!DEPRECATED
# def _fov_coord_grid(
#     fov_bb_coords: torch.Tensor,
#     affine_vox2real: torch.Tensor,
# ):
#     spacing = torch.tensor(
#         nib.affines.voxel_sizes(affine_vox2real.detach().cpu().numpy())
#     ).to(fov_bb_coords)
#     extent = fov_bb_coords[1] - fov_bb_coords[0]
#     # Go in the negative direction for flipped axes.
#     spacing = torch.where(extent < 0, -spacing, spacing)
#     coord_axes = [
#         torch.arange(
#             fov_bb_coords[0, i],
#             fov_bb_coords[1, i] + (spacing[i] / 100),  # include endpoint
#             step=spacing[i],
#             dtype=fov_bb_coords.dtype,
#             device=fov_bb_coords.device,
#         )
#         for i in range(spacing.shape[0])
#     ]
#     coord_grid = torch.stack(torch.meshgrid(coord_axes, indexing="ij"), dim=-1)

#     return coord_grid


# #!DEPRECATED
# def scale_fov_spacing(
#     fov_bb_coords: torch.Tensor,
#     affine_vox2real: torch.Tensor,
#     spacing_scale_factors: Tuple[float, ...],
#     set_affine_orig_to_fov_orig: bool,
#     new_fov_align_direction: Literal["interior", "exterior"] = "interior",
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     scale_transform = np.ones(affine_vox2real.shape[-1])
#     for i, scale in enumerate(spacing_scale_factors):
#         scale_transform[i] = scale
#     scale_transform = torch.diag(torch.Tensor(scale_transform)).to(affine_vox2real)
#     affine_unit2rescaled_space = einops.einsum(
#         affine_vox2real, scale_transform, "... i j, ... j k -> ... i k"
#     )
#     # Round for slightly better numerical stability with the affine transforms.
#     # affine_unit2rescaled_space = torch.round(affine_unit2rescaled_space, decimals=7)
#     # Treat the vox space as just a unit, positive directed space, for convenience.
#     unit_fov_bb = transform_coords(
#         fov_bb_coords, inv_affine(affine_unit2rescaled_space, rounding_decimals=8)
#     )
#     # Clamp almost-zero values to be 0, as they should almost certainly be 0.
#     unit_fov_bb[0] = torch.where(
#         torch.isclose(
#             unit_fov_bb[0], torch.zeros_like(unit_fov_bb[0]), atol=1e-4, rtol=1e-5
#         ),
#         torch.zeros_like(unit_fov_bb[0]),
#         unit_fov_bb[0],
#     )

#     unit_fov_extent = unit_fov_bb[1] - unit_fov_bb[0]
#     new_unit_fov_bb = unit_fov_bb.clone()
#     residual_fov = unit_fov_extent % 1
#     # Dims that fall on multiples of the new spacing do not need to be rounded, such
#     # as when a spacing is scaled by an integer factor.
#     translate_dim_indicator = ~torch.isclose(residual_fov, residual_fov.new_tensor([0]))

#     # Align the new fov such that the length of each side is evenly divisible by the
#     # new spacing. The alignment may be pushed "inside" the original fov, or "outside"
#     # of it.
#     fov_align = new_fov_align_direction.lower().strip()
#     if fov_align in {"in", "inside", "internal", "interior"}:
#         fov_align = "interior"
#     elif fov_align in {"out", "outside", "external", "exterior"}:
#         fov_align = "exterior"
#     else:
#         raise ValueError(
#             f"ERROR: Invalid new_fov_align_direction: {new_fov_align_direction}"
#         )

#     residual_side = residual_fov / 2
#     # Translate lower bound "up"
#     new_unit_fov_bb[0] = new_unit_fov_bb[0] + (translate_dim_indicator * residual_side)
#     # Translate upper bound "down"
#     new_unit_fov_bb[1] = new_unit_fov_bb[1] - (translate_dim_indicator * residual_side)
#     if fov_align == "exterior":
#         # Move each side by 0.5 units
#         # Translate lower bound "down"
#         new_unit_fov_bb[0] = new_unit_fov_bb[0] - (translate_dim_indicator * 0.5)
#         # Translate upper bound "up"
#         new_unit_fov_bb[1] = new_unit_fov_bb[1] + (translate_dim_indicator * 0.5)

#     # Bring new fov bb coordinates back into real space.
#     new_fov_bb = transform_coords(new_unit_fov_bb, affine_unit2rescaled_space)
#     if set_affine_orig_to_fov_orig:
#         affine_out = affine_unit2rescaled_space.clone()
#         affine_out[:-1, -1] = new_fov_bb[0]
#     else:
#         affine_out = affine_unit2rescaled_space

#     return new_fov_bb, affine_out


# #!DEPRECATED
# def affine_vox2normalized_grid(
#     spatial_fov_vox: tuple, lower: float, upper: float, to_tensor=None
# ) -> torch.Tensor:
#     """Construct an affine transformation that maps from voxel space to [lower, upper]."""
#     diff = upper - lower

#     if to_tensor is not None:
#         aff_vox2grid = torch.eye(4, dtype=to_tensor.dtype, device=to_tensor.device)
#     else:
#         aff_vox2grid = torch.eye(4, dtype=torch.float)
#     # Scale all coordinates to be in range [0, diff].
#     aff_diag = diff / (aff_vox2grid.new_tensor(spatial_fov_vox) - 1)
#     aff_diag = torch.cat([aff_diag, aff_diag.new_ones(1)], 0)
#     aff_vox2grid = aff_vox2grid.diagonal_scatter(aff_diag)
#     # Translate coords "back" by `lower` arbitrary unit(s) to make all coords within
#     # range [lower, upper], rather than [0, diff].
#     aff_vox2grid[:3, 3:4] = lower

#     return aff_vox2grid


# #!DEPRECATED
# def fov_bb_coords_from_vox_shape(
#     affine_homog: torch.Tensor,
#     vox_vol: Optional[torch.Tensor] = None,
#     shape: Optional[Tuple[int, ...]] = None,
# ) -> torch.Tensor:
#     fov_shape: Tuple[int, ...]
#     if shape is not None and vox_vol is None:
#         if len(shape) == 4:
#             shape = shape[1:]
#         fov_shape = tuple([int(s) for s in shape])
#     elif shape is None and vox_vol is not None:
#         if vox_vol.ndim == 4:
#             vox_shape = tuple(vox_vol.shape[1:])
#         else:
#             vox_shape = tuple(vox_vol.shape)
#         fov_shape = tuple([int(s) for s in vox_shape])
#     else:
#         raise RuntimeError()

#     if len(fov_shape) != (affine_homog.shape[-1] - 1):
#         raise RuntimeError()

#     # Convert to 0-based voxel indexing.
#     fov_end_points = torch.Tensor(fov_shape).to(affine_homog) - 1
#     edge_points = torch.stack([torch.zeros_like(fov_end_points), fov_end_points], dim=0)

#     bb_coords = transform_coords(edge_points, affine_homog)

#     return bb_coords


#!DEPRECATED
def vox_shape_from_fov(
    fov_real_bb_coords: torch.Tensor, affine_vox2real: torch.Tensor, tol=1e-4
) -> Tuple[int, ...]:
    fov_vox = transform_coords(
        fov_real_bb_coords, inv_affine(affine_vox2real, rounding_decimals=8)
    )

    # While edges of the fov may not lie directly on a voxel's center, the size of the
    # fov should be unit length.
    fov_vox_len = (fov_vox[1] - fov_vox[0]) + 1
    if torch.max(torch.abs(fov_vox_len - fov_vox_len.round())) > tol:
        raise RuntimeError("ERROR: Vox fov rounding not within tol")
    fov_vox_shape = tuple(fov_vox_len.round().int().tolist())

    return fov_vox_shape


# #!DEPRECATED
# class _AffineSpace(NamedTuple):
#     affine: torch.Tensor
#     fov_bb_coords: torch.Tensor


# #!DEPRECATED
# def _transform_affine_space(
#     s: _AffineSpace,
#     affine_transform: torch.Tensor,
# ) -> _AffineSpace:
#     aff_p = einops.einsum(affine_transform, s.affine, "... i j, ... j k -> ... i k")
#     bb_p = transform_coords(s.fov_bb_coords, affine_transform)

#     return _AffineSpace(affine=aff_p, fov_bb_coords=bb_p)


# #!DEPRECATED
# def _reorientation_transform(
#     input_ornt_code: tuple[str, ...],
#     output_ornt_code: tuple[str, ...],
#     input_vox_space_shape: tuple[int, ...],
# ):
#     a_code = input_ornt_code
#     b_code = output_ornt_code

#     a_ornt = nib.orientations.axcodes2ornt(a_code)
#     b_ornt = nib.orientations.axcodes2ornt(b_code)
#     a2b_ornt = nib.orientations.ornt_transform(a_ornt, b_ornt)

#     transform_aff_inv = nib.orientations.inv_ornt_aff(
#         a2b_ornt, shape=input_vox_space_shape
#     )
#     transform_aff = inv_affine(transform_aff_inv)

#     return transform_aff


# #!DEPRECATED
# def _reorient_affine_space(
#     s: _AffineSpace, target_orientation_code: str
# ) -> _AffineSpace:
#     """Reorient the axes of a 3D affine space according to standard ornt codes.

#     Parameters
#     ----------
#     s : AffineSpace
#         Source affine space matrix and fov coordinates.
#     target_orientation_code : str
#         Target axis orientation. This must go by the MRI convention of R/L, A/P, and
#         I/S, in any order, as one single upper-cased string. Ex. 'LAS'.

#     Returns
#     -------
#     AffineSpace
#         Reoriented affine matrix and fov boundary coordinates.
#     """
#     target_ornt_code = tuple(target_orientation_code.upper().strip())
#     current_ornt_code = nib.orientations.aff2axcodes(s.affine.detach().cpu().numpy())

#     # Get shape for output space when discretized.
#     unit_sizes = nib.affines.voxel_sizes(s.affine.detach().cpu().numpy())
#     fov_sizes = torch.abs(s.fov_bb_coords[1] - s.fov_bb_coords[0]).cpu().numpy()
#     # Add 1 to get the actual shape, instead of the "length" of each "side."
#     input_shape = (
#         np.round(fov_sizes / unit_sizes, decimals=0).astype(int) + 1
#     ).tolist()

#     transform_aff = _reorientation_transform(
#         current_ornt_code, target_ornt_code, input_vox_space_shape=input_shape
#     )

#     transform_aff = torch.from_numpy(transform_aff).to(s.affine)

#     return _transform_affine_space(s, transform_aff)


def _plot_ras_bb_planes(*corner_coords: torch.Tensor):
    fig, axs = plt.subplots(
        nrows=1, ncols=3, sharex=False, sharey=False, figsize=(7.5, 3)
    )
    colors = plt.colormaps["tab10"].colors

    for i, c in enumerate(corner_coords):
        c = c.detach().cpu().numpy()
        color = colors[i]

        # XY
        # sort by z, take the first 4 points
        plane_corners = c[np.argsort(c[:, 2])][:4, :2]
        plotting_corners = plane_corners[
            np.lexsort((plane_corners[:, 0], plane_corners[:, 1]))
        ]
        plotting_corners = np.concatenate(
            [
                plotting_corners[:2],
                plotting_corners[None, 3],
                plotting_corners[None, 2],
                plotting_corners[4:],
                plotting_corners[None, 0],
            ],
            0,
        )
        ax = axs[0]
        ax.plot(plotting_corners[:, 0], plotting_corners[:, 1], marker=".", color=color)

        # YZ
        # sort by x, take the first 4 points
        plane_corners = c[np.argsort(c[:, 0])][:4, 1:]
        # Traverse the square by multi-value sorting.
        plotting_corners = plane_corners[
            np.lexsort((plane_corners[:, 0], plane_corners[:, 1]))
        ]
        # Rearrange the corner points for plotting.
        plotting_corners = np.concatenate(
            [
                plotting_corners[:2],
                plotting_corners[None, 3],
                plotting_corners[None, 2],
                plotting_corners[4:],
                plotting_corners[None, 0],
            ],
            0,
        )
        ax = axs[1]
        ax.plot(plotting_corners[:, 0], plotting_corners[:, 1], marker=".", color=color)

        # XZ
        # sort by y, take the first 4 points
        plane_corners = c[np.argsort(c[:, 1])][:4, (0, 2)]
        # Traverse the square by multi-value sorting.
        plotting_corners = plane_corners[
            np.lexsort((plane_corners[:, 0], plane_corners[:, 1]))
        ]
        # Rearrange the corner points for plotting.
        plotting_corners = np.concatenate(
            [
                plotting_corners[:2],
                plotting_corners[None, 3],
                plotting_corners[None, 2],
                plotting_corners[4:],
                plotting_corners[None, 0],
            ],
            0,
        )
        ax = axs[2]
        ax.plot(plotting_corners[:, 0], plotting_corners[:, 1], marker=".", color=color)

    axs[0].set_xlabel("X: Left <-> Right")
    axs[0].set_ylabel("Y: Posterior <-> Anterior")
    axs[0].set_title("Axial")
    axs[0].grid()

    axs[1].set_xlabel("Y: Posterior <-> Anterior")
    axs[1].set_ylabel("Z: Inferior <-> Superior")
    axs[1].set_title("Saggital")
    axs[1].grid()

    axs[2].set_xlabel("X: Left <-> Right")
    axs[2].set_ylabel("Z: Inferior <-> Superior")
    axs[2].set_title("Coronal")
    axs[2].grid()

    return fig
