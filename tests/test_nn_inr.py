# -*- coding: utf-8 -*-
# Tests for INR-related functions.
import itertools

import einops
import monai
import monai.transforms
import nibabel as nib
import nibabel.affines
import pytest
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from test_grid_resample import el_coord_grid

import mrinr

###### 3D


@pytest.mark.parametrize(
    "grid_spatial_size",
    [(9, 9, 9), (10, 10, 10), (11, 11, 11), (57, 57, 57)],
)
def test_inr_spatial_ensemble_3d_coord_features_generator_simple_grids(
    grid_spatial_size: tuple[int, ...],
):
    vox_grid = torch.stack(
        torch.meshgrid([torch.arange(s) for s in grid_spatial_size], indexing="ij"), -1
    )[None]
    affine_id = torch.eye(4).unsqueeze(0)
    # Query exactly in the center of each voxel grid
    q = vox_grid[:, 1:, 1:, 1:] - 0.5

    for ensemble_feats in mrinr.nn.spatial_ensemble_3d_coord_features_generator(
        query_coords=q,
        affine_vox2coords=affine_id,
    ):
        ensemble_vox_idx = ensemble_feats.input_el_idx
        d_x_q = ensemble_feats.diff_ensemble_corner_to_ensemble_q
        w = ensemble_feats.spatial_weight
        local_corner_abc = ensemble_feats.ensemble_corner

        assert local_corner_abc.ndim == 1
        assert tuple(local_corner_abc.shape) == (3,)
        assert (local_corner_abc == local_corner_abc.int()).all()
        assert ((local_corner_abc == 0) | (local_corner_abc == 1)).all()

        assert (ensemble_vox_idx == ensemble_vox_idx.int()).all()

        # All coordinate diffs should be -0.5
        assert torch.isclose(torch.abs(d_x_q), torch.ones_like(d_x_q) * 0.5).all()
        assert torch.isclose(w, torch.ones_like(w) * (0.5**3)).all()
        # Test values in ensemble diff.
        for corner_dim in range(3):
            c = local_corner_abc[..., corner_dim].flatten()
            d = d_x_q[..., corner_dim].flatten()
            if c == 0:
                assert ((d <= 0) & (d >= -1.0)).all()
            elif c == 1:
                assert ((d >= 0) & (d <= 1.0)).all()


@pytest.mark.parametrize("grid_size", list(itertools.permutations((6, 13, 24), 3)))
@pytest.mark.parametrize(
    "q_vox_offset", list(itertools.permutations((0.0, 0.3141297, 0.9149), 3))
)
@pytest.mark.parametrize("q_vox_scale", [(0.8, 1.0, 1.24)])
@pytest.mark.parametrize(
    "affine_scale", list(itertools.permutations((0.9479, 1.217, 1.0), 3))
)
@pytest.mark.parametrize("affine_translate", [(-30.413, 119.7545, 87.4)])
@pytest.mark.parametrize(
    "affine_rot", [(torch.pi / 8.438, torch.pi / 11.493, torch.pi / 5.7)]
)
def test_inr_spatial_ensemble_3d_coord_features_generator_affine_grids(
    grid_size: tuple,
    q_vox_offset: tuple,
    q_vox_scale: tuple,
    affine_scale: tuple,
    affine_translate: tuple,
    affine_rot: tuple,
):
    batch_size = 2
    vox_grid = el_coord_grid(grid_size)

    # Set the query coordinates to be some arbitrary patch taken from the original grid,
    # offset by some amount.
    q_vox_coord = (
        vox_grid[:-1, :-3, 2:-2] + torch.tensor(q_vox_offset).reshape(1, 1, 1, 3)
    ) * torch.tensor(q_vox_scale).reshape(1, 1, 1, 3)

    affine_tf = monai.transforms.AffineGrid(
        rotate_params=affine_rot,
        translate_params=affine_translate,
        scale_params=affine_scale,
        align_corners=True,
    )
    _, affine = affine_tf(spatial_size=grid_size)

    q_coord = torch.from_numpy(
        nib.affines.apply_affine(affine.numpy(), q_vox_coord.numpy())
    )

    vox_grid = einops.repeat(vox_grid, "... coords -> b ... coords", b=batch_size)
    affine = einops.repeat(
        torch.as_tensor(affine), "row col -> b row col", b=batch_size
    )
    q_coord = einops.repeat(q_coord, "... coords -> b ... coords", b=batch_size)

    w_sum = None
    for ensemble_feats in mrinr.nn.spatial_ensemble_3d_coord_features_generator(
        query_coords=q_coord, affine_vox2coords=affine
    ):
        ensemble_vox_idx = ensemble_feats.input_el_idx
        d_x_q = ensemble_feats.diff_ensemble_corner_to_ensemble_q
        w = ensemble_feats.spatial_weight
        local_corner_abc = ensemble_feats.ensemble_corner

        assert local_corner_abc.ndim == 1
        assert tuple(local_corner_abc.shape) == (3,)
        assert (local_corner_abc == local_corner_abc.int()).all()
        assert ((local_corner_abc == 0) | (local_corner_abc == 1)).all()
        assert (ensemble_vox_idx == ensemble_vox_idx.int()).all()
        assert ((w >= 0) & (w <= 1.0)).all()
        assert torch.isclose(
            w,
            einops.reduce(
                1 - torch.abs(d_x_q),
                "b x y z coord -> b 1 x y z",
                reduction="prod",
            ),
        ).all()
        # Check that the sum of all weights = 1.0 at the end of the loop.
        if w_sum is None:
            w_sum = w
        else:
            w_sum += w

        # Check that diffs are in the correct range.
        for corner_dim in range(3):
            c = local_corner_abc[..., corner_dim].flatten().item()
            d = d_x_q[..., corner_dim].flatten()
            if c == 0:
                assert ((d <= 0) & (d >= -1.0)).all()
            elif c == 1:
                assert ((d >= 0) & (d <= 1.0)).all()

    assert torch.isclose(w_sum, w_sum.new_ones(1)).all()


####### 2D


@pytest.mark.parametrize(
    "grid_spatial_size",
    [(9, 9), (10, 10), (11, 11), (57, 57)],
)
def test_inr_spatial_ensemble_2d_coord_features_generator_simple_grids(
    grid_spatial_size: tuple[int, ...],
):
    vox_grid = torch.stack(
        torch.meshgrid([torch.arange(s) for s in grid_spatial_size], indexing="ij"), -1
    )[None]
    affine_id = torch.eye(3).unsqueeze(0)
    # Query exactly in the center of each pixel grid
    q = vox_grid[:, 1:, 1:] - 0.5

    for ensemble_feats in mrinr.nn.spatial_ensemble_2d_coord_features_generator(
        query_coords=q,
        affine_pixel2coords=affine_id,
    ):
        ensemble_idx = ensemble_feats.input_el_idx
        d_x_q = ensemble_feats.diff_ensemble_corner_to_ensemble_q
        w = ensemble_feats.spatial_weight
        local_corner_ab = ensemble_feats.ensemble_corner

        assert local_corner_ab.ndim == 1
        assert tuple(local_corner_ab.shape) == (2,)
        assert (local_corner_ab == local_corner_ab.int()).all()
        assert ((local_corner_ab == 0) | (local_corner_ab == 1)).all()

        assert (ensemble_idx == ensemble_idx.int()).all()

        # All coordinate diffs should be -0.5
        assert torch.isclose(torch.abs(d_x_q), torch.ones_like(d_x_q) * 0.5).all()
        assert torch.isclose(w, torch.ones_like(w) * (0.5**2)).all()
        # Test values in ensemble diff.
        for corner_dim in range(2):
            c = local_corner_ab[..., corner_dim].flatten()
            d = d_x_q[..., corner_dim].flatten()
            if c == 0:
                assert ((d <= 0) & (d >= -1.0)).all()
            elif c == 1:
                assert ((d >= 0) & (d <= 1.0)).all()


@pytest.mark.parametrize("grid_size", list(itertools.permutations((6, 13, 24), 2)))
@pytest.mark.parametrize(
    "q_pixel_offset", list(itertools.permutations((0.0, 0.3141297, 0.9149), 2))
)
@pytest.mark.parametrize("q_pixel_scale", [(0.8, 1.0)])
@pytest.mark.parametrize("affine_scale", list(itertools.permutations((0.9479, 1.0), 2)))
@pytest.mark.parametrize("affine_translate", [(-30.413, 87.4)])
@pytest.mark.parametrize("affine_rot", [(torch.pi / 8.438, torch.pi / 11.493)])
def test_inr_spatial_ensemble_2d_coord_features_generator_affine_grids(
    grid_size: tuple,
    q_pixel_offset: tuple,
    q_pixel_scale: tuple,
    affine_scale: tuple,
    affine_translate: tuple,
    affine_rot: tuple,
):
    batch_size = 2
    pix_grid = el_coord_grid(grid_size)

    # Set the query coordinates to be some arbitrary patch taken from the original grid,
    # offset by some amount.
    q_pix_coord = (
        pix_grid[:-3, 2:-2] + torch.tensor(q_pixel_offset).reshape(1, 1, 2)
    ) * torch.tensor(q_pixel_scale).reshape(1, 1, 2)
    affine_tf = monai.transforms.AffineGrid(
        rotate_params=affine_rot,
        translate_params=affine_translate,
        scale_params=affine_scale,
        align_corners=True,
    )
    _, affine = affine_tf(spatial_size=grid_size)

    q_coord = torch.from_numpy(
        nib.affines.apply_affine(affine.numpy(), q_pix_coord.numpy())
    )

    pix_grid = einops.repeat(pix_grid, "... coords -> b ... coords", b=batch_size)
    affine = einops.repeat(
        torch.as_tensor(affine), "row col -> b row col", b=batch_size
    )
    q_coord = einops.repeat(q_coord, "... coords -> b ... coords", b=batch_size)

    w_sum = None
    for ensemble_feats in mrinr.nn.spatial_ensemble_2d_coord_features_generator(
        query_coords=q_coord, affine_pixel2coords=affine
    ):
        ensemble_pix_idx = ensemble_feats.input_el_idx
        d_x_q = ensemble_feats.diff_ensemble_corner_to_ensemble_q
        w = ensemble_feats.spatial_weight
        local_corner_ab = ensemble_feats.ensemble_corner

        assert local_corner_ab.ndim == 1
        assert tuple(local_corner_ab.shape) == (2,)
        assert (local_corner_ab == local_corner_ab.int()).all()
        assert ((local_corner_ab == 0) | (local_corner_ab == 1)).all()
        assert (ensemble_pix_idx == ensemble_pix_idx.int()).all()
        assert ((w >= 0) & (w <= 1.0)).all()
        assert torch.isclose(
            w,
            einops.reduce(
                1 - torch.abs(d_x_q),
                "b x y coord -> b 1 x y",
                reduction="prod",
            ),
        ).all()
        # Check that the sum of all weights = 1.0 at the end of the loop.
        if w_sum is None:
            w_sum = w
        else:
            w_sum += w

        # Check that diffs are in the correct range.
        for corner_dim in range(2):
            c = local_corner_ab[..., corner_dim].flatten().item()
            d = d_x_q[..., corner_dim].flatten()
            if c == 0:
                assert ((d <= 0) & (d >= -1.0)).all()
            elif c == 1:
                assert ((d >= 0) & (d <= 1.0)).all()

    assert torch.isclose(w_sum, w_sum.new_ones(1)).all()


# ==============================================================================


# def test_inr_ensemble_coord_feature_generator_simple_vox_grid():
#     grid_spatial_size = (9, 9, 9)
#     vox_grid = torch.stack(
#         torch.meshgrid([torch.arange(s) for s in grid_spatial_size], indexing="ij"), -1
#     )
#     affine_id = torch.eye(4).unsqueeze(0)
#     # Query exactly in the center of each voxel grid
#     q = vox_grid[1:, 1:, 1:] - 0.5

#     vox_grid = vox_grid.reshape(1, -1, 3)
#     q = q.reshape(1, -1, 3)

#     for ensemble_feats in mrinr.nn.ensemble_3d_coord_features_generator(
#         query_coords=q,
#         input_spatial_size_vox=grid_spatial_size,
#         affine_vox2coords=affine_id,
#     ):
#         input_vox_idx = ensemble_feats.input_vox_idx_ijk
#         input_batch_idx = ensemble_feats.input_batch_idx
#         d_x_q = ensemble_feats.diff_abc_to_ensemble_q
#         w = ensemble_feats.trilin_vol_weight
#         local_corner = ensemble_feats.abc_ensemble_corner

#         assert input_batch_idx.ndim == 1
#         assert (input_batch_idx == 0).all()
#         assert (
#             (input_vox_idx >= 0)
#             & (input_vox_idx <= torch.as_tensor(grid_spatial_size).int() - 1)
#         ).all()
#         # All coordinate diffs should be -0.5
#         assert torch.isclose(torch.abs(d_x_q), torch.ones_like(d_x_q) * 0.5).all()
#         assert ((local_corner == 1) | (local_corner == 0)).all()
#         assert torch.isclose(w, torch.ones_like(w) * (0.5**3)).all()

#         for corner_dim in range(3):
#             c = local_corner[..., corner_dim].flatten()
#             d = d_x_q[..., corner_dim].flatten()
#             if c == 0:
#                 assert ((d <= 0) & (d >= -1.0)).all()
#             elif c == 1:
#                 assert ((d >= 0) & (d <= 1.0)).all()


# def test_inr_ensemble_coord_feature_generator_affine_grids():
#     grid_spatial_sizes = list(itertools.permutations((6, 11, 24), 3))
#     q_vox_offsets = list(itertools.permutations((0.0, 0.3141297, 0.5, 0.9149), 3))
#     affine_scales = list(itertools.permutations((0.9479, 1.217, 1.0), 3))
#     affine_translates = [(-30.413, 119.7545, 87.4)]
#     affine_rots = [(0, 0, 0), (torch.pi / 8.438, torch.pi / 11.493, torch.pi / 5.7)]

#     for grid_size in grid_spatial_sizes:
#         vox_grid = torch.stack(
#             torch.meshgrid([torch.arange(s) for s in grid_size], indexing="ij"), -1
#         )
#         vox_grid = einops.repeat(vox_grid, "... coords -> b ... coords", b=2)
#         for q_offset in q_vox_offsets:
#             q_vox_coord = vox_grid[:, :-1, :-3, 2:-2] + torch.tensor(q_offset).reshape(
#                 1, 1, 1, 1, 3
#             )
#             for s, t, r in itertools.product(
#                 affine_scales, affine_translates, affine_rots
#             ):
#                 affine_tf = monai.transforms.AffineGrid(
#                     rotate_params=r,
#                     translate_params=t,
#                     scale_params=s,
#                     align_corners=True,
#                 )
#                 _, affine = affine_tf(spatial_size=grid_size)
#                 affine = einops.repeat(
#                     torch.as_tensor(affine), "row col -> b row col", b=2
#                 )

#                 coord_grid = mrinr.coords.transform_coords(vox_grid, affine)
#                 coord_grid = einops.rearrange(
#                     coord_grid, "b x y z coord -> b (x y z) coord"
#                 )
#                 q_coord = mrinr.coords.transform_coords(q_vox_coord, affine)
#                 q_coord = einops.rearrange(q_coord, "b x y z coord -> b (x y z) coord")

#                 w_sum = None
#                 for ensemble_feats in mrinr.nn.ensemble_3d_coord_features_generator(
#                     query_coords=q_coord,
#                     input_spatial_size_vox=grid_size,
#                     affine_vox2coords=affine,
#                 ):
#                     input_vox_idx = ensemble_feats.input_vox_idx_ijk
#                     input_batch_idx = ensemble_feats.input_batch_idx
#                     d_x_q = ensemble_feats.diff_abc_to_ensemble_q
#                     w = ensemble_feats.trilin_vol_weight
#                     local_corner = ensemble_feats.abc_ensemble_corner

#                     # Batch index should be flat.
#                     assert input_batch_idx.ndim == 1
#                     # Voxel index should be between 0 and vox_size - 1.
#                     assert (
#                         (input_vox_idx >= 0)
#                         & (input_vox_idx <= torch.as_tensor(grid_size).int() - 1)
#                     ).all()
#                     # Corner values should either be 0 or 1
#                     assert ((local_corner == 1) | (local_corner == 0)).all()
#                     # Weights should be between 0 and 1, and should equal the compliment
#                     # of the volume of the diff.
#                     assert ((w >= 0) & (w <= 1.0)).all()
#                     assert torch.isclose(
#                         w,
#                         einops.reduce(
#                             1 - torch.abs(d_x_q),
#                             "b_n coord -> b_n 1",
#                             reduction="prod",
#                         ),
#                     ).all()
#                     # Check that the sum of all weights = 1.0 at the end of the loop.
#                     if w_sum is None:
#                         w_sum = w
#                     else:
#                         w_sum += w

#                     # Test that the input_vox_idx and input_batch_idx can be used to
#                     # index into the input space.
#                     vox_grid[
#                         input_batch_idx,
#                         input_vox_idx[..., 0],
#                         input_vox_idx[..., 1],
#                         input_vox_idx[..., 2],
#                         :,
#                     ]
#                     # Check that diffs are in the correct range.
#                     for corner_dim in range(3):
#                         c = local_corner[..., corner_dim].flatten().item()
#                         d = d_x_q[..., corner_dim].flatten()
#                         if c == 0:
#                             assert ((d <= 0) & (d >= -1.0)).all()
#                         elif c == 1:
#                             assert ((d >= 0) & (d <= 1.0)).all()

#                 assert torch.isclose(w_sum, w_sum.new_ones(1)).all()
