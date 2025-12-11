# -*- coding: utf-8 -*-
import collections
import functools
import itertools
from typing import Iterator, Literal, Optional

import einops
import numpy as np
import pytest

# import pytransform3d
# import pytransform3d.rotations as rots
# import pytransform3d.transformations as tfs
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import monai
import monai.transforms.utils
import nibabel as nib
import nibabel.affines
import scipy
import scipy.ndimage
import skimage
import torch

import mrinr

# High quality entropy created with: f"0x{secrets.randbits(128):x}"
NP_RNG_SEED = 0x6B22E0406E0F6738F3F209D06465CAC6


SpaceToResample = collections.namedtuple(
    "SpaceToResample",
    ["x", "affine_x_el2coords", "x_coords", "sample_coords", "sample_el_coords"],
)


@functools.lru_cache
def el_coord_grid(spatial_shape: tuple):
    coord_grid = np.stack(
        np.stack(np.meshgrid(*[np.arange(s) for s in spatial_shape], indexing="ij"), -1)
    )
    return torch.from_numpy(coord_grid).to(torch.float32)


@functools.lru_cache
def get_blob_sample(
    length: int,
    spatial_dims: Literal[2, 3],
    blob_size_fraction: float = 0.1,
    volume_fraction: float = 0.5,
    channels: Optional[int] = None,
    batch_size: Optional[int] = None,
    blur_sigma: float = 0.0,
) -> SpaceToResample:
    np_rng = np.random.default_rng(NP_RNG_SEED)

    c = 1 if channels is None else channels
    b = 1 if batch_size is None else batch_size

    blobs = list()
    for i in range(b):
        for j in range(c):
            binary_blob = skimage.data.binary_blobs(
                length=length,
                blob_size_fraction=blob_size_fraction,
                volume_fraction=volume_fraction,
                n_dim=spatial_dims,
                rng=np_rng,
            )
            blob_dt = scipy.ndimage.distance_transform_edt(binary_blob).astype(
                np.float32
            )
            if blur_sigma > 0.0:
                blob_dt = scipy.ndimage.gaussian_filter(
                    blob_dt, sigma=blur_sigma, truncate=2.0
                )
            blob_dt = blob_dt / blob_dt.max()
            blobs.append(torch.from_numpy(blob_dt))
    if spatial_dims == 2:
        affine = np.eye(3)
    elif spatial_dims == 3:
        affine = np.eye(4)
    # Center coordinates to the volume isocenter.
    affine[:-1, -1] = -(np.asarray(blobs[0].shape) - 1) / 2

    unit_coord_grid = el_coord_grid(blobs[0].shape).numpy()
    coord_grid = nib.affines.apply_affine(affine, unit_coord_grid)

    affines = np.stack([np.copy(affine) for _ in range(b)], 0)
    affines = torch.from_numpy(affines).to(torch.float32)
    coord_grids = np.stack([np.copy(coord_grid) for _ in range(b)], 0)
    coord_grids = torch.from_numpy(coord_grids).to(torch.float32)
    if spatial_dims == 2:
        blobs = einops.rearrange(blobs, "(b c) x y -> b c x y", b=b, c=c).to(
            torch.float32
        )
    elif spatial_dims == 3:
        blobs = einops.rearrange(blobs, "(b c) x y z -> b c x y z", b=b, c=c).to(
            torch.float32
        )

    if batch_size is None:
        blobs = blobs.squeeze(0)
        affines = affines.squeeze(0)
        coord_grids = coord_grids.squeeze(0)
        if channels is None:
            blobs = blobs.squeeze(0)
    # If batch_size is indicated, then channels will be returned as a singleton, and the
    # channels argument will be ignored.
    return SpaceToResample(
        x=blobs,
        affine_x_el2coords=affines,
        x_coords=coord_grids,
        sample_coords=None,
        sample_el_coords=None,
    )


def space_resample_generator(
    spatial_dims: int,
    batch_size: Optional[int],
    single_x: torch.Tensor,
    single_affine_x_el2coords: torch.Tensor,
    single_x_coords: torch.Tensor,
    scale_factors: tuple[float, ...],
    rotations: tuple[float, ...],
    translations: tuple[float, ...],
) -> Iterator[tuple[SpaceToResample, tuple[np.ndarray, ...]]]:
    if (spatial_dims == 2 and single_x.ndim == 2) or (
        spatial_dims == 3 and single_x.ndim == 3
    ):
        # add channel dim
        x = single_x.unsqueeze(0)
        channels = None
    else:
        x = single_x
        channels = x.shape[0]

    # Scaling
    scales = [list(scale_factors)] * spatial_dims
    all_scales = tuple(itertools.product(*scales))

    # Rotations
    if spatial_dims == 2:
        rotations = [list(rotations)]
        all_rotations = tuple(itertools.product(*rotations))
    elif spatial_dims == 3:
        rotations = [list(rotations)] * spatial_dims
        all_rotations = tuple(itertools.product(*rotations))

    # Translations
    translations = [list(translations)] * spatial_dims
    all_translations = tuple(itertools.product(*translations))

    single_x_coords_numpy = single_x_coords.numpy()
    single_affine_x_el2coords_numpy = single_affine_x_el2coords.numpy()
    # x_el_coords = el_coord_grid(x.shape[1:]).numpy()
    yield_every = 1 if batch_size is None else batch_size
    curr_batch = list()

    for s, r, t in itertools.product(all_scales, all_rotations, all_translations):
        m_s = monai.transforms.utils.create_rotate(spatial_dims, r)
        m_r = monai.transforms.utils.create_scale(spatial_dims, s)
        m_t = monai.transforms.utils.create_translate(spatial_dims, t)

        tf = nib.affines.dot_reduce(m_t, m_r, m_s)
        # Target coordinates in "coordinate space" (i.e., real-world space).
        sample_coords = nib.affines.apply_affine(tf, single_x_coords_numpy)
        # Target coordinates in x's "element space".
        # Transform the real-world target coordinates back to x element coordinates,
        # while maintaining the transformation.)
        sample_el_coords = nib.affines.apply_affine(
            np.linalg.inv(single_affine_x_el2coords_numpy), sample_coords
        )

        b_i = SpaceToResample(
            x=x,
            affine_x_el2coords=single_affine_x_el2coords,
            x_coords=single_x_coords,
            sample_coords=torch.from_numpy(sample_coords).to(torch.float32),
            sample_el_coords=torch.from_numpy(sample_el_coords).to(torch.float32),
        )
        curr_batch.append(b_i)

        if len(curr_batch) == yield_every:
            x_ = torch.stack([b.x for b in curr_batch], 0)
            affines_ = torch.stack([b.affine_x_el2coords for b in curr_batch], 0)
            x_coords_ = torch.stack([b.x_coords for b in curr_batch], 0)
            sample_coords_ = torch.stack([b.sample_coords for b in curr_batch], 0)
            sample_el_coords_ = torch.stack([b.sample_el_coords for b in curr_batch], 0)
            if batch_size is None:
                x_ = x_.squeeze(0)
                affines_ = affines_.squeeze(0)
                x_coords_ = x_coords_.squeeze(0)
                sample_coords_ = sample_coords_.squeeze(0)
                sample_el_coords_ = sample_el_coords_.squeeze(0)
                if channels is None:
                    x_ = x_.squeeze(0)
            yield (
                SpaceToResample(
                    x=x_,
                    affine_x_el2coords=affines_,
                    x_coords=x_coords_,
                    sample_coords=sample_coords_,
                    sample_el_coords=sample_el_coords_,
                ),
                (s, r, t),
            )
            curr_batch.clear()


BASIC_SCALE = (1.0,)
BASIC_ROTATION = (0.0,)
BASIC_TRANSLATION = (0.0,)

TEST_SCALES = (0.54, 1.03, 2.0)
TEST_ROTATIONS = (np.pi / 4, np.pi / 2.53, np.pi / 0.179)
TEST_TRANSLATIONS = (-1.36, 0.05, 10.0)


def test_basic_2d_grid_resample_torch_trilinear():
    s = get_blob_sample(
        64, channels=None, batch_size=None, spatial_dims=2, blur_sigma=0.5
    )
    for resample_space, tf_params in space_resample_generator(
        spatial_dims=2,
        batch_size=None,
        single_x=s.x,
        single_affine_x_el2coords=s.affine_x_el2coords,
        single_x_coords=s.x_coords,
        scale_factors=BASIC_SCALE,
        rotations=BASIC_ROTATION,
        translations=BASIC_TRANSLATION,
    ):
        scipy_resample_grid = resample_space.sample_el_coords.movedim(-1, 0).numpy()
        im_tf_scipy = scipy.ndimage.map_coordinates(
            input=resample_space.x.numpy(),
            coordinates=scipy_resample_grid,
            order=1,
            cval=0.0,
            mode="grid-constant",
            prefilter=False,
        )
        im_tf_scipy = torch.from_numpy(im_tf_scipy)

        # Our sampling function.
        im_tf_mrinr = mrinr.grid_resample(
            x=resample_space.x,
            affine_x_el2coords=resample_space.affine_x_el2coords,
            sample_coords=resample_space.sample_coords,
            interp_lib="torch",
            mode_or_interpolation="linear",
            padding_mode_or_bound="zeros",
        )
        assert im_tf_mrinr.ndim == 2
        assert tuple(im_tf_mrinr.shape) == tuple(
            resample_space.sample_coords.shape[:-1]
        )
        assert im_tf_mrinr.dtype == resample_space.x.dtype

        assert torch.isclose(
            im_tf_scipy,
            im_tf_mrinr,
            atol=1e-4,
        ).all()


def test_basic_3d_grid_resample_torch_trilinear():
    s = get_blob_sample(
        64, channels=None, batch_size=None, spatial_dims=3, blur_sigma=0.5
    )
    for resample_space, tf_params in space_resample_generator(
        spatial_dims=3,
        batch_size=None,
        single_x=s.x,
        single_affine_x_el2coords=s.affine_x_el2coords,
        single_x_coords=s.x_coords,
        scale_factors=BASIC_SCALE,
        rotations=BASIC_ROTATION,
        translations=BASIC_TRANSLATION,
    ):
        scipy_resample_grid = resample_space.sample_el_coords.movedim(-1, 0).numpy()
        vol_tf_scipy = scipy.ndimage.map_coordinates(
            input=resample_space.x.numpy(),
            coordinates=scipy_resample_grid,
            order=1,
            cval=0.0,
            mode="grid-constant",
            prefilter=False,
        )
        vol_tf_scipy = torch.from_numpy(vol_tf_scipy)

        # Our sampling function.
        vol_tf_mrinr = mrinr.grid_resample(
            x=resample_space.x,
            affine_x_el2coords=resample_space.affine_x_el2coords,
            sample_coords=resample_space.sample_coords,
            interp_lib="torch",
            mode_or_interpolation="linear",
            padding_mode_or_bound="zeros",
        )
        assert vol_tf_mrinr.ndim == 3
        assert tuple(vol_tf_mrinr.shape) == tuple(
            resample_space.sample_coords.shape[:-1]
        )
        assert vol_tf_mrinr.dtype == resample_space.x.dtype

        assert torch.isclose(
            vol_tf_scipy,
            vol_tf_mrinr,
            atol=1e-4,
        ).all()


@pytest.mark.parametrize(
    "scales,rotations,translations",
    [
        (TEST_SCALES, BASIC_ROTATION, BASIC_TRANSLATION),
        (BASIC_SCALE, TEST_ROTATIONS, BASIC_TRANSLATION),
        (BASIC_SCALE, BASIC_ROTATION, TEST_TRANSLATIONS),
        ((0.54, 2.12), (np.pi / 6.75,), (-0.42, 2.57)),  # combined transformations
    ],
)
@pytest.mark.parametrize(
    "interp_lib,mode_or_interpolation,padding_mode_or_bound,interp_kwargs,scipy_order",
    [
        ("torch", "linear", "zeros", dict(), 1),
        ("interpol", "linear", "zero", dict(prefilter=False), 1),
        ("interpol", "cubic", "zero", dict(prefilter=False), 3),
    ],
)
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_nontrivial_target_space_isolated_transform_grid_resample(
    interp_lib: str,
    mode_or_interpolation: str,
    padding_mode_or_bound: str,
    spatial_dims: int,
    scipy_order: int,
    scales: tuple[float, ...],
    rotations: tuple[float, ...],
    translations: tuple[float, ...],
    interp_kwargs: dict,
):
    s = get_blob_sample(
        49, channels=None, batch_size=None, spatial_dims=spatial_dims, blur_sigma=0.5
    )
    for resample_space, tf_params in space_resample_generator(
        spatial_dims=spatial_dims,
        batch_size=None,
        single_x=s.x,
        single_affine_x_el2coords=s.affine_x_el2coords,
        single_x_coords=s.x_coords,
        scale_factors=scales,
        rotations=rotations,
        translations=translations,
    ):
        # print(tf_params)
        scipy_resample_grid = resample_space.sample_el_coords.movedim(-1, 0).numpy()
        x_tf_scipy = scipy.ndimage.map_coordinates(
            input=resample_space.x.numpy(),
            coordinates=scipy_resample_grid,
            order=scipy_order,
            cval=0.0,
            mode="grid-constant",
            prefilter=False,
        )
        x_tf_scipy = torch.from_numpy(x_tf_scipy).to(torch.float32)

        # MRINR sampling function.
        x_tf_mrinr = mrinr.grid_resample(
            x=resample_space.x,
            affine_x_el2coords=resample_space.affine_x_el2coords,
            sample_coords=resample_space.sample_coords,
            interp_lib=interp_lib,
            mode_or_interpolation=mode_or_interpolation,
            padding_mode_or_bound=padding_mode_or_bound,
            clamp=False,
            **interp_kwargs,
        ).to(torch.float32)

        assert x_tf_mrinr.ndim == spatial_dims
        assert tuple(x_tf_mrinr.shape) == tuple(resample_space.sample_coords.shape[:-1])
        assert x_tf_mrinr.dtype == resample_space.x.dtype

        assert torch.isclose(
            x_tf_scipy,
            x_tf_mrinr,
            atol=1e-4,
        ).all()


@pytest.mark.parametrize(
    "batch_size,channels",
    [
        (None, 1),
        (None, 3),
        (1, 1),
        (1, 6),
        (7, 1),
        (4, 3),
    ],
)
@pytest.mark.parametrize(
    "interp_lib,mode_or_interpolation,padding_mode_or_bound,interp_kwargs",
    [
        ("torch", "linear", "zeros", dict()),
        ("interpol", "linear", "zero", dict(prefilter=False)),
        ("interpol", "cubic", "zero", dict(prefilter=False)),
    ],
)
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_channels_and_batches_nontrivial_target_space_grid_resample(
    interp_lib: str,
    mode_or_interpolation: str,
    padding_mode_or_bound: str,
    spatial_dims: int,
    channels: int,
    batch_size: Optional[int],
    interp_kwargs: dict,
):
    scales = (0.9,)
    rotations = (np.pi / 2.145, np.pi / 6.573)
    translations = (-0.1,)
    # Batches will be composed of different transformations of the same input.
    s = get_blob_sample(
        43,
        channels=channels,
        batch_size=None,
        spatial_dims=spatial_dims,
        blur_sigma=0.5,
    )

    for resample_space, tf_params in space_resample_generator(
        spatial_dims=spatial_dims,
        batch_size=batch_size,
        single_x=s.x,
        single_affine_x_el2coords=s.affine_x_el2coords,
        single_x_coords=s.x_coords,
        scale_factors=scales,
        rotations=rotations,
        translations=translations,
    ):
        # MRINR sampling function.
        vectorized_x_tf_mrinr = mrinr.grid_resample(
            x=resample_space.x,
            affine_x_el2coords=resample_space.affine_x_el2coords,
            sample_coords=resample_space.sample_coords,
            interp_lib=interp_lib,
            mode_or_interpolation=mode_or_interpolation,
            padding_mode_or_bound=padding_mode_or_bound,
            clamp=False,
            **interp_kwargs,
        ).to(torch.float32)

        target_shape = list()
        if batch_size is not None:
            target_shape.append(batch_size)
        target_shape.append(channels)
        target_shape.extend(
            resample_space.sample_coords.shape[-(spatial_dims + 1) : -1]
        )

        assert tuple(vectorized_x_tf_mrinr.shape) == tuple(target_shape)
        assert vectorized_x_tf_mrinr.dtype == resample_space.x.dtype

        if batch_size is None:
            batch_iter = (None,)
        else:
            batch_iter = range(batch_size)

        for i in batch_iter:
            for j in range(channels):
                if i is None:
                    idx = j
                else:
                    idx = (i, j)
                x_element_from_vectorized = vectorized_x_tf_mrinr[idx]
                x_ij = resample_space.x[i, j] if i is not None else resample_space.x[j]
                affine_x_el2coords_ij = (
                    resample_space.affine_x_el2coords[i]
                    if i is not None
                    else resample_space.affine_x_el2coords
                )
                sample_coords_ij = (
                    resample_space.sample_coords[i]
                    if i is not None
                    else resample_space.sample_coords
                )

                single_x_tf_mrinr = mrinr.grid_resample(
                    x=x_ij,
                    affine_x_el2coords=affine_x_el2coords_ij,
                    sample_coords=sample_coords_ij,
                    interp_lib=interp_lib,
                    mode_or_interpolation=mode_or_interpolation,
                    padding_mode_or_bound=padding_mode_or_bound,
                    clamp=False,
                    **interp_kwargs,
                ).to(torch.float32)

                assert torch.isclose(
                    x_element_from_vectorized,
                    single_x_tf_mrinr,
                    atol=1e-6,
                ).all()


# plt.figure(); plt.imshow(x_tf_scipy); plt.colorbar();plt.show(block=False);plt.figure();plt.imshow(x_tf_mrinr);plt.colorbar();plt.show(block=False);plt.figure();plt.imshow(x_tf_scipy-x_tf_mrinr);plt.colorbar(); plt.show(block=False)
