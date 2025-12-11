# -*- coding: utf-8 -*-
# Functions that can be applied to any type of volumetric data.
from typing import Literal, Optional, Tuple, Union

import einops
import numpy as np
import scipy
import sklearn
import sklearn.preprocessing
import torch
import transforms3d

import mrinr

__all__ = [
    "crop_pad",
    "crop_to_mask",
    "center_crop_pad_to_shape",
    "_crop_frs_inside_lr",
    "_crop_lr_inside_smallest_lr",
    "crop_vox",
    "crop_to_fov_bb",
    "distance_transform_mask",
    "pad_vox",
    "prefilter_gaussian_blur",
    "robust_scale_spatial_data",
]


def crop_pad(
    x: Union[
        mrinr.typing.SingleVolume | mrinr.typing.SingleImage,
        tuple[mrinr.typing.SingleVolume | mrinr.typing.SingleImage, ...],
    ],
    affine_el2coords: mrinr.typing.SingleHomogeneousAffine3D
    | mrinr.typing.SingleHomogeneousAffine2D,
    *spatial_pads_low_high: tuple[int, int],
    **torch_pad_kwargs,
) -> tuple[
    Union[
        mrinr.typing.SingleVolume | mrinr.typing.SingleImage,
        tuple[mrinr.typing.SingleVolume | mrinr.typing.SingleImage, ...],
    ],
    mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D,
]:
    # Outputs can be checked using a volume file and mrtrix & fsl commands:
    # Given a volume file `in_vol` in RAS orientation, and `spatial_pads_low_high`
    # equal to `((p_x_low, p_x_high), (p_y_low,...), ...)`:
    # ```bash
    # mrgrid -force "$in_vol" pad \
    #   -axis 0 p_x_low,p_x_high \
    #   -axis 1 p_y_low,p_y_high \
    #   -axis 2 p_z_low,p_z_high
    #   tmp_crop.nii.gz \
    #   && mrinfo -size tmp_crop.nii.gz \
    #   && mri_info --vox2ras tmp_crop.nii.gz
    # ```
    if affine_el2coords.shape[-1] == 3:
        spatial_dims = 2
    elif affine_el2coords.shape[-1] == 4:
        spatial_dims = 3

    if not isinstance(x, (list, tuple)):
        single_output = True
        x = (x,)
    else:
        single_output = False
    orig_dtypes = [x_i.dtype for x_i in x]

    if spatial_dims == 2:
        xs = tuple(mrinr.utils.ensure_image_channels(x_i, batch=True) for x_i in x)
    elif spatial_dims == 3:
        xs = tuple(mrinr.utils.ensure_vol_channels(x_i, batch=True) for x_i in x)
    # Convert any bool tensors for padding.
    xs = tuple(x_i.to(torch.uint8) if x_i.dtype == torch.bool else x_i for x_i in xs)
    p = spatial_pads_low_high
    # If pads/crops are excluded for later spatial dimensions, add 0-padding for those
    # dims.
    while len(p) < spatial_dims:
        p = p + ((0, 0),)

    expansions = np.asarray(p, dtype=int)
    # Torch expects a reversed, flattened order of padding amounts relative to
    # what numpy/humans would expect. So, flipping and flattening the padding
    # should give the pytorch expected order. At least pytorch padding allows negative
    # pad values as cropping...
    torch_expansions = tuple(np.flip(expansions, axis=0).flatten().tolist())
    ys = tuple(
        torch.nn.functional.pad(x_i, pad=torch_expansions, **torch_pad_kwargs)
        for x_i in xs
    )

    # Calculate the translations needed to maintain the element->coord mapping of
    # the affine matrix.
    # Only pads/crops at the "negative" end of each dimension are considered for the
    # affine translation, as they are the only translations that affect the origin.
    t = torch.eye(
        spatial_dims + 1, dtype=affine_el2coords.dtype, device=affine_el2coords.device
    )
    # Negate the low crop/pad values to compensate for the fov translation.
    spatial_crop_pad_low = -expansions[:, 0]
    t[..., :-1, -1] = t.new_tensor(spatial_crop_pad_low)
    # Apply translation in (pi|vo)xel coordinates, then combine with the original.
    new_affine = mrinr.coords.combine_affines(
        t, affine_el2coords, transform_order_left_to_right=True
    )
    ys = tuple(y_i.to(dtype) for y_i, dtype in zip(ys, orig_dtypes))

    # Undo shape changes for output tensors.
    if spatial_dims == 2:
        ys = tuple(mrinr.utils.undo_image_channels(y_i, x_i) for y_i, x_i in zip(ys, x))
    elif spatial_dims == 3:
        ys = tuple(mrinr.utils.undo_vol_channels(y_i, x_i) for y_i, x_i in zip(ys, x))

    if single_output:
        r = ys[0], new_affine
    else:
        r = ys, new_affine
    return r


def center_crop_pad_to_shape(
    x: Union[
        mrinr.typing.SingleVolume | mrinr.typing.SingleImage,
        tuple[mrinr.typing.SingleVolume | mrinr.typing.SingleImage, ...],
    ],
    affine_el2coords: mrinr.typing.SingleHomogeneousAffine3D
    | mrinr.typing.SingleHomogeneousAffine2D,
    target_shape: tuple[int, ...],
    return_spatial_pads: bool = False,
    **torch_pad_kwargs,
) -> tuple[
    Union[
        mrinr.typing.SingleVolume | mrinr.typing.SingleImage,
        tuple[mrinr.typing.SingleVolume | mrinr.typing.SingleImage, ...],
    ],
    mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D,
]:
    if not isinstance(x, (list, tuple)):
        single_output = True
        x = (x,)
    else:
        single_output = False

    if affine_el2coords.shape[-1] == 3:
        spatial_dims = 2
    elif affine_el2coords.shape[-1] == 4:
        spatial_dims = 3

    target_shape = np.asarray(target_shape).flatten().astype(int)[-spatial_dims:]
    input_shapes = [
        np.asarray(x_i.shape[-spatial_dims:]).flatten().astype(int) for x_i in x
    ]
    # All input shapes should be equal!
    assert all(
        [
            np.allclose(input_shapes[i], input_shapes[0])
            for i in range(len(input_shapes))
        ]
    ), f"Input shapes are not equal: {input_shapes}"

    to_crop_pad = target_shape - input_shapes[0]
    pad_low = np.ceil(to_crop_pad / 2).astype(int)
    pad_high = (to_crop_pad - pad_low).astype(int)
    pads_low_high = einops.rearrange(
        [pad_low, pad_high], "low_high coord -> coord low_high"
    )
    pads_low_high_tuple = tuple(
        [tuple(pads_low_high[i].tolist()) for i in range(pads_low_high.shape[0])]
    )
    if (pads_low_high != 0).any():
        y, y_affine = crop_pad(
            x, affine_el2coords, *pads_low_high_tuple, **torch_pad_kwargs
        )
    else:
        y, y_affine = x, affine_el2coords

    if return_spatial_pads:
        r = (y, y_affine, pads_low_high_tuple)
    else:
        r = (y, y_affine)
    if single_output:
        r = (r[0][0],) + r[1:]

    return r


def crop_to_mask(
    x: Union[
        mrinr.typing.SingleVolume | mrinr.typing.SingleImage,
        tuple[mrinr.typing.SingleVolume | mrinr.typing.SingleImage, ...],
    ],
    mask: mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage,
    affine_el2coords: mrinr.typing.SingleHomogeneousAffine3D
    | mrinr.typing.SingleHomogeneousAffine2D,
    crop_buffer: int,
    return_spatial_pads: bool = False,
    **torch_pad_kwargs,
):
    if affine_el2coords.shape[-1] == 3:
        spatial_dims = 2
    elif affine_el2coords.shape[-1] == 4:
        spatial_dims = 3

    # Determine the cropping bounds from the mask.
    el_coord_grid = mrinr.coords.el_coord_grid(mask.shape[1:]).to(affine_el2coords)

    el_coords_in_mask = el_coord_grid[mask.squeeze(0)].reshape(-1, spatial_dims)
    mask_el_lower_bound = torch.amin(el_coords_in_mask, dim=0).int()
    mask_el_upper_bound = torch.amax(el_coords_in_mask, dim=0).int()

    crop_low = mask_el_lower_bound - crop_buffer
    crop_high = (
        torch.as_tensor(mask.shape[1:]).int() - 1 - mask_el_upper_bound - crop_buffer
    )

    crops_low_high = einops.rearrange(
        [crop_low, crop_high], "low_high coord -> coord low_high"
    )
    # Negative padding is cropping.
    pads_low_high = -crops_low_high
    del crops_low_high, crop_low, crop_high
    pads_low_high = [
        tuple(pads_low_high[i].tolist()) for i in range(pads_low_high.shape[0])
    ]

    crop_x, crop_affine_el2coords = crop_pad(
        x, affine_el2coords, *pads_low_high, **torch_pad_kwargs
    )

    if return_spatial_pads:
        r = crop_x, crop_affine_el2coords, pads_low_high
    else:
        r = crop_x, crop_affine_el2coords
    return r


def crop_vox(
    vol_vox: Union[Tuple[mrinr.typing.SingleVolume], mrinr.typing.SingleVolume],
    affine_vox2real: mrinr.typing.SingleHomogeneousAffine3D,
    *crops_low_high: Tuple[int, int],
    **np_pad_kwargs,
) -> Tuple[
    Union[Tuple[mrinr.typing.SingleVolume], mrinr.typing.SingleVolume],
    mrinr.typing.SingleHomogeneousAffine3D,
]:
    if not torch.is_tensor(vol_vox):
        multi_vol = True
        v_inputs = list(vol_vox)
    else:
        multi_vol = False
        v_inputs = [vol_vox]
    v_chs = list()
    for v in v_inputs:
        v_chs.append(mrinr.utils.ensure_vol_channels(v))
    example_v_ch = v_chs[0]
    crop_low = [0] * (example_v_ch.ndim - 1)
    crop_high = [0] * (example_v_ch.ndim - 1)
    v_spatial_shape = tuple(example_v_ch.shape[-3:])
    # If negative crops are given, set that crop side to 0 and pad later.
    post_crop_pad = False
    pad_low = [0] * (example_v_ch.ndim - 1)
    pad_high = [0] * (example_v_ch.ndim - 1)
    for i, dim_i_crops in enumerate(crops_low_high):
        crop_low[i] = dim_i_crops[0]
        if crop_low[i] < 0:
            pad_low[i] = abs(crop_low[i])
            crop_low[i] = 0
            post_crop_pad = True
        crop_high[i] = dim_i_crops[1]
        if crop_high[i] < 0:
            pad_high[i] = abs(crop_high[i])
            crop_high[i] = 0
            post_crop_pad = True

    for i in range(len(v_chs)):
        # Subset the volume tensor(s).
        v_chs[i] = v_chs[i][
            ...,
            crop_low[0] : (v_spatial_shape[0] - crop_high[0]),
            crop_low[1] : (v_spatial_shape[1] - crop_high[1]),
            crop_low[2] : (v_spatial_shape[2] - crop_high[2]),
        ]

    # Low crops require a translation of the affine matrix to maintain vox->real
    # mappings.
    crop_low_vox_aff = torch.eye(affine_vox2real.shape[-1]).to(affine_vox2real)
    crop_low_vox_aff[:-1, -1] = torch.Tensor(crop_low).to(affine_vox2real)
    new_affine = einops.einsum(
        affine_vox2real, crop_low_vox_aff, "... i j, ... j k -> ... i k"
    )

    # Pad any negative crop values.
    if post_crop_pad:
        pads = [p for p in zip(pad_low, pad_high)]
        for i in range(len(v_chs)):
            new_v_ch, padded_new_affine = pad_vox(
                v_chs[i], new_affine, *pads, **np_pad_kwargs
            )
            v_chs[i] = new_v_ch
        new_affine = padded_new_affine
    # Reshape outputs back to their original channel status.
    out_v = list()
    for i in range(len(v_chs)):
        out_v.append(mrinr.utils.undo_vol_channels(v_chs[i], v_inputs[i]))
    # If the input was not a tuple of tensors, make sure the output is not, either.
    if not multi_vol:
        out_v = out_v[0]
    else:
        out_v = tuple(out_v)

    return (out_v, new_affine)


def pad_vox(
    vol_vox: torch.Tensor,
    affine_vox2real: torch.Tensor,
    *spatial_pads_low_high: Tuple[int, int],
    **np_pad_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_ch = mrinr.utils.ensure_vol_channels(vol_vox)
    pad_widths = np.zeros((v_ch.ndim, 2), dtype=int).tolist()
    # Add a (0, 0) padding at the start to not pad the channel dimension.
    if len(spatial_pads_low_high) != 4:
        spatial_pads_low_high = [(0, 0)] + list(spatial_pads_low_high)

    # If negative pads are given, set that crop side to 0 and pad later.
    post_pad_crop = False
    crop_low = [0] * (v_ch.ndim - 1)
    crop_high = [0] * (v_ch.ndim - 1)
    for i, dim_i_pads in enumerate(spatial_pads_low_high):
        pad_widths[i][0] = dim_i_pads[0]
        if i > 0 and dim_i_pads[0] < 0:
            crop_low[i - 1] = abs(dim_i_pads[0])
            pad_widths[i][0] = 0
            post_pad_crop = True

        pad_widths[i][1] = dim_i_pads[1]
        if i > 0 and dim_i_pads[1] < 0:
            crop_high[i - 1] = abs(dim_i_pads[1])
            pad_widths[i][1] = 0
            post_pad_crop = True

    # Pad the volume in voxel space.
    v_ch = torch.from_numpy(
        np.pad(v_ch.cpu().numpy(), pad_width=pad_widths, **np_pad_kwargs)
    ).to(v_ch)

    # Low pads require a translation of the affine matrix to maintain vox->real
    # mappings.
    pad_low = [p[0] for p in pad_widths[1:]]
    pad_low_vox_aff = torch.eye(affine_vox2real.shape[-1]).to(affine_vox2real)
    pad_low_vox_aff[:-1, -1] = -torch.Tensor(pad_low).to(affine_vox2real)
    new_affine = einops.einsum(
        affine_vox2real, pad_low_vox_aff, "... i j, ... j k -> ... i k"
    )
    # Crop any negative pad values.
    if post_pad_crop:
        crops = [c for c in zip(crop_low, crop_high)]
        v_ch, new_affine = crop_vox(v_ch, new_affine, *crops, **np_pad_kwargs)

    v = mrinr.utils.undo_vol_channels(v_ch, vol_vox)

    return (v, new_affine)


def _rotated_rect_max_area(w, h, theta) -> tuple[float, float]:
    """Find maximum inscribed rectangle
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.

    From <https://stackoverflow.com/a/16778797/13225248>, seconded by
    <https://math.stackexchange.com/a/4544735>.

    Parameters
    ----------
    w : _type_
        _description_
    h : _type_
        _description_
    theta : _type_
        _description_
    """

    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = np.abs(np.sin(theta)), np.abs(np.cos(theta))
    if side_short <= 2.0 * sin_a * cos_a * side_long or np.abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def _inscribe_voxel_fov_to_affine_space(
    target_fov_vox_corners, affine_vox2real, shear_tol=1e-5
) -> mrinr.typing.SingleFOV3D:
    raise NotImplementedError("Inscribe not implemented!")

    A = affine_vox2real.detach().cpu().numpy()
    T, R, Z, S = transforms3d.affines.decompose44(A)
    if (np.abs(S) > shear_tol).any():
        raise NotImplementedError(
            f"ERROR: Shear transforms not supported, given shear params {S}"
        )


# def _exscribe_fov_to_affine_space(
#     target_fov_corners: Optional[mrinr.typing.Coord3D] = None,
#     target_fov: Optional[mrinr.typing.SingleFOV3D] = None,
#     affine_vox2fov: Optional[mrinr.typing.SingleHomogeneousAffine3D] = None,
#     vox_expansion_buffer: float = 0.1,
# ) -> mrinr.typing.SingleFOV3D:
#     if (target_fov is not None and affine_vox2fov is None) or (
#         target_fov is None and affine_vox2fov is not None
#     ):
#         raise ValueError("Using target_fov requires fov_a2b")
#     if target_fov_corners is not None and (
#         target_fov is not None or affine_vox2fov is not None
#     ):
#         raise ValueError("target_fov_corners excludes target_fov")

#     # If given an fov bounding box and an affine transform, re-create the corner
#     # coordinates.
#     if target_fov is not None:
#         target_fov_vox = mrinr.coords.transform_coords(
#             target_fov, mrinr.coords.inv_affine(affine_vox2fov)
#         )
#         target_vox_corners = torch.cartesian_prod(*(target_fov_vox))
#         target_fov_vox_corners = mrinr.coords.transform_coords(
#             target_vox_corners, affine_vox2fov
#         )

#     exscribe_target_fov_vox_coords = torch.stack(
#         [
#             torch.amin(target_fov_vox_corners, dim=0) - vox_expansion_buffer,
#             torch.amax(target_fov_vox_corners, dim=0) + vox_expansion_buffer,
#         ],
#         0,
#     )

#     return exscribe_target_fov_vox_coords


def _exscribe_voxel_fov_to_affine_space(
    target_fov_vox_corners: mrinr.typing.Coord3D,
    vox_expansion_buffer: float = 0.1,
) -> mrinr.typing.SingleFOV3D:
    # Form a voxel-aligned bounding box from the target fov corners. It's "exscribed",
    # as opposed to "inscribed." That's not the English definition, but it should be.
    exscribe_target_fov_vox_coords = torch.stack(
        [
            torch.amin(target_fov_vox_corners, dim=0) - vox_expansion_buffer,
            torch.amax(target_fov_vox_corners, dim=0) + vox_expansion_buffer,
        ],
        0,
    )
    # Snap bounding box to voxel indices.
    # Translate lower bound "down"
    exscribe_target_fov_vox_coords[0] = exscribe_target_fov_vox_coords[0].floor()
    # Translate upper bound "up"
    exscribe_target_fov_vox_coords[1] = exscribe_target_fov_vox_coords[1].ceil()

    return exscribe_target_fov_vox_coords


def crop_to_fov_bb(
    vol: mrinr.typing.SingleVolume,
    affine_vox2real: mrinr.typing.SingleHomogeneousAffine3D,
    target_fov_real_coords: mrinr.typing.SingleFOV3D,
    affine_vox2fov_real_space: mrinr.typing.SingleHomogeneousAffine3D,
    new_fov_align_to_target_side: Literal["interior", "exterior"],
    return_crops=False,
    target_crop_vox_delta_tol: float = 0.1,
) -> (
    mrinr.coords.RealAffineSpaceVol
    | tuple[mrinr.coords.RealAffineSpaceVol, tuple[tuple[int, ...], ...]]
):
    # Find corners in the fov voxel space, then transform the corners back into real
    # space. Corners in voxel coordinates should be integers, so round for correcting
    # floating point error.
    fovox_space_fov = (
        mrinr.coords.transform_coords(
            target_fov_real_coords, mrinr.coords.inv_affine(affine_vox2fov_real_space)
        )
        .round()
        .abs()
    )
    # The cartesian product of coordinates fully describes the fov in voxel space, but
    # not always in real space, so go backwards to voxel space to recover the real fov.
    fovox_corners = torch.cartesian_prod(*(fovox_space_fov.T))
    # FoV voxel space -> real space -> volume voxel space.
    target_fov_real_corners = mrinr.coords.transform_coords(
        fovox_corners, affine_vox2fov_real_space
    )

    # Align the new fov such that the length of each side is evenly divisible by the
    # new spacing. The alignment may be pushed "inside" the original fov, or "outside"
    # of it.
    fov_align = new_fov_align_to_target_side.lower().strip()
    if fov_align in {"in", "inside", "internal", "interior"}:
        fov_align = "interior"
    elif fov_align in {"out", "outside", "external", "exterior"}:
        fov_align = "exterior"
    else:
        raise ValueError(
            f"ERROR: Invalid new_fov_align_to_target_side: {new_fov_align_to_target_side}"
        )

    if fov_align == "exterior":
        # Exscribe the fov in real coordinates, then transform back into volume voxel
        # coordinates, then exscribe *that* 3d rectangle to get the new fov.
        epsilon = 1e-2
        exscribe_real_fov = torch.stack(
            [
                torch.amin(target_fov_real_corners, dim=0) - epsilon,
                torch.amax(target_fov_real_corners, dim=0) + epsilon,
            ],
            0,
        )
        exscribe_real_corners = torch.cartesian_prod(*(exscribe_real_fov.T))
        exscribe_vox_corners = mrinr.coords.transform_coords(
            exscribe_real_corners, mrinr.coords.inv_affine(affine_vox2real)
        )

        new_fov_vox = _exscribe_voxel_fov_to_affine_space(
            exscribe_vox_corners, vox_expansion_buffer=target_crop_vox_delta_tol
        )
    elif fov_align == "interior":
        raise NotImplementedError("ERROR: interior inscribed FOVs not implemented")
    # round() shouldn't change any values here, it's just to account for possible
    # floating point errors.
    new_fov_vox = new_fov_vox.round().int()

    # Bring new fov bb coordinates back into real space.
    new_fov_real = mrinr.coords.transform_coords(new_fov_vox.float(), affine_vox2real)

    # Crop volume to the new fov.
    crops_low = new_fov_vox[0]
    crops_high = (torch.tensor(vol.shape[1:]) - 1) - new_fov_vox[1]

    crops = einops.rearrange(
        torch.stack([crops_low, crops_high], dim=0), "low_high coord -> coord low_high"
    )
    crops = [tuple(crops[i].tolist()) for i in range(crops.shape[0])]
    crop_vol, crop_affine_vox2real = crop_vox(vol, affine_vox2real, *crops)

    result = mrinr.coords.RealAffineSpaceVol(
        vol=crop_vol, affine_vox2real=crop_affine_vox2real, fov_bb_real=new_fov_real
    )
    if return_crops:
        result = (result, crops)
    return result


def _crop_lr_inside_smallest_lr(
    lr_vox_vol: torch.Tensor,
    affine_lr_vox2real: torch.Tensor,
    fr_fov_bb_real: torch.Tensor,
    affine_fr_vox2real: torch.Tensor,
    max_spacing_scale_factor: float,
) -> mrinr.coords._VoxRealAffineSpace:
    min_lr_fov_bb, affine_min_lr_vox2real = mrinr.coords.scale_fov_spacing(
        fr_fov_bb_real,
        affine_fr_vox2real,
        spacing_scale_factors=(max_spacing_scale_factor,) * 3,
        new_fov_align_direction="interior",
        set_affine_orig_to_fov_orig=True,
    )
    min_lr_shape = mrinr.coords.vox_shape_from_fov(
        min_lr_fov_bb, affine_min_lr_vox2real
    )

    # Determine current LR shape and crop bb to match the minimum LR shape.
    # Assume that the lr real-coordinate fov bb lines up with the vox-coordinate fov bb.
    lr_fov_bb_real = mrinr.coords.fov_bb_coords_from_vox_shape(
        affine_lr_vox2real, vox_vol=lr_vox_vol
    )
    lr_shape = mrinr.coords.vox_shape_from_fov(lr_fov_bb_real, affine_lr_vox2real)

    crop_lr_low = np.floor((np.array(lr_shape) - np.array(min_lr_shape)) / 2).astype(
        int
    )
    crop_lr_high = np.ceil((np.array(lr_shape) - np.array(min_lr_shape)) / 2).astype(
        int
    )
    crops_low_high = [
        (crop_lr_low[i], crop_lr_high[i]) for i in range(len(crop_lr_low))
    ]

    vox_conform_lr_vox_vol, vox_conform_lr_affine_vox2real = mrinr.vols.crop_vox(
        lr_vox_vol, affine_lr_vox2real, *crops_low_high
    )
    vox_conform_lr_fov_bb = mrinr.coords.fov_bb_coords_from_vox_shape(
        vox_conform_lr_affine_vox2real, vox_vol=vox_conform_lr_vox_vol
    )

    return mrinr.coords._VoxRealAffineSpace(
        vox_vol=vox_conform_lr_vox_vol,
        affine_vox2real=vox_conform_lr_affine_vox2real,
        fov_bb_real=vox_conform_lr_fov_bb,
    )


def _crop_frs_inside_lr(
    *fr_vox_vols: torch.Tensor,
    affine_fr_vox2real: torch.Tensor,
    lr_fov_bb_coords: torch.Tensor,
) -> mrinr.coords._BatchVoxRealAffineSpace:
    fr_vol = fr_vox_vols[0]
    fr_fov_bb = mrinr.coords.fov_bb_coords_from_vox_shape(
        affine_fr_vox2real, vox_vol=fr_vol
    )
    affine_real2fr_vox = torch.linalg.inv(affine_fr_vox2real)
    fr_fov_in_fr_vox_space = mrinr.coords.transform_coords(
        fr_fov_bb, affine_real2fr_vox
    )
    lr_fov_in_fr_vox_space = mrinr.coords.transform_coords(
        lr_fov_bb_coords, affine_real2fr_vox
    )

    EPSILON_FOV_DIFF = 1e-4
    diff_fov_low = lr_fov_in_fr_vox_space[0] - fr_fov_in_fr_vox_space[0]
    # If the fr and lr fovs are aligned within some epsilon, then the fr should be
    # cropped by at least one to prevent numerical errors with indexing later.
    diff_fov_low = torch.where(
        torch.isclose(
            diff_fov_low, diff_fov_low.round(), atol=EPSILON_FOV_DIFF, rtol=0
        ),
        diff_fov_low + 0.51,
        diff_fov_low,
    )
    crop_low = torch.clip(torch.ceil(diff_fov_low), min=0, max=torch.inf).int().tolist()
    diff_fov_high = fr_fov_in_fr_vox_space[1] - lr_fov_in_fr_vox_space[1]
    diff_fov_high = torch.where(
        torch.isclose(
            diff_fov_high, diff_fov_high.round(), atol=EPSILON_FOV_DIFF, rtol=0
        ),
        diff_fov_high + 0.51,
        diff_fov_high,
    )
    crop_high = (
        torch.clip(torch.ceil(diff_fov_high), min=0, max=torch.inf).int().tolist()
    )

    crops_low_high = [(crop_low[i], crop_high[i]) for i in range(len(crop_low))]

    vols = list()
    for v in fr_vox_vols:
        v_crop, affine_fr_vox2real_crop = mrinr.vols.crop_vox(
            v, affine_fr_vox2real, *crops_low_high
        )
        vols.append(v_crop)
    fr_crop_fov_bb = mrinr.coords.fov_bb_coords_from_vox_shape(
        affine_fr_vox2real_crop, vox_vol=vols[0]
    )

    return mrinr.coords._BatchVoxRealAffineSpace(
        vox_vols=tuple(vols),
        affine_vox2real=affine_fr_vox2real_crop,
        fov_bb_real=fr_crop_fov_bb,
    )


@torch.no_grad()
def robust_scale_spatial_data(
    spatial_data: mrinr.typing.SingleImage | mrinr.typing.SingleVolume,
    quantile_range: tuple[float, float],
    with_centering: bool,
    with_scaling: bool,
    unit_variance: bool = False,
    mask: Optional[mrinr.typing.SingleMaskImage | mrinr.typing.SingleMaskVolume] = None,
    return_scale_params: bool = False,
):
    spatial_dims = spatial_data.ndim - 1
    scale_tf = sklearn.preprocessing.RobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=tuple(quantile_range),
        unit_variance=unit_variance,
    )
    if mask is None:
        mask = torch.ones(
            (1,) + tuple(spatial_data.shape[1:]),
            dtype=torch.bool,
            device=spatial_data.device,
        )
    if mask.shape[0] != 1:
        mask = mask.unsqueeze(0)
    mask = mask.bool()
    # Only calculate scaling params over selected elements.
    sampled_feats = einops.rearrange(spatial_data, "c x y ... -> x y ... c")
    sampled_feats = sampled_feats[mask.movedim(0, -1).squeeze(-1)]
    scale_tf.fit(sampled_feats.detach().cpu().numpy())
    # Rescale the entire image/volume.
    scaled_feats = scale_tf.transform(
        einops.rearrange(
            spatial_data.detach().cpu().numpy(), "c x y ... -> (x y ...) c"
        )
    )
    if spatial_dims == 2:
        scaled_x = einops.rearrange(
            torch.from_numpy(scaled_feats).to(spatial_data),
            "(x y) c -> c x y",
            x=spatial_data.shape[1],
            y=spatial_data.shape[2],
        )
    elif spatial_dims == 3:
        scaled_x = einops.rearrange(
            torch.from_numpy(scaled_feats).to(spatial_data),
            "(x y z) c -> c x y z",
            x=spatial_data.shape[1],
            y=spatial_data.shape[2],
            z=spatial_data.shape[3],
        )
    else:
        raise ValueError(f"ERROR: Invalid spatial dims: {spatial_dims}")
    del scaled_feats, sampled_feats

    # Store center and scale params, if we want to later unnormalize the data.
    scale_center = (
        scale_tf.center_
        if scale_tf.center_ is not None
        else torch.zeros(scale_tf.n_features_in_, dtype=scaled_x.dtype)
    )
    scale_center = torch.as_tensor(scale_center, dtype=scaled_x.dtype)
    scale_scale = (
        scale_tf.scale_
        if scale_tf.scale_ is not None
        else torch.ones(scale_tf.n_features_in_, dtype=scaled_x.dtype)
    )
    scale_scale = torch.as_tensor(scale_scale, dtype=scaled_x.dtype)

    if return_scale_params:
        r = scaled_x, scale_center, scale_scale
    else:
        r = scaled_x

    return r


@torch.no_grad()
def distance_transform_mask(
    m: mrinr.typing.SingleMaskImage | mrinr.typing.SingleMaskVolume,
    spacing: tuple[float, ...],
) -> mrinr.typing.SingleScalarImage | mrinr.typing.SingleScalarVolume:
    m_np = m.cpu().squeeze(0).numpy()
    dt = scipy.ndimage.distance_transform_edt(
        m_np, sampling=spacing, return_distances=True
    )
    return torch.from_numpy(dt).to(device=m.device).expand_as(m).to(torch.float64)


# def gaussian_weigh_center_of_mass_mask(
#     m: torch.Tensor,
#     sigma_axis_aligned_physical_units: float | Tuple[float, ...],
#     radius_axis_aligned_physical_units: Optional[float | Tuple[float, ...]] = None,
# ):
#     mask = m.detach().cpu().numpy().astype(float)
#     center_coord = np.array(mask.shape) // 2
#     # Recover a gaussian kernel by filtering/convolving a dirac delta function.
#     dirac = np.zeros_like(mask)
#     dirac[tuple(center_coord.astype(int))] = 1.0
#     k = scipy.ndimage.gaussian_filter(
#         dirac, sigma=sigma, radius=voxel_radius, mode="nearest"
#     )
#     k = torch.from_numpy(k)

#     # Get index of the center of mass.
#     center_of_mass = np.array(scipy.ndimage.center_of_mass(mask)).round().astype(int)
#     # translate_center_to_com =


def prefilter_gaussian_blur(
    vol: torch.Tensor,
    src_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    sigma_scale_coeff: float = 2.0,
    sigma_truncate: float = 4.0,
):
    v = vol.detach().cpu().numpy()
    # Assume isotropic resampling.
    scale_ratio_high_to_low = (
        torch.mean(torch.Tensor(src_spacing)) / torch.mean(torch.Tensor(target_spacing))
    ).item()
    # Assume the src spacing is lower (i.e., higher spatial resolution) than the target
    # spacing.
    assert scale_ratio_high_to_low <= 1.0
    sigma = 1 / (sigma_scale_coeff * scale_ratio_high_to_low)
    if len(v.shape) == 4:
        sigma = (0, sigma, sigma, sigma)
    else:
        sigma = (sigma,) * 3
    v_filter = scipy.ndimage.gaussian_filter(
        v, sigma=sigma, order=0, mode="nearest", truncate=sigma_truncate
    )

    vol_blur = torch.from_numpy(v_filter).to(vol)

    return vol_blur
