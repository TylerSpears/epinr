# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Literal, Optional, TypedDict

import einops
import monai.transforms.utils
import nibabel as nib
import nibabel.filebasedimages
import numpy as np
import pandas as pd
import torch

# import skimage
import mrinr


class VolDataDict(TypedDict):
    vol: torch.Tensor
    affine: torch.Tensor


class ImDataDict(TypedDict):
    im: torch.Tensor
    affine: torch.Tensor
    header: Optional[dict]


class DWIVolDict(TypedDict):
    dwi: torch.Tensor
    affine: torch.Tensor
    grad_table: pd.DataFrame
    header: dict


def reorient_nib_im(
    im: nib.spatialimages.DataobjImage, target_orientation: str = "same"
):
    target_ornt_str = target_orientation.strip()
    if target_ornt_str.lower() != "same":
        src_code = nib.orientations.aff2axcodes(im.affine)
        target_code = tuple(target_ornt_str.upper())
        if src_code != target_code:
            src_ornt = nib.orientations.axcodes2ornt(src_code)
            target_ornt = nib.orientations.axcodes2ornt(target_code)
            src2target_ornt = nib.orientations.ornt_transform(src_ornt, target_ornt)
            ret = im.as_reoriented(src2target_ornt)
        else:
            ret = im
    else:
        ret = im

    return ret


def load_im(
    im_f: Path,
    affine_el2coords: Optional[Path | np.ndarray] = None,
    ensure_channel_dim=False,
) -> ImDataDict:
    # Import transforms3d here to avoid an dependency for an edge-case use.
    import transforms3d

    # Try reading the file with nibabel first, to see if it has a singleton z-dimension
    # that can be squeezed to 2D.

    try:
        pseudo_vol_data = load_vol(
            im_f,
            affine_el2coords=affine_el2coords,
            reorient_im_to="RAS",
            ensure_channel_dim=True,
        )
    except nibabel.filebasedimages.ImageFileError:
        im = None
        affine = None
    else:
        im = pseudo_vol_data["vol"]
        # load_vol should handle any affine overrides and/or reorientations.
        affine = pseudo_vol_data["affine"]

        # im should have 1 and only 1 spatial dimension as a singleton.
        spatial_shape = tuple(im.shape[1:])
        if spatial_shape.count(1) != 1:
            raise ValueError(
                f"Image has shape '{spatial_shape}', but only one dimension can be a "
                + "singleton."
            )
        squeeze_spatial_dim = spatial_shape.index(1)

        im = im.squeeze(squeeze_spatial_dim + 1)
        dim_sub_select_tuple = tuple(set(range(3)) - {squeeze_spatial_dim})
        vec_sub_select_idx = np.array(dim_sub_select_tuple)
        rot_mat_sub_select_dim_idx = np.ix_(dim_sub_select_tuple, dim_sub_select_tuple)
        # Decompose the 3D affine, sub-select the components, then re-compose.
        affine = affine.cpu().numpy()
        t, r, zoom, s = transforms3d.affines.decompose(affine)
        # The rotation matrix has dependent columns, so it must be decomposed into
        # euler angles.
        r_z, r_y, r_x = nib.eulerangles.mat2euler(r)
        # Only take rotation about the squeezed axis, i.e. in the 2D plane of the
        # remaining dimensions.
        euler2mat_kwargs = dict(
            z=r_z if squeeze_spatial_dim == 2 else 0,
            y=r_y if squeeze_spatial_dim == 1 else 0,
            x=r_x if squeeze_spatial_dim == 0 else 0,
        )
        M_r = nib.eulerangles.euler2mat(**euler2mat_kwargs)
        # Recompose into a 2D homogeneous affine matrix. Ignore shearing.
        affine = transforms3d.affines.compose(
            T=t[vec_sub_select_idx],
            R=M_r[rot_mat_sub_select_dim_idx],
            Z=zoom[vec_sub_select_idx],
        )
        affine = torch.from_numpy(affine).to(torch.float32)

    # If the file cannot be read by nibabel, then try with skimage and a provided affine
    # file/matrix.
    if im is None:
        raise NotImplementedError("Reading images with skimage is not yet supported.")
        # if affine_f is None:
        #     raise ValueError("No affine file provided.")
        # im = skimage.io.imread(im_f, as_gray=False)
        # im = torch.from_numpy(im)
        # if im.ndim == 3:
        #     # Move channel dimension to the front.
        #     im = im.movedim(-1, 0)
        # elif ensure_channel_dim:
        #     # Add a channel dimension.
        #     im = im.unsqueeze(0)
        # affine = np.loadtxt(affine_f)
        # affine = torch.from_numpy(affine).to(torch.float32)

    if not ensure_channel_dim:
        im = im.squeeze(0)
    elif ensure_channel_dim and im.ndim == 2:
        im = im.unsqueeze(0)
    return {"im": im, "affine": affine}


def save_im_as_pseudo_vol(
    im: mrinr.typing.SingleImage,
    affine: mrinr.typing.SingleHomogeneousAffine2D,
    output_f: Path | str,
    expand_dim: Literal[0, 1, 2] = 2,
) -> None:
    # Import transforms3d here to avoid a dependency.
    import transforms3d

    im = torch.as_tensor(im)
    # Nifti files do not support bools.
    if im.dtype == torch.bool:
        im = im.to(torch.uint8)
    remove_singleton_channel = True if im.ndim == 2 else False
    # Standardize to 2D with a channel dim.
    im = mrinr.utils.ensure_image_channels(im, batch=False)
    if expand_dim == 0:
        pseudo_vol = einops.rearrange(im, "channel x y -> 1 x y channel")
    elif expand_dim == 1:
        pseudo_vol = einops.rearrange(im, "channel x y -> x 1 y channel")
    elif expand_dim == 2:
        pseudo_vol = einops.rearrange(im, "channel x y -> x y 1 channel")
    pseudo_vol = pseudo_vol.detach().cpu().numpy()

    affine = torch.as_tensor(affine).detach().cpu().numpy()
    # Decompose the 2D affine, expand into 3D where the remaining parameters are
    # identity, then re-compose.
    t, r, zoom, s = transforms3d.affines.decompose(affine)
    # Translation 2D -> 3D
    t: np.ndarray
    t = t.tolist()
    t.insert(expand_dim, 0)
    t = np.asarray(t)

    # Zoom/scale 2D -> 3D
    zoom: np.ndarray
    zoom = zoom.tolist()
    zoom.insert(expand_dim, 1.0)
    zoom = np.asarray(zoom)

    # Rotation 2D -> 3D
    # Create rotation matrix using monai.
    euler_angle_2d = np.arccos(r[0, 0])
    euler_angles_3d = np.zeros(3)
    euler_angles_3d[expand_dim] = euler_angle_2d
    r_3d = monai.transforms.utils.create_rotate(
        spatial_dims=3, radians=euler_angles_3d, backend="numpy"
    )[:-1, :-1]

    # Ignore shearing.
    affine_3d = transforms3d.affines.compose(T=t, R=r_3d, Z=zoom)
    if remove_singleton_channel:
        pseudo_vol = np.squeeze(pseudo_vol, -1)

    nib.save(nib.Nifti1Image(pseudo_vol, affine_3d), output_f)


def load_vol(
    vol_f: Path,
    affine_el2coords: Optional[Path | np.ndarray] = None,
    reorient_im_to: str = "same",
    ensure_channel_dim=False,
    spatial_slices: Optional[tuple[slice, slice, slice]] = None,
    channel_slice: Optional[slice] = None,
) -> VolDataDict:
    vol_im = nib.load(vol_f)
    # If an affine is provided, assume that it applies to the volume/image in its
    # raw form, i.e. before requested reorientation.
    if affine_el2coords is not None:
        if isinstance(affine_el2coords, Path):
            a = np.loadtxt(affine_el2coords, comments="#")
        else:
            a = np.asarray(affine_el2coords)
        vol_im = nib.Nifti1Image(vol_im.get_fdata(), affine=a, header=vol_im.header)

    reorient_im_to = reorient_im_to.strip()
    if reorient_im_to.lower() != "same":
        src_code = nib.orientations.aff2axcodes(vol_im.affine)
        target_code = tuple(reorient_im_to.upper())
        src_ornt = nib.orientations.axcodes2ornt(src_code)
        target_ornt = nib.orientations.axcodes2ornt(target_code)
        src2target_ornt = nib.orientations.ornt_transform(src_ornt, target_ornt)
        vol_im = vol_im.as_reoriented(src2target_ornt)

    if spatial_slices is not None or channel_slice is not None:
        slices = spatial_slices if spatial_slices is not None else ((slice(None),) * 3)
        # Add channel slice at the end of the slice tuple.
        if channel_slice is not None:
            slices = slices + (channel_slice,)
        vol_im = vol_im.slicer[slices]

    try:
        vol = torch.from_numpy(
            vol_im.get_fdata(caching="unchanged", dtype=vol_im.get_data_dtype())
        )
    except ValueError:
        vol = torch.from_numpy(
            vol_im.get_fdata(caching="unchanged").astype(vol_im.get_data_dtype())
        )
    if vol.ndim == 4:
        vol = einops.rearrange(vol, "a b c channel -> channel a b c")
    # Create a channel dimension if requested.
    elif ensure_channel_dim:
        vol = vol.unsqueeze(0)
    affine = torch.from_numpy(vol_im.header.get_best_affine()).to(torch.float32)
    header = dict(vol_im.header)

    return {"vol": vol, "affine": affine, "header": header}


def load_dwi(
    dwi_vol_f: Path,
    grad_mrtrix_f: Path,
    reorient_im_to: str = "same",
) -> DWIVolDict:
    vol_data = load_vol(dwi_vol_f, reorient_im_to=reorient_im_to)
    dwi = vol_data["vol"]
    affine = vol_data["affine"]
    header = vol_data["header"]

    grad_table = np.loadtxt(grad_mrtrix_f, comments="#")
    grad_table = pd.DataFrame(grad_table, columns=("x", "y", "z", "b"))

    return {"dwi": dwi, "affine": affine, "grad_table": grad_table, "header": header}
