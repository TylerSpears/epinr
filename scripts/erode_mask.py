#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import einops
import nibabel as nib
import numpy as np
import skimage


def ellipsoid_mask(real_coord, sphere_size_mm):
    return (np.sum(real_coord**2 / sphere_size_mm**2, -1) <= 1.0).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Erode a binary mask with non-isotropic sphere kernel in mm"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input binary mask file to be eroded; may have non-isotropic spacing",
    )
    parser.add_argument(
        "erode_mm",
        type=float,
        help="Scalar erosion amount in mm for all dimensions",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output eroded binary mask file",
    )
    args = parser.parse_args()
    mask_im = nib.load(args.input)
    erode_mm = args.erode_mm

    if erode_mm == 0.0:
        eroded_mask = mask_im.get_fdata().astype(np.uint8)
    else:
        spacing = nib.affines.voxel_sizes(mask_im.affine)
        k_size_vox = int(2 * (np.ceil(erode_mm / min(spacing))) + 1)
        k_vox_coord = np.stack(
            np.meshgrid(*([np.arange(k_size_vox)] * 3), indexing="ij"), axis=-1
        )
        aff_vox2center = np.array(
            [
                [1, 0, 0, -(k_size_vox - 1) / 2],
                [0, 1, 0, -(k_size_vox - 1) / 2],
                [0, 0, 1, -(k_size_vox - 1) / 2],
                [0, 0, 0, 1],
            ]
        ).astype(float)
        aff_vox_center2mm = np.array(
            [
                [spacing[0], 0, 0, 0],
                [0, spacing[1], 0, 0],
                [0, 0, spacing[2], 0],
                [0, 0, 0, 1],
            ]
        ).astype(float)
        aff_vox2mm = aff_vox_center2mm @ aff_vox2center
        k_mm_coord = (
            einops.einsum(
                aff_vox2mm[..., :3, :3], k_vox_coord, "... i j, ... j -> ... i"
            )
            + aff_vox2mm[..., :3, -1]
        )
        kernel = ellipsoid_mask(k_mm_coord, erode_mm)
        eroded_mask = skimage.morphology.binary_erosion(
            mask_im.get_fdata().astype(bool), kernel.astype(bool)
        ).astype(np.uint8)

    nib.save(nib.Nifti1Image(eroded_mask, mask_im.affine), args.output)
