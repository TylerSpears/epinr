#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code taken from torchio
# <https://github.com/fepegar/torchio/blob/f8638f33a60ef5e37b101a9deb5d74509b2b9970/src/torchio/data/io.py#L357>
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

# Matrices used to switch between LPS and RAS
FLIPXY_44 = np.diag([-1, -1, 1, 1])


def _to_itk_convention(matrix):
    """RAS to LPS."""
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.linalg.inv(matrix)
    return matrix


def _matrix_to_itk_transform(matrix, dimensions=3):
    matrix = _to_itk_convention(matrix)
    rotation = matrix[:dimensions, :dimensions].ravel().tolist()
    translation = matrix[:dimensions, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform


def _read_niftyreg_matrix(trsf_path):
    """Read a NiftyReg matrix and return it as a NumPy array."""
    matrix = np.loadtxt(trsf_path)
    matrix = np.linalg.inv(matrix)
    return torch.as_tensor(matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RAS MRtrix3 .txt affine matrix to ANTS/ITK .mat transform."
    )
    parser.add_argument(
        "mrtrix_affine",
        type=Path,
        help="MRtrix3 RAS affine transform .txt file, maps B->A (pull-back affine).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output .mat affine transform file.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force writing to output file, even if file already exists.",
    )
    args = parser.parse_args()

    # Check that the input affine is in RAS orientation.
    affine = _read_niftyreg_matrix(args.mrtrix_affine).numpy()
    ornt = "".join(nib.orientations.aff2axcodes(affine)).upper()
    if ornt != "RAS":
        raise ValueError(
            f"Expected RAS orient. for affine transform {args.mrtrix_affine}, got {ornt}"
        )
    itk_affine_tf = _matrix_to_itk_transform(affine, dimensions=3)

    if args.output.exists() and (os.path.getsize(args.output) > 0) and (not args.force):
        raise RuntimeError(f"Output file '{args.output}' already exists! Exiting.")
    itk_affine_tf.WriteTransform(str(args.output))
