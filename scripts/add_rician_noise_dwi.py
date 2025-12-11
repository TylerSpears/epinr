#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import secrets
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

S0_B0_QUANTILE = 0.95


def create_subj_rng_seed(base_rng_seed: int, subj_id: Union[int, str]) -> int:
    try:
        subj_int = int(subj_id)
    except ValueError as e:
        if not isinstance(subj_id, str):
            raise e
        # Max hexdigest length that can fit into a 64-bit integer is length 8.
        hash_str = (
            hashlib.shake_128(subj_id.encode(), usedforsecurity=False)
            .hexdigest(8)
            .encode()
        )
        subj_int = int(hash_str, base=16)

    return base_rng_seed ^ subj_int


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add Rician-distributed noise to DWI data"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("-s", "--subj-id", type=str, default=None)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bvals = np.loadtxt(args.bvals, comments="#")
    b0_idx = ((bvals >= -100) & (bvals <= 100)).nonzero()[0]
    dwi_im = nib.load(args.input)
    dwi = dwi_im.get_fdata().astype(np.float32)
    b0s = np.clip(dwi[..., b0_idx], a_min=0, a_max=None)
    S0 = np.mean(b0s, axis=-1, keepdims=True)
    S0 = np.quantile(S0, S0_B0_QUANTILE) * np.ones_like(dwi)

    # Generate Rician distributed noise.
    if args.subj_id is not None:
        seed = create_subj_rng_seed(0, subj_id=args.subj_id)
    else:
        seed = secrets.randbits(64)
        print("Random seed created for noise generation: ", seed)
    # Create a random number generator with a fixed seed for reproducibility
    sigma = S0 / args.snr
    sigma = np.broadcast_to(sigma.astype(np.float32), dwi.shape)
    rng = np.random.default_rng(seed=seed)
    N_real = rng.normal(0, sigma, size=dwi.shape)
    N_complex = rng.normal(0, sigma, size=dwi.shape)
    # Equal to abs(dwi + complex_noise)
    S = np.sqrt((dwi + N_real) ** 2 + N_complex**2)

    if args.mask is not None:
        mask = nib.load(args.mask).get_fdata().astype(bool)
    else:
        mask = np.ones(dwi.shape, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., np.newaxis]
    S = S * mask
    S = S.astype(np.float32)
    S_im = nib.Nifti1Image(S, dwi_im.affine, header=dwi_im.header)
    nib.save(S_im, args.output)
