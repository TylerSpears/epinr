#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import textwrap
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch

# Generated with np.random.randint(0, 2**32 - 1)
DEFAULT_SHUFFLE_SEED = 834057272


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create seed coordinates from a max volume"
    )
    parser.add_argument(
        "mask",
        type=Path,
        help="NIFTI file that contains the mask for seed point creation",
    )
    parser.add_argument(
        "seeds_per_vox_dim",
        type=int,
        help="Number of seed points within each voxel per dimension",
    )
    parser.add_argument(
        "output_seeds",
        type=Path,
        help="Output seeds .csv file that contains seed x, y, z real coordinates",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="Shuffle the seed coordinates before saving them, useful for early stopping "
        + "tractography while maintaining whole-brain tractogram, by default False.",
    )
    args = parser.parse_args()

    # Move import here to speed up the '-h' help command.
    import mrinr

    mask_d = mrinr.data.io.load_vol(args.mask, ensure_channel_dim=True)
    mask = mask_d["vol"].squeeze(0)
    mask = mask.bool()
    affine_vox2real = mask_d["affine"]

    vol_real_coords = mrinr.coords.affine_coord_grid(affine_vox2real, tuple(mask.shape))
    select_real_coords = vol_real_coords[mask.bool()].to(torch.float64)
    del vol_real_coords

    if args.seeds_per_vox_dim == 1:
        output_real_coords = select_real_coords
    else:
        select_vox_coords = mrinr.coords.transform_coords(
            select_real_coords, mrinr.coords.inv_affine(affine_vox2real)
        )
        del select_real_coords

        # Create voxel coordinate offsets to broadcast over the selected mask coords.
        vox_offsets = torch.stack(
            torch.meshgrid(
                [torch.linspace(-0.5, 0.5, steps=args.seeds_per_vox_dim + 2)] * 3,
                indexing="ij",
            ),
            dim=-1,
        )
        # Take off voxel borders
        vox_offsets = vox_offsets[1:-1, 1:-1, 1:-1]
        vox_offsets = vox_offsets.reshape(1, -1, 3)
        expanded_vox_coords = select_vox_coords.unsqueeze(1) + vox_offsets
        del select_vox_coords
        expand_vox_coords = einops.rearrange(
            expanded_vox_coords,
            "offsets n_select coords -> (offsets n_select) coords",
        )
        output_real_coords = mrinr.coords.transform_coords(
            expand_vox_coords, affine_vox2real
        )
        del expand_vox_coords

    output_real_coords = output_real_coords.cpu().numpy()
    if args.shuffle:
        print("Shuffling the seed coordinates")
        rng = np.random.default_rng(DEFAULT_SHUFFLE_SEED)
        rng.shuffle(output_real_coords, axis=0)

    output_real_coords = pd.DataFrame(output_real_coords, columns=["x", "y", "z"])

    print(f"Saving {output_real_coords.shape[0]} coordinates")

    # Try to save some information about the original seed mask/space.
    csv_preamble = f"""
    # Real-space seed coordinates
    # From mask file '{args.mask.name}'
    # {"Randomly shuffled" if args.shuffle else ""}
    # voxel FOV shape {str(tuple(mask.shape[-3:])).replace(",", "")}
    # affine vox to real space:
    # Row 1 {str(affine_vox2real[0].tolist()).replace(",", "")}
    # Row 2 {str(affine_vox2real[1].tolist()).replace(",", "")}
    # Row 3 {str(affine_vox2real[2].tolist()).replace(",", "")}
    # Row 4 {str(affine_vox2real[3].tolist()).replace(",", "")}"""

    csv_preamble = textwrap.dedent(csv_preamble).strip()
    with open(args.output_seeds, "wt") as f:
        f.write(csv_preamble)
        f.write("\n")

    output_real_coords.to_csv(
        args.output_seeds, index=False, sep=",", float_format="%g", mode="a"
    )

    print(f"Saved coordinates to {args.output_seeds}")
