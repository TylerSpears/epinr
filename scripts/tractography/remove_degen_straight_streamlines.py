#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

DEBUG = bool(int(os.environ.get("DEBUG", "0")))
import argparse
import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import nibabel
import numpy as np


def find_runs(x):
    """Find runs of consecutive items in an array."""
    # From <https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065>

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]
    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        # find run values
        run_values = x[loc_run_start]
        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def cumul_arc_len(pos: np.ndarray) -> np.ndarray:
    return np.concatenate(
        ([0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    )


def fwd_finite_difference(
    x,
    y,
    h,
) -> np.ndarray:
    d = np.diff(y, axis=0) / h
    return d


def streamline_curvature(p: np.ndarray) -> np.ndarray:
    vec_norm = lambda x: np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    s = cumul_arc_len(p)

    # Tractography step size should be constant.
    eps = np.linalg.norm(np.diff(p, axis=0), ord=2, axis=1, keepdims=True)
    # Calculate curvature and torsion using the explicit formulas from r(s). See
    # <https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas#Other_expressions_of_the_frame>
    r_p = fwd_finite_difference(x=s, y=p, h=eps)
    r_pp = fwd_finite_difference(x=s, y=r_p, h=eps[1:])
    # Cut off the first point where FD is not defined.
    r_p = r_p[:-1]

    curvature = vec_norm(np.cross(r_p, r_pp)) / np.maximum(
        vec_norm(r_p) ** 3, 1e-8
    )  # Avoid division by zero.
    curvature = curvature.flatten()

    return curvature


def _streamline_curv_filter_worker_fn(
    p: np.ndarray,
    curvature_thresh: float,
    consec_steps_thresh: int,
) -> bool:
    if p.shape[0] <= 2:
        is_degenerate = False
    else:
        consec_steps_thresh = min(consec_steps_thresh, p.shape[0] - 1)
        kappa = streamline_curvature(p=p)
        consec_vals, _, consec_lengths = find_runs(kappa <= curvature_thresh)
        try:
            is_degenerate = np.any(
                (consec_vals) & (consec_lengths >= consec_steps_thresh)
            ).item()
        except Exception as e:
            print("ERROR")
            raise e
    return not is_degenerate


def mp_bundle_scaled_curvature_torsion_length(
    streamlines: list[np.ndarray],
    num_processes: int,
    curvature_thresh: float,
    consec_steps_thresh: int,
) -> list[np.ndarray]:
    if len(streamlines) == 0:
        valid_streamlines = list()
    else:
        # Calculate curvature and torsion for each streamline, then take the integral of
        # curvature/torsion and divide by the streamline length.
        chunksize = max(1, len(streamlines) // (num_processes * 4))
        with multiprocessing.Pool(processes=num_processes) as pool:
            valid_streamline_mask = list(
                pool.map(
                    partial(
                        _streamline_curv_filter_worker_fn,
                        curvature_thresh=curvature_thresh,
                        consec_steps_thresh=consec_steps_thresh,
                    ),
                    streamlines,
                    chunksize=chunksize,
                ),
            )
        valid_streamlines = list(
            s for s, valid in zip(streamlines, valid_streamline_mask) if valid
        )

    return valid_streamlines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove degenerate/straight streamlines from a .tck file.",
    )
    parser.add_argument(
        "in_tck_file",
        type=Path,
        help="Path to input .tck file.",
    )
    parser.add_argument(
        "out_tck_file",
        type=Path,
        help="Path to output .tck file.",
    )
    parser.add_argument(
        "-c",
        "--min-curvature",
        type=float,
        default=1e-5,
        help="Minimum curvature threshold. "
        "Streamlines with curvature below this value will be removed.",
    )
    parser.add_argument(
        "-d",
        "--num-consec-steps-degenerate",
        type=int,
        default=10,
        help="Number of consecutive streamline steps that, if curvature is "
        "<= min curvature, denotes the streamline as invalid.",
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=max(os.cpu_count() - 1, 1),
        help="Number of processes to use for parallel processing.",
    )
    if DEBUG:
        args = parser.parse_args(
            [
                # "/home/tas6hh/mnt/magpie/outputs/results/fenri/2025-05-10T13_17_11_fenri_unet_2.5mm-val_test/tractography/hcp-187345_in-2.5mm_snr-30_csd-b1000x64/tractography/CST_left/unfiltered_CST_left_seed-tract-density.tck",
                "/tmp/tmp.bn6EukK8MZ_refine_trax_streamlines/ICP_left/tmp_ICP_left_seed-tract-density.tck",
                "tmp_test_out.tck",
                "-n",
                "1",
            ]
        )
    else:
        args = parser.parse_args()

    print(f"Loading streamlines from {args.in_tck_file}")
    tract_obj = nibabel.streamlines.load(args.in_tck_file.resolve())
    streamlines = tract_obj.streamlines
    print("Checking streamlines for degenerate/straight segments.")
    valid_streamlines = mp_bundle_scaled_curvature_torsion_length(
        streamlines=streamlines,
        num_processes=args.num_processes,
        curvature_thresh=args.min_curvature,
        consec_steps_thresh=args.num_consec_steps_degenerate,
    )

    if len(valid_streamlines) == 0:
        print(
            f"Warning: No valid streamlines found in {args.in_tck_file}. "
            "Outputting empty .tck file.",
        )
    elif len(valid_streamlines) == len(streamlines):
        print("No degenerate streamlines found, copying input to output.")
        shutil.copyfile(args.in_tck_file, args.out_tck_file)
    elif len(valid_streamlines) < len(streamlines):
        print(
            f"Removed {len(streamlines) - len(valid_streamlines)}/{len(streamlines)}="
            + f"{((len(streamlines) - len(valid_streamlines)) / len(streamlines)) * 100:.4f}% "
            + "degenerate streamlines.",
        )
        new_tract_obj = nibabel.streamlines.Tractogram(
            streamlines=valid_streamlines,
            affine_to_rasmm=tract_obj.tractogram.affine_to_rasmm,
            data_per_streamline=tract_obj.tractogram.data_per_streamline,
            data_per_point=tract_obj.tractogram.data_per_point,
        )
        new_tract_file = nibabel.streamlines.TckFile(new_tract_obj)
        print(f"Saving filtered streamlines to {args.out_tck_file}")
        args.out_tck_file.parent.mkdir(parents=True, exist_ok=True)
        if not args.out_tck_file.exists():
            args.out_tck_file.touch()
        new_tract_file.save(args.out_tck_file.resolve())
    else:
        raise RuntimeError("Unexpected error: more valid streamlines than input.")
