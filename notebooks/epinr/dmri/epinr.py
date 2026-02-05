#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run EPINR susceptibility distortion correction."""

import argparse
import ast
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import papermill
import torch
from box import Box
from data_utils import (
    EMPTY_RUN_TOKEN,
    EMPTY_SESSION_TOKEN,
    PE_DIR_ALIASES,
    dataset_table_cols,
)

import mrinr


def multi_file_hash_str(*files) -> str:
    """Get a hash string representing the contents of multiple files."""
    hash_md5 = hashlib.md5()
    # Sort files by size to ensure a consistent order.
    files = sorted([Path(f).resolve() for f in files], key=lambda x: os.path.getsize(x))
    for f in files:
        with open(f, "rb") as f_in:
            for chunk in iter(lambda: f_in.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


#
class ParseKwargs(argparse.Action):
    """Parse key-value pairs from the command line into a dictionary.

    Taken from
    <https://sumit-ghosh.com/posts/parsing-dictionary-key-value-pairs-kwargs-argparse-python/>
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = str(value)
            getattr(namespace, self.dest)[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EPINR susceptibility distortion correction."
    )
    subparsers = parser.add_subparsers(
        help="Run in single subject or batch subject mode."
    )
    single_subj_parser = subparsers.add_parser(
        "single", help="Run EPINR on a single subject by specifying input files."
    )
    batch_subj_parser = subparsers.add_parser(
        "batch",
        help="Run EPINR on multiple subjects by specifying a table of subjects.",
    )

    ## Single subject arguments.
    single_subj_parser.add_argument(
        "-m",
        "--b0",
        type=Path,
        required=True,
        help="File containing a preprocessed single distorted b0 volume.",
    )
    single_subj_parser.add_argument(
        "-f",
        "--t1w",
        type=Path,
        required=True,
        help="File containing a preprocessed T1w volume rigidly aligned to the b0.",
    )
    single_subj_parser.add_argument(
        "-s",
        "--b0-mask",
        type=Path,
        required=True,
        help="File containing a brain mask for the distorted b0 volume. "
        "This should be in the same space as the b0 image and can be used to restrict "
        "the EPINR model fitting to brain voxels.",
    )
    single_subj_parser.add_argument(
        "-k",
        "--t1w-mask",
        type=Path,
        required=True,
        help="File containing a brain mask for the T1w volume. "
        "This should be in the same space as the T1w image and can be used to restrict "
        "the EPINR model fitting to brain voxels.",
    )
    pe_dir_choices = list()
    for k in PE_DIR_ALIASES.keys():
        pe_dir_choices.extend(PE_DIR_ALIASES[k])
    pe_dir_vals = ", ".join(pe_dir_choices[:-1]) + f", and {pe_dir_choices[-1]}"
    single_subj_parser.add_argument(
        "-p",
        "--pe-dir",
        required=True,
        type=str,
        help=f"""Phase-encoding direction of the distorted b0 volume, not case-sensitive.\n
        Valid options include: {pe_dir_vals}.""",
    )
    single_subj_parser.add_argument(
        "-r",
        "--total-readout-time",
        type=float,
        help="""Total readout time of the b0 scan in seconds. This follows the FSL
        definition of total readout time. See
        <https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/FAQ/index.html>
        and
        <https://bids-specification.readthedocs.io/en/stable/glossary.html#objects.metadata.TotalReadoutTime>
        for more details. This value is used to scale the predicted susceptibility field, so
        it must be consistent with the value that would be given to a topup/eddy run.""",
    )
    single_subj_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output reconstructed susceptibility field in Hz.",
    )
    single_subj_parser.add_argument(
        "--output-b0",
        type=Path,
        default=None,
        help="Output b0 volume corrected by the predicted susceptibility field.",
    )
    single_subj_parser.add_argument(
        "--output-weights",
        type=Path,
        default=None,
        help="Output file containing the weights of the final EPINR model.",
    )
    single_subj_parser.add_argument(
        "--output-debug-dir",
        type=Path,
        default=None,
        help="Output directory used for saving other debug/validation files.",
    )
    single_subj_parser.add_argument(
        "--extra-info",
        action=ParseKwargs,
        required=False,
        nargs="+",
        default=dict(),
        help=f"""Extra subject-specific files or information to be passed to the
            EPINR run, such as subject id, dataset name, or topup-corrected comparison
            volumes, in a key-value format.\n
            Accepted keys include: {", ".join(dataset_table_cols)}.\n
            Key-value pairs should be given as 'kwarg=value', and values will be run
            through 'ast.literal_eval()'. If a kwarg should be a string, ensure that the
            string is quoted, e.g. 'kwarg="value".""",
    )

    ## Batch subject run mode.
    batch_subj_parser.add_argument(
        "-t",
        "--dataset-table",
        type=Path,
        help=f"""CSV file containing a table of subjects on which to run EPINR.\n
        Valid column names include: {", ".join(dataset_table_cols)}.\n
        Not all columns are required.""",
    )
    batch_subj_parser.add_argument(
        "-d",
        "--dataset-dirs",
        action=ParseKwargs,
        nargs="+",
        default=dict(),
        required=True,
        help="""Dataset-specific base directories in a key-value format. Key-value pairs
        should be given as 'dataset_name=/path/to/dataset_base_dir'. The dataset_name
        should match the `dataset_name` column in the dataset table.""",
    )
    batch_subj_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory to save all subject outputs.",
    )
    batch_out_dir_group = batch_subj_parser.add_mutually_exclusive_group(required=False)
    batch_out_dir_group.add_argument(
        "--continue",
        action="store_true",
        help="Continue a previous batch run by searching for the temporary output directory.",
        dest="continue_prev_run",
    )
    batch_out_dir_group.add_argument(
        "--prepend-output-timestamp",
        action="store_true",
        help="""Prepend a timestamp to the output directory.
        This helps avoid overwriting previous runs. If you would like to continue
        a previous run, this flag should not be set.""",
    )

    # Add arguments shared by both subcommands.
    for p in (single_subj_parser, batch_subj_parser):
        p.add_argument(
            "--params",
            action=ParseKwargs,
            required=False,
            nargs="+",
            default=None,
            help="""Parameters for running EPINR in a key-value format. Key-value pairs
            should be given as 'kwarg=value', and values will be run through
            'ast.literal_eval()'. Keys with a dot '.' are considered to be nested keys
            in the parameters dictionary.""",
        )
        p.add_argument(
            "--device",
            type=str,
            default="cuda:0" if torch.cuda.is_available() else "cpu",
            help="Pytorch device string to run EPINR.",
        )
        p.add_argument(
            "--no-param-defaults",
            action="store_true",
            help="""Skip merging parameters given in --params with default parameters.
            By default, parameters given in --params will be merged with the default
            parameters given in the notebook. If this flag is set, only parameters given in
            --params will be used, and all parameters must be explicitly set.""",
        )
        p.add_argument(
            "--save-debug-outputs",
            action="store_true",
            help="""Save debug/validation outputs that record details about
            model performance during the optimization process.""",
        )
        p.add_argument(
            "--no-compile-hessian",
            action="store_true",
            help="""Do not compile the Hessian for EPINR optimization.
            Some GPUs throw errors when combining pytorch's `torch.compile()` with
            `torch.func.hessian()`, so this flag allows skipping Hessian compilation while
            still allowing for Hessian-based regularization.""",
        )

    single_subj_parser.set_defaults(_subcommand="single")
    batch_subj_parser.set_defaults(_subcommand="batch")

    args = parser.parse_args()
    print(args)

    this_script_path = Path(__file__).resolve()
    nb_path = this_script_path.parent / "epinr.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Could not find notebook at: {nb_path}.")
    # Set env var for the notebook to find its own path.
    os.environ["JPY_SESSION_NAME"] = str(nb_path)

    # Collect and parse CLI arguments for passing to papermill.
    nb_kwargs = dict()
    nb_kwargs["in_pml"] = True  # Indicate to the notebook that it's run via papermill.
    nb_kwargs["pml_subj_run_mode"] = args._subcommand
    nb_kwargs["device"] = args.device
    nb_kwargs["pml_merge_with_default_params"] = not args.no_param_defaults
    nb_kwargs["save_debug_outputs"] = args.save_debug_outputs
    nb_kwargs["compile_hessian"] = not args.no_compile_hessian
    if args.params is not None:
        # Parse run parameters specified via the CLI. May include nested keys via dot
        # notation, which will be handled by Box.
        b = Box(default_box=True, box_dots=True)
        for k, v in args.params.items():
            b[k] = v
        nb_kwargs["pml_params"] = b.to_dict()
    else:
        nb_kwargs["pml_params"] = dict()

    if nb_kwargs["pml_subj_run_mode"] == "batch":
        if args.continue_prev_run and args.prepend_output_timestamp:
            raise ValueError(
                """Cannot both continue a previous run and prepend a timestamp to the
                output directory."""
            )

        nb_kwargs["continue_previous_run"] = args.continue_prev_run
        nb_kwargs["pml_dataset_table_f"] = str(args.dataset_table.resolve())
        nb_kwargs["pml_batch_prepend_timestamp"] = args.prepend_output_timestamp
        output_dir = args.output_dir.expanduser().resolve()
        out_base_dir = output_dir.parent
        if args.prepend_output_timestamp:
            timestamp_str = mrinr.utils.timestamp_now()
            out_dir = out_base_dir / f"{timestamp_str}_{output_dir.name}"
        else:
            out_dir = out_base_dir
        nb_kwargs["pml_batch_out_dir"] = str(args.output_dir.resolve())
        # Resolve dataset base directories.
        data_dirs = dict()
        for k, v in args.dataset_dirs.items():
            data_dirs[k] = str(Path(v).expanduser().resolve())
        nb_kwargs["pml_batch_dataset_dirs"] = data_dirs

    elif nb_kwargs["pml_subj_run_mode"] == "single":
        # raise NotImplementedError("Single subject mode not yet implemented.")
        nb_kwargs["continue_previous_run"] = False
        # Create a fake subject id by hashing the input files together. This ensures
        # consistent seeding when given the same input files.
        fake_subj_id = multi_file_hash_str(
            args.b0, args.t1w, args.b0_mask, args.t1w_mask
        )
        subj_id = f"subj-tmp{fake_subj_id[:16]}"

        single_subj_row = (
            dict(
                # Mostly-dummy values for some columns that may accept non-None values.
                subj_id=subj_id,
                dataset_name="EMPTY",
                session_id=EMPTY_SESSION_TOKEN,
                run_id=EMPTY_RUN_TOKEN,
            )
            | args.extra_info
            | dict(
                # Values taken from the CLI arguments.
                dwi=str(args.b0.expanduser().resolve()),
                t1w_reg_dwi=str(args.t1w.expanduser().resolve()),
                dwi_mask=str(args.b0_mask.expanduser().resolve()),
                t1w_mask=str(args.t1w_mask.expanduser().resolve()),
                pe_dir=PE_DIR_ALIASES[str(args.pe_dir).lower()],
                total_readout_time_s=args.total_readout_time,
            )
        )
        # Ensure all paths are absolute paths, not relative.
        for k in single_subj_row.keys():
            try:
                v = Path(single_subj_row[k]).expanduser().resolve()
            except TypeError:
                continue
            if v.exists():
                single_subj_row[k] = str(v)
        # Fill in the remaining cols with None values.
        missing_cols = set(dataset_table_cols) - set(single_subj_row.keys())
        for c in missing_cols:
            single_subj_row[c] = None
        tmp_table_fd, tmp_table_name = tempfile.mkstemp(
            prefix="epinr_single_subj_table_", suffix=".csv"
        )
        os.close(tmp_table_fd)
        tmp_table_f = Path(tmp_table_name)
        pd.DataFrame([single_subj_row]).to_csv(tmp_table_f, index=False)
        nb_kwargs["pml_dataset_table_f"] = str(tmp_table_f.resolve())
        # Set output paths.
        if args.output_debug_dir is not None and not nb_kwargs["save_debug_outputs"]:
            nb_kwargs["save_debug_outputs"] = True
        elif args.output_debug_dir is None and nb_kwargs["save_debug_outputs"]:
            raise ValueError(
                """ --save-debug-outputs is set but --output-debug-dir is not given."""
            )
        nb_kwargs["pml_single_subj_out_fmap"] = (
            str(args.output.resolve()) if args.output else None
        )
        nb_kwargs["pml_single_subj_out_corrected_b0"] = (
            str(args.output_b0.resolve()) if args.output_b0 else None
        )
        nb_kwargs["pml_single_subj_out_weights"] = (
            str(args.output_weights.resolve()) if args.output_weights else None
        )
        nb_kwargs["pml_single_subj_out_debug_dir"] = (
            str(args.output_debug_dir.resolve()) if args.output_debug_dir else None
        )

    # print(args)
    # print(f"Running notebook at: {nb_path} with args:\n{nb_kwargs}")

    papermill.execute_notebook(
        input_path=str(nb_path),
        output_path=None,
        parameters=nb_kwargs,
        report_mode=False,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
        cwd=str(this_script_path.parent),
        progress_bar=False,
    )

    if nb_kwargs["pml_subj_run_mode"] == "single":
        # Clean up the temporary dataset table file.
        tmp_table_f.unlink()

    print("Done running EPINR.")
