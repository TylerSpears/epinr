# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path

__all__ = ["_find_mrtrix_bin"]


def _find_mrtrix_bin(bin_name: str, try_conda_env: bool = True) -> str:
    """Find the path of the MRtrix binary.

    This function searches for the specified MRtrix binary in the system environment
    and returns its path. If the binary is not found, fall back to the path used in
    the shell.

    Parameters
    ----------
    bin_name : str
        The name of the MRtrix binary to search for.
    try_conda_env : bool, optional
        Flag indicating whether to search for the binary in the Conda environment,
        by default True.

    Returns
    -------
    str
        The path of the MRtrix binary.

    """
    bin_path = None
    conda_prefix = os.environ.get("CONDA_PREFIX", None)
    if try_conda_env and conda_prefix is not None:
        env_bin_path = (Path(conda_prefix) / "bin" / bin_name).resolve()
        if env_bin_path.exists():
            bin_path = str(env_bin_path)
    if bin_path is None:
        # Use `type -p` to get the location from the shell.
        proc = subprocess.Popen(
            f"type -p {bin_name}", stdout=subprocess.PIPE, shell=True, cwd=os.getcwd()
        )
        # Store and format the subprocess' output.
        proc_out = proc.communicate()[0].strip().decode("utf-8")
        if len(proc_out) > 0 and Path(proc_out).exists():
            bin_path = str(Path(proc_out).resolve())
    # Just let the subprocess runner figure it out.
    if bin_path is None:
        bin_path = bin_name

    return bin_path
