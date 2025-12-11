# -*- coding: utf-8 -*-
import dataclasses
import datetime
import functools
import hashlib
import inspect
import io
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Union

import dipy
import dipy.io
import dipy.io.streamline
import torch

from mrinr._lazy_loader import LazyLoader

dotenv = LazyLoader("dotenv", globals(), "dotenv")
jax = LazyLoader("jax", globals(), "jax")
lax = LazyLoader("lax", globals(), "jax.lax")
jnp = LazyLoader("jnp", globals(), "jax.numpy")

__all__ = [
    "docstring",
    "dict_tensor_pin_to",
    "create_subj_rng_seed",
    "dipy_save_trk2tck",
    "dotenv_vars_from_cwd",
    "fork_rng",
    "get_file_glob_unique",
    "get_gpu_specs",
    "gpu_mem_restore",
    "is_vol",
    "j2t",
    "make_dataclass_fields_from_class",
    "pretty_time",
    "t2j",
    "timestamp_now",
    "ensure_vol_channels",
    "undo_vol_channels",
    "ensure_image_channels",
    "undo_image_channels",
    "tee",
]


def dict_tensor_pin_to(tensor_dict: dict, device: torch.device, pin: bool) -> dict:
    d = dict()
    for k, v in tensor_dict.items():
        if torch.is_tensor(v):
            d[k] = v.pin_memory().to(device) if pin else v.to(device)
        else:
            d[k] = v
    return d


def get_file_glob_unique(root_path: Path, glob_pattern: str) -> Path:
    root_path = Path(root_path)
    glob_pattern = str(glob_pattern)
    files = list(root_path.glob(glob_pattern))

    if len(files) == 0:
        files = list(root_path.rglob(glob_pattern))

    if len(files) > 1:
        raise RuntimeError(
            f"ERROR: {len(files)} file matches for glob pattern "
            f"{glob_pattern} under directory {str(root_path)}. "
            "Expect only one match."
        )
    elif len(files) == 0:
        raise RuntimeError(
            "ERROR: No files match glob pattern "
            f"{glob_pattern} under directory {str(root_path)}; "
            "Expect one match."
        )

    return files[0]


def dotenv_vars_from_cwd():
    """Get .env and .envrc environment variables with direnv.

    This requires the python-dotenv package, and direnv be installed on the system
    This will not work on Windows.
    NOTE: This is kind of hacky, and not necessarily safe. Be careful...
    """

    # Form command to be run in direnv's context. This command will print out
    # all environment variables defined in the subprocess/sub-shell.
    # command = "/usr/bin/env direnv exec {} /usr/bin/env".format(os.getcwd())
    command = "/usr/bin/env direnv export bash"
    # Run command in a new subprocess.
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, cwd=os.getcwd()
    )
    # Store and format the subprocess' output.
    proc_out = proc.communicate()[0].strip().decode("utf-8")
    # `direnv export bash` gives an environment var diff as
    # `export ENV_VAR=$'var_value';export ENV_VAR2=$'var2_value';...`
    # So, modify the direnv diff to be parsable by dotenv.dotenv_values.
    proc_out = proc_out.replace(";export ", "\n").replace(r"$'", r"'")
    # Use python-dotenv to load the environment variables by using the output of
    # 'direnv exec ...' as a 'dummy' .env file.
    v = dotenv.dotenv_values(stream=io.StringIO(proc_out), verbose=True)
    return v


def timestamp_now(iso: bool = False) -> str:
    t = f"{datetime.datetime.now().replace(microsecond=0).isoformat()}"
    if not iso:
        t = t.replace(":", "_")
    return t


def pretty_time(t_start_sec: float, t_end_sec: float) -> str:
    d = t_end_sec - t_start_sec
    hours = d // 3600
    minutes = (d - (hours * 3600)) // 60
    seconds = round(d - (hours * 3600 + minutes * 60))
    return f"{hours} hours, {minutes} minutes, and {seconds} seconds"


def make_dataclass_fields_from_class(
    cls: type,
) -> tuple[tuple[str, type] | tuple[str, type, dataclasses.Field]]:
    fields = list()
    for arg in inspect.signature(cls).parameters.values():
        field_i = list()
        field_i.append(arg.name)
        if arg.annotation != inspect.Parameter.empty:
            field_i.append(arg.annotation)
        else:
            field_i.append(Any)
        if arg.default != inspect.Parameter.empty:
            # Use __class__.__hash__ as a proxy for mutable state; taken from the
            # python dataclasses source code.
            mutable = arg.default.__class__.__hash__ is None
            if mutable:
                field_i.append(dataclasses.field(default_factory=lambda: arg.default))
            else:
                field_i.append(dataclasses.field(default=arg.default))
        fields.append(tuple(field_i))
    return tuple(fields)


def fork_rng(rng: torch.Generator) -> torch.Generator:
    rng_fork = torch.Generator(device=rng.device)
    rng_fork.set_state(rng.get_state().clone())
    return rng_fork


def gpu_mem_restore(func):
    """Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted.

    From fast.ai <https://fastai1.fast.ai/troubleshoot.html#custom-solutions>.
    See also
    - <https://docs.fast.ai/dev/gpu.html#gpu-memory-notes>
    - <https://discuss.pytorch.org/t/a-guide-to-recovering-from-cuda-out-of-memory-and-other-exceptions/35628>
    - <https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781?u=ptrblck>

    Note that using this wrapper may break debugging, especially in jupyter notebooks
    or interactive sessions.

    Parameters
    ----------
    func : Callable
        Any function that may allocate large GPU objects
    """

    # From <https://fastai1.fast.ai/troubleshoot.html#custom-solutions>
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:  # noqa E722
            type, val, tb = sys.exc_info()
            traceback.clear_frames(tb)
            raise type(val).with_traceback(tb) from None

    return wrapper


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


def is_vol(x: torch.Tensor, ch=True, batch=False):
    if not torch.is_tensor(x):
        s = False
    else:
        if ch:
            if batch:
                s = x.ndim == 5
            else:
                s = x.ndim == 4
        else:
            if not batch:
                s = x.ndim == 3
            else:
                raise ValueError("ERROR: Must assume channel dim if assuming batch dim")
    return s


def ensure_image_channels(x: torch.Tensor, batch=False) -> torch.Tensor:
    if x.ndim == 2:
        y = x.unsqueeze(0)
    else:
        y = x
    if batch and y.ndim == 3:
        y = y.unsqueeze(0)

    if (batch and y.ndim != 4) or (not batch and y.ndim != 3):
        raise ValueError(
            "Expected Tensor to be shape "
            + ("[batch] x" if batch else "")
            + "[channel] x i x j, "
            + f"got shape {tuple(x.shape)}"
        )
    return y


def undo_image_channels(
    x_expanded: torch.Tensor,
    orig_x: torch.Tensor,
    strict: bool = False,
) -> torch.Tensor:
    if x_expanded.ndim != orig_x.ndim:
        # If the expanded tensor has both a batch and a channel dimension, and the batch
        # dimension is > 1, then the channel dimension should not be removed even if
        # the channel size is 1.
        if x_expanded.ndim == 4 and x_expanded.shape[0] > 1:
            y = x_expanded
        else:
            dims_to_squeeze = tuple(range(x_expanded.ndim - orig_x.ndim))
            # Squeeze all dims at once in the event that x_expanded broadcast a singleton
            # dimension to be > 1, but still allow for squeezing remaining singletons.
            y = x_expanded.squeeze(dim=dims_to_squeeze)
    else:
        y = x_expanded
    if strict and y.shape != orig_x.shape:
        raise ValueError(
            f"Could not reshape Tensor of shape {tuple(x_expanded.shape)} "
            + f"to original shape of {tuple(orig_x.shape)}."
        )
    return y


def ensure_vol_channels(x: torch.Tensor, batch: bool = False) -> torch.Tensor:
    if x.ndim == 3:
        y = x.unsqueeze(0)
    else:
        y = x
    if batch and y.ndim == 4:
        y = y.unsqueeze(0)

    if (batch and y.ndim != 5) or (not batch and y.ndim != 4):
        raise RuntimeError(
            "Expected Tensor to be shape "
            + ("[batch] x" if batch else "")
            + "[channel] x i x j x k, "
            + f"got shape {tuple(x.shape)}"
        )
    return y


def undo_vol_channels(
    x_expanded: torch.Tensor,
    orig_x: torch.Tensor,
    strict: bool = False,
) -> torch.Tensor:
    if x_expanded.ndim != orig_x.ndim:
        # If the expanded tensor has both a batch and a channel dimension, and the batch
        # dimension is > 1, then the channel dimension should not be removed even if
        # the channel size is 1.
        if x_expanded.ndim == 5 and x_expanded.shape[0] > 1:
            y = x_expanded
        else:
            dims_to_squeeze = tuple(range(x_expanded.ndim - orig_x.ndim))
            # Squeeze all dims at once in the event that x_expanded broadcast a singleton
            # dimension to be > 1, but still allow for squeezing remaining singletons.
            y = x_expanded.squeeze(dim=dims_to_squeeze)
    else:
        y = x_expanded
    if strict and y.shape != orig_x.shape:
        raise RuntimeError(
            f"Could not reshape Tensor of shape {tuple(x_expanded.shape)} "
            + f"to original shape of {tuple(orig_x.shape)}."
        )

    return y


def __ensure_vol_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        y = x.unsqueeze(0)
    elif x.ndim == 4:
        y = x
    else:
        raise ValueError(
            "ERROR: Tensor must be shape (i, j, k) or shape (channel, i, j, k), "
            + f"got shape {tuple(x.shape)}"
        )
    return y


def __undo_vol_channels(x_ch: torch.Tensor, orig_x: torch.Tensor) -> torch.Tensor:
    if orig_x.ndim == 3:
        y = x_ch[0]
    else:
        y = x_ch
    return y.to(x_ch)


def docstring(docstr, sep="\n"):
    """
    Decorator: Append to an object's, function or class, docstring.
    """

    # Taken from <https://stackoverflow.com/a/12218693>.
    def _decorator(obj):
        if obj.__doc__ is None:
            obj.__doc__ = docstr
        else:
            obj.__doc__ = sep.join([obj.__doc__, docstr])
        return obj

    return _decorator


def t2j(t_tensor: torch.Tensor) -> "jax.Array":
    t = t_tensor.contiguous()
    # Dlpack does not handle boolean arrays.
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
        to_bool = True
    else:
        to_bool = False
    if not jax.config.x64_enabled and t.dtype == torch.float64:
        # Unsafe casting, but it's necessary if jax can only handle 32-bit floats. In
        # some edge cases, like if any dimension size is 1, the conversion will error
        # out.
        t = t.to(torch.float32)

    # 1-dims cause all sorts of problems, so just remove them before conversion, then
    # add them back afterwards.
    if 1 in tuple(t.shape):
        orig_shape = tuple(t.shape)
        t = t.squeeze()
        to_expand = tuple(
            filter(lambda i_d: orig_shape[i_d] == 1, range(len(orig_shape)))
        )
    else:
        to_expand = None

    if t.device.type.casefold() == "cuda":
        target_dev_idx = t.device.index
        jax_dev = list(filter(lambda d: d.id == target_dev_idx, jax.devices()))[0]
    else:
        jax_dev = None

    j = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
    if jax_dev is not None:
        j = jax.device_put(j, jax_dev)
    j = j.astype(bool) if to_bool else j

    if to_expand is not None:
        j = lax.expand_dims(j, to_expand)

    return j


def j2t(j_tensor: "jax.Array") -> torch.Tensor:
    j = j_tensor.block_until_ready()
    if j.dtype == bool:
        j = j.astype(jnp.uint8)
        to_bool = True
    else:
        to_bool = False

    if list(j.devices())[0].platform.casefold() == "gpu":
        target_dev_idx = list(j.devices())[0].id
        torch_dev = f"cuda:{target_dev_idx}"
        if target_dev_idx > (torch.cuda.device_count() - 1):
            torch_dev = None
    else:
        torch_dev = None

    t = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(j))
    if torch_dev is not None:
        t = t.to(torch_dev)
    t = t.bool() if to_bool else t

    return t


def dipy_save_trk2tck(
    trk_f,
    target_tck_f,
    trk_reference="same",
    load_trk_kwargs=dict(),
    save_tck_kwargs=dict(),
):
    trk_f = str(Path(trk_f).resolve())
    trk = dipy.io.streamline.load_trk(trk_f, trk_reference, **load_trk_kwargs)
    trk.to_rasmm()
    tck_f = str(Path(target_tck_f).resolve())
    dipy.io.streamline.save_tck(trk, tck_f, **save_tck_kwargs)
    return trk


def get_gpu_specs():
    """Return string describing GPU specifications.

    Taken from
    <https://www.thepythoncode.com/article/get-hardware-system-information-python>.

    Returns
    -------
    str
        Human-readable string of specifications.
    """
    # These packages are not used elsewhere, so only import them in the function.
    import GPUtil
    import tabulate

    gpus = GPUtil.getGPUs()
    specs = list()
    specs.append("".join(["=" * 50, "GPU Specs", "=" * 50]))
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        driver_version = gpu.driver
        cuda_version = torch.version.cuda
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_uuid = gpu.uuid
        list_gpus.append(
            (
                gpu_id,
                gpu_name,
                driver_version,
                cuda_version,
                gpu_total_memory,
                gpu_uuid,
            )
        )

    table = tabulate.tabulate(
        list_gpus,
        headers=(
            "id",
            "Name",
            "Driver Version",
            "CUDA Version",
            "Total Memory",
            "uuid",
        ),
    )

    specs.append(table)

    return "\n".join(specs)


def tee(
    *args,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
    file: Optional[Path] = None,
) -> None:
    if file is not None:
        with open(file, "a") as f:
            f.write(sep.join(map(str, args)) + end)
            if flush:
                f.flush()
    print(*args, sep=sep, end=end, flush=flush)
