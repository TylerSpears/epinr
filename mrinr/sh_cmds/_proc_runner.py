# -*- coding: utf-8 -*-
import atexit
import collections
import datetime
import multiprocessing
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import docker
import nibabel as nib
import numpy as np
from box import Box
from more_itertools import collapse

DOCKER_CLIENT = None

docker_run_default_config = dict(
    # detach=True,
    ipc_mode="host",
    pid_mode="host",
    remove=True,
    auto_remove=False,
    # init=True,
    security_opt=["seccomp=unconfined"],
    cap_add=["SYS_PTRACE"],
    volumes={"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}},
    # stdin_open=True,
    # tty=True,
    # stderr=True,
    privileged=True,
    stop_signal="SIGINT",
    # stdout=True,
)


def _flatten(
    iterable: collections.abc.Iterable,
    parent_key=False,
    seperator: str = ".",
    as_dict: bool = False,
) -> collections.abc.Iterable:
    result = None
    if isinstance(iterable, collections.abc.MutableMapping):
        result = __flatten_dict(iterable, parent_key=parent_key, separator=seperator)
    elif isinstance(iterable, (list, tuple, set, frozenset, bytearray)):
        result = __flatten_dict(
            {"_": iterable}, parent_key=parent_key, separator=seperator
        )
        if not as_dict:
            result = tuple(result.values())
    else:
        result = iterable

    return result


def __flatten_dict(
    dictionary: collections.abc.MutableMapping,
    parent_key=False,
    separator: str = ".",
) -> collections.abc.MutableMapping:
    """Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary

    Taken from
    <https://github.com/ScriptSmith/socialreaper/blob/master/socialreaper/tools.py#L8>
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(_flatten(value, new_key, separator).items())
        elif isinstance(value, (list, tuple, set, frozenset, bytearray)):
            for k, v in enumerate(value):
                items.extend(_flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def rerun_indicator_from_mtime(
    input_files: Sequence[Union[str, Path]], output_files: Sequence[Union[str, Path]]
):
    in_files = [Path(str(f)) for f in collapse(input_files)]
    out_files = [Path(str(f)) for f in collapse(output_files)]
    in_mtimes = [p.stat().st_mtime_ns if p.exists() else -1 for p in in_files]
    out_mtimes = [p.stat().st_mtime_ns if p.exists() else -1 for p in out_files]

    result: bool
    if any(map(lambda t: t < 0, out_mtimes)):
        result = True
    elif max(in_mtimes) > min(out_mtimes):
        result = True
    else:
        result = False

    return result


def rerun_indicator_from_nibabel(
    input_im_data: np.ndarray, input_im_affine: np.ndarray, output_nifti: Path
) -> bool:
    rerun = True
    out = Path(str(output_nifti)).resolve()
    if not out.exists():
        rerun = True
    else:
        out_im = nib.load(str(out))
        # Compare affines first to possibly avoid loading the full file.
        out_im_affine = out_im.affine
        try:
            if not np.isclose(out_im_affine, input_im_affine).all():
                rerun = True
            else:
                out_im_data = out_im.get_fdata()
                data_is_similar = np.isclose(out_im_data, input_im_data).all()
                rerun = not data_is_similar
        except ValueError:
            rerun = True

    return rerun


def union_parent_dirs(*paths, resolve=True) -> Tuple[Path]:
    ps = [Path(str(p)) for p in _flatten(paths, as_dict=True).values()]
    if resolve:
        ps = [p.resolve() for p in ps]
    parent_ps = set()
    for p in ps:
        if p.is_dir():
            parent_ps.add(str(p))
        elif p.is_file():
            parent_ps.add(str(p.parent))
        elif not p.exists():
            # Check one level up from the non-existent Path, but only one level.
            if p.parent.exists():
                parent_ps.add(str(p.parent))
            else:
                raise RuntimeError(f"ERROR: Path {p} and {p.parent} do not exist.")

    return tuple(parent_ps)


def multiline_script2docker_cmd(script: str):
    # wrapped_script = "\n".join([r"""bash -s << "EOF" """, script, r"""EOF"""])
    # wrapped_script = ["bash", "-s", "<<", r'"EOF"\n', script, "\nEOF"]
    # wrapped_script = ["\n".join([r"""bash --login -s << "EOF" """, script, r"""EOF"""])]
    # Creating a list for the command should prevent other functions from
    # `shlex.split`-ing the multi-line script.
    wrapped_script = ["bash", "-c", script]
    return wrapped_script


def call_docker_run(
    img: str,
    cmd: Union[str, List[str]],
    env: dict = dict(),
    run_config: dict = dict(),
):
    global DOCKER_CLIENT
    if DOCKER_CLIENT is None:
        DOCKER_CLIENT = docker.from_env()

    client = DOCKER_CLIENT

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    # Merge user run config and the default config. Merginb Box objects ensures
    # sub-dicts are merged properly.
    default_config = Box(docker_run_default_config)
    user_config = Box(run_config)
    run_opts = default_config + user_config
    # Some options are lists, and lists should be appended.
    for k in {"security_opts", "cap_add", "mounts"}:
        if k in run_opts.keys():
            run_opts[k] = default_config.get(k, list()) + user_config.get(k, list())
    run_opts["environment"] = env
    run_opts = run_opts.to_dict()

    # Run container in detached mode.
    container = client.containers.run(img, cmd, detach=True, **run_opts)

    def _sigint_handler(sig, frame):
        print(f"Signal {sig} caught!")
        print(f"PID {os.getpid()}, PPID {os.getppid()}")
        # Stop container first
        container.stop(timeout=5)
        # Clean up any other child processes.
        # terminate all active children
        active_cprocs = multiprocessing.active_children()
        for child in active_cprocs:
            child.terminate()
        # block until all children have closed
        for child in active_cprocs:
            child.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    tail_logs = list()
    tail_maxsize = 30
    try:
        for line in container.logs(stream=True, stdout=True, stderr=True):
            print(
                f"Docker {datetime.datetime.now().replace(microsecond=0)} | ",
                line.decode().strip(),
            )
            tail_logs.append(line.decode().strip())
            if len(tail_logs) > tail_maxsize:
                tail_logs.pop(0)
        exit_status = container.wait()["StatusCode"]
        exc = None
    except Exception as e:
        exit_status = 1
        exc = e
    if exit_status > 0:
        raise docker.errors.ContainerError(
            container,
            exit_status,
            f"Exception {exc}",
            cmd,
            img,
            "\n".join(tail_logs),
        )

    return 0


def get_docker_mount_obj(path: Path, **mount_obj_kwargs):
    p = Path(path).resolve()
    m_obj = None
    if p.is_file():
        kw_defaults = dict(target=str(p), type="bind")
        mount_kw = {**kw_defaults, **mount_obj_kwargs}
        m_obj = docker.types.Mount(source=str(p), **mount_kw)
    elif p.is_dir():
        kw_defaults = dict(mode="rw")
        vol_kw = {**kw_defaults, **mount_obj_kwargs}
        m_obj = {"bind": str(p), **vol_kw}
    else:
        raise ValueError(f"ERROR: {path} is invalid path")

    return m_obj


def call_shell_exec(
    cmd: str,
    args: str,
    cwd: Path,
    env: dict = None,
    prefix: str = "",
    popen_args_override=None,
    stdout_identifier: str = "STDOUT",
    stderr_identifier: str = "STDOUT",
):
    env = os.environ if env is None else env

    # Taken from
    # <https://sharats.me/posts/the-ever-useful-and-neat-subprocess-module/#watching-both-stdout-and-stderr>
    io_queue = queue.Queue()

    def stream_watcher(identifier, stream):
        for line in stream:
            io_queue.put((identifier, line))
        if not stream.closed:
            stream.close()

    # Split the prefix up into tokens, with the cmd + args as the "string to run"
    # argument to the prefix.
    # Start the new process and assign it to a new process group. This allows for
    # killing child processes; see <https://stackoverflow.com/a/34027738/13225248>
    if popen_args_override is None:
        popen_args = shlex.split(prefix) + [cmd + " " + args]
    else:
        popen_args = popen_args_override
    proc = subprocess.Popen(
        popen_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        bufsize=10,
        text=True,
        preexec_fn=os.setpgrp,
    )

    out_lines = list()

    def printer():
        while True:
            try:
                # Block for a few seconds.
                item = io_queue.get(True, 2)
            except queue.Empty:
                # No output in either streams for a few seconds. Are we done?
                if proc.poll() is not None:
                    break
            else:
                identifier, line = item
                log_line = f"{identifier}: {line}"
                print(log_line.rstrip(), flush=True)
                out_lines.append(log_line)

    t_stdout = threading.Thread(
        target=stream_watcher,
        name="stdout-watcher",
        args=(stdout_identifier, proc.stdout),
    )
    t_stdout.start()

    t_stderr = threading.Thread(
        target=stream_watcher,
        name="stderr-watcher",
        args=(stderr_identifier, proc.stderr),
    )
    t_stderr.start()

    t_print = threading.Thread(target=printer, name="printer")
    t_print.start()

    # Make sure proc is killed when exiting.
    @atexit.register
    def kill_proc():
        print("Maybe kill proc? ", proc.pid, flush=True)
        if proc.poll() is None:
            p_group_id = os.getpgid(proc.pid)
            os.killpg(p_group_id, signal.SIGKILL)
            print(
                "Killed proc ", proc.pid, " and process group ", p_group_id, flush=True
            )
            proc.terminate()
            proc.kill()
            # os.kill(proc.pid, signal.SIGTERM)
            # os.kill(proc.pid, signal.SIGKILL)

    try:
        t_stdout.join()
        t_stderr.join()
        t_print.join()
    except KeyboardInterrupt as e:
        kill_proc()
        raise e
    else:
        return_code = proc.poll()
        # Proc should be done by now, if threads have exited.
        if return_code is None:
            kill_proc()

    # Proc absolutely has to be dead by now.
    atexit.unregister(kill_proc)
    if return_code > 0:
        raise subprocess.CalledProcessError(
            return_code, proc.args, "\n".join(out_lines)
        )

    return out_lines
