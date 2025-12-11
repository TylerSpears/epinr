# -*- coding: utf-8 -*-

import ast
import warnings
from typing import Any, Optional

import einops
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pyrsistent
import torch

from ..typing import NNModuleConstructT
from ..utils._utils import docstring
from ._activate_fn import ACTIVATE_FN_CLS_LOOKUP, INPLACE_ACTIVATE_FNS

__all__ = [
    "GroupNormRunningStats3d",
    "get_module_init_kwargs",
    "normalize_lookup_str",
    "make_norm_module",
    "make_activate_fn_module",
    "QuantizeRegularizationLayer",
    "coords_as_channels",
    "channels_as_coords",
    "grad_norm",
    "plot_grad_flow",
    "record_param_diffs_grads",
    "module_param_grad_dict",
    "overwrite_module_param_grads",
    "clone_module_named_params",
    "_parse_module_init_obj",
    "_NORM_CLS_LOOKUP",
]


@docstring(
    "\n".join(
        [
            "Original GroupNorm docstring:",
            "=" * max(map(lambda l: len(l), torch.nn.GroupNorm.__doc__.splitlines())),
            torch.nn.GroupNorm.__doc__,
            "=" * max(map(lambda l: len(l), torch.nn.GroupNorm.__doc__.splitlines())),
        ]
    )
)
class GroupNormRunningStats3d(torch.nn.InstanceNorm3d):
    __constants__ = torch.nn.InstanceNorm3d.__constants__ + [
        "num_groups",
        "num_channels",
    ]
    num_groups: int
    num_channels: int

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        if self.num_channels % self.num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_channels_per_group = num_channels // self.num_groups
        super().__init__(
            num_features=self.num_groups,
            eps=eps,
            momentum=momentum,
            affine=False,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        # If per-channel affines are requested, then instantiate them here. Otherwise,
        # just leave them as None as created in the _NormBase class.
        self.affine = affine
        if self.affine:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = torch.nn.Parameter(
                torch.empty(num_channels, **factory_kwargs)
            )
            self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return ", ".join(
            [
                "{num_groups}",
                "{num_channels}",
                "eps={eps}",
                "momentum={momentum}",
                "affine={affine}",
                "track_running_stats={track_running_stats}",
            ]
        ).format(**self.__dict__)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        feature_dim = input.dim() - self._get_no_batch_dim()
        # Feature size should be num_groups, but the affine size should be num_channels.
        if (
            input.size(feature_dim) != (self.num_groups * self.num_channels_per_group)
        ) and self.affine:
            raise ValueError(
                f"expected input's size at dim={feature_dim} to match"
                f" ({self.num_groups * self.num_channels_per_group}),"
                f" but got: {input.size(feature_dim)}."
            )

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)

    def _apply_instance_norm(self, input):
        # Input order is very particular, if the goal is to stay consistent with the
        # existing instance and group norm layers in pytorch. This order has been
        # validated against instance norm
        # (num_groups=N, num_channels=N, affine=True, track_running_stats=True)
        # and group norm
        # (num_groups=N, num_channels=M, affine=True, track_running_stats=False) layers
        # in pytorch.
        # Can also check another github user who recreated the group norm layer in a
        # more readable way (compared to pytorch):
        # <https://github.com/RoyHEyono/Pytorch-GroupNorm/blob/db7c29bf506ab768a11de620a101af0615405cc7/groupnorm.py>
        x = einops.rearrange(
            input,
            "b (g c_g) x y z -> b g c_g x y z",
            g=self.num_groups,
            c_g=self.num_channels_per_group,
        )
        y = torch.nn.functional.instance_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=None,
            bias=None,
            use_input_stats=self.training or not self.track_running_stats,
            momentum=self.momentum if self.momentum is not None else 0.0,
            eps=self.eps,
        )
        y = einops.rearrange(
            y,
            "b g c_g x y z -> b (g c_g) x y z",
            g=self.num_groups,
            c_g=self.num_channels_per_group,
        )
        # Apply affine transform here instead of within the instance_norm function.
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y


_NORM_CLS_LOOKUP = pyrsistent.m(
    batchnorm1d=torch.nn.BatchNorm1d,
    batchnorm2d=torch.nn.BatchNorm2d,
    batchnorm3d=torch.nn.BatchNorm3d,
    instancenorm1d=torch.nn.InstanceNorm1d,
    instancenorm2d=torch.nn.InstanceNorm2d,
    instancenorm3d=torch.nn.InstanceNorm3d,
    groupnorm=torch.nn.GroupNorm,
    groupnorm1d=torch.nn.GroupNorm,
    groupnorm2d=torch.nn.GroupNorm,
    groupnorm3d=torch.nn.GroupNorm,
    groupnormrunningstats3d=GroupNormRunningStats3d,
)


def get_module_init_kwargs(
    locals_dict: dict,
    extra_kwargs_dict: dict = dict(),
    ignore_locals_keys={"self", "__class__", "kwargs"},
    warn: bool = True,
) -> dict:
    init_kwargs = dict()
    for k, v in locals_dict.items():
        if k not in ignore_locals_keys:
            if torch.is_tensor(v):
                init_kwargs[k] = torch.clone(v).detach().cpu()
            else:
                init_kwargs[k] = v

    # Allow kwargs to catch extra init parameters, but warn if a non-None kwargs is
    # being sent over.
    kwargs_contains_non_empty_values = any(
        map(lambda v: (v is not None) and (v != dict()), extra_kwargs_dict.values())
    )
    if warn and kwargs_contains_non_empty_values:
        warnings.warn(f"Network init ignoring extra kwargs {extra_kwargs_dict}")

    return init_kwargs


def normalize_lookup_str(s: str) -> str:
    return str(s).strip().replace(" ", "").replace("_", "").replace(".", "").lower()


def _parse_module_init_obj(
    o: str | tuple[str, dict[str, Any]],
) -> tuple[str, dict | None]:
    if isinstance(o, str):
        s = o.strip().replace(" ", "").replace("\n", "")
        if "(" in s or ")" in s or "=" in s:
            if not ((s.index("(") < s.index(")")) and "(" in s and ")" in s):
                raise ValueError(f"Invalid init string {o}")
            init_lookup_str, kwargs_str = s.split("(", 1)
            kwargs_str = kwargs_str.rsplit(")", 1)[0]
            if kwargs_str == "":
                kwargs = None
            else:
                kwargs = dict()
                for kwarg_str in kwargs_str.split(","):
                    kw, v = kwarg_str.split("=")
                    kwargs[kw] = ast.literal_eval(v)
        else:
            init_lookup_str = s
            kwargs = None
    elif isinstance(o, (tuple, list)):
        o = tuple(o)
        if len(o) != 2:
            raise ValueError(f"Invalid init object {o}")
        init_lookup_str, kwargs = o
    else:
        raise ValueError(f"Invalid init object {o}")

    init_lookup_str = normalize_lookup_str(init_lookup_str)
    return init_lookup_str, kwargs


def make_norm_module(
    init_obj: NNModuleConstructT,
    num_channels_or_features: int,
    is_checkpointed: bool = False,
    allow_none: bool = False,
    **kwargs,
) -> torch.nn.Module | None:
    if init_obj is None:
        r = torch.nn.Identity()
    else:
        cls_str, kwargs_from_init_obj = _parse_module_init_obj(init_obj)
        try:
            norm_cls = _NORM_CLS_LOOKUP[cls_str]
        except KeyError:
            raise ValueError(f"Invalid norm layer {init_obj}")
        if kwargs_from_init_obj is None:
            kwargs_from_init_obj = dict()
        if "group" in cls_str:
            kwargs["num_channels"] = num_channels_or_features
        elif "instance" in cls_str or "batch" in cls_str:
            kwargs["num_features"] = num_channels_or_features
        r = norm_cls(**(kwargs_from_init_obj | kwargs))
        # If the norm layer falls under a checkpoint, then any norm that maintains a
        # running mean will update its running mean twice. So, the momentum term is
        # set to its square root.
        # See
        # <https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb>
        # and <https://discuss.pytorch.org/t/checkpoint-with-batchnorm-running-averages/17738>.
        if is_checkpointed:
            if hasattr(r, "momentum"):
                r.momentum = np.sqrt(r.momentum)

    if isinstance(r, torch.nn.Identity) and allow_none:
        r = None
    return r


def make_activate_fn_module(
    init_obj: NNModuleConstructT,
    try_inplace: bool = False,
    allow_none: bool = False,
    **kwargs,
) -> torch.nn.Module | None:
    kwargs = dict(**kwargs)
    if init_obj is None:
        r = torch.nn.Identity()
    else:
        cls_str, kwargs_from_init_obj = _parse_module_init_obj(init_obj)
        try:
            cls = ACTIVATE_FN_CLS_LOOKUP[cls_str]
        except KeyError:
            raise ValueError(f"Invalid activation function name {init_obj}")
        if kwargs_from_init_obj is None:
            kwargs_from_init_obj = dict()
        if (
            try_inplace or kwargs.get("try_inplace", False)
        ) and cls_str in INPLACE_ACTIVATE_FNS:
            kwargs = {"inplace": True} | kwargs
        kwargs = kwargs_from_init_obj | kwargs

        # If compile is in the kwargs, then compile the activation module and apply the
        # given kwargs to the compile function.
        if "compile" in kwargs.keys():
            compile_kwargs = dict(kwargs["compile"])
            kwargs = {
                k: kwargs[k]
                for k in list(
                    set(kwargs.keys())
                    - {
                        "compile",
                    }
                )
            }
            cls = torch.compile(cls, **compile_kwargs)
        r = cls(**kwargs)

    if isinstance(r, torch.nn.Identity) and allow_none:
        r = None
    return r


class QuantizeRegularizationLayer(torch.nn.Module):
    def __init__(self, bins: int):
        super().__init__()
        self._bins = float(bins)

    def extra_repr(self) -> str:
        return f"bins={int(self._bins)}"

    def forward(self, x):
        y = x + ((x * self._bins).round() / self._bins).detach() - x.detach()
        return y


def coords_as_channels(x: torch.Tensor, has_batch_dim: bool = True) -> torch.Tensor:
    spatial_dims = x.shape[-1]

    if spatial_dims == 2:
        if has_batch_dim:
            pattern = "b x y coord -> b coord x y"
        else:
            pattern = "x y coord -> coord x y"
    elif spatial_dims == 3:
        if has_batch_dim:
            pattern = "b x y z coord -> b coord x y z"
        else:
            pattern = "x y z coord -> coord x y z"
    else:
        raise ValueError(f"Invalid spatial_dims {spatial_dims}")

    return einops.rearrange(x, pattern, coord=spatial_dims)


def channels_as_coords(x: torch.Tensor, has_batch_dim: bool = True) -> torch.Tensor:
    if has_batch_dim:
        spatial_dims = x.shape[1]
    else:
        spatial_dims = x.shape[0]

    if spatial_dims == 2:
        if has_batch_dim:
            pattern = "b coord x y -> b x y coord"
        else:
            pattern = "coord x y -> x y coord"
    elif spatial_dims == 3:
        if has_batch_dim:
            pattern = "b coord x y z -> b x y z coord"
        else:
            pattern = "coord x y z -> x y z coord"
    else:
        raise ValueError(f"Invalid spatial_dims {spatial_dims}")

    return einops.rearrange(x, pattern, coord=spatial_dims)


def clone_module_named_params(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def module_param_grad_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    grad_dict = dict()
    for name, p in module.named_parameters():
        if p.requires_grad:
            if torch.is_tensor(p.grad):
                grad_dict[name] = p.grad.clone()
            else:
                grad_dict[name] = p.grad
    return grad_dict


def overwrite_module_param_grads(
    module: torch.nn.Module, grad_dict: dict[str, torch.Tensor]
) -> torch.nn.Module:
    for name, p in module.named_parameters():
        if p.requires_grad:
            if torch.is_tensor(p.grad) and torch.is_tensor(grad_dict[name]):
                p.grad.copy_(grad_dict[name], non_blocking=False)
            else:
                p.grad = grad_dict[name]
    return module


def grad_norm(model: torch.nn.Module, norm: float = 2.0) -> float:
    # From <https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/3>
    total_norm = 0
    params = list()
    for p in model.parameters():
        if p.grad is None:
            continue
        params.append(p.grad.detach().data.flatten())
        # total_norm += norm.item() ** 2
    # A negative norm is impossible, so a small negative value indicates that all
    # module grads were None.
    if len(params) == 0:
        total_norm = -0.1
    else:
        total_norm = torch.linalg.vector_norm(torch.cat(params), norm).item()
    # total_norm = math.sqrt(total_norm)
    return total_norm


@torch.no_grad()
def record_param_diffs_grads(
    epoch: int,
    step: int,
    model: torch.nn.Module,
    prev_model_named_params: dict[str, torch.nn.Parameter],
    norm: float = 2.0,
    exclude_bias_params: bool = True,
    param_name_transform_fn=lambda s: s,
    optim: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict[str, list[Any]]:
    """Record parameter grads and weight changes into a DataFrame-friendly structure.

    Parameters
    ----------
    epoch : int
        Current epoch of the training run.
    step : int
        Current step of the training run.
    model : torch.nn.Module
        Training model.
    prev_model_named_params : dict
        State dict of the model at the previous step.
    norm : float, optional
        Vector norm for calculating gradient and weight diff norms, by default 2.0
    exclude_bias_params : bool, optional
        Whether to exclude params with 'bias' in their name, by default True
    param_name_transform_fn : Callable[str, str], optional
        Function to transform raw parameter for readability, by default identity mapping
    optim : Optional[torch.optim.Optimizer], optional
        Optimizer object for 'model', by default None
    lr_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
        Learning rate scheduler for the model, by default None

    Returns
    -------
    dict[str, list[Any]]
        Dictionary with parameter information at this step.

        The dict has keys "epoch", "step", "param_name", "measure_type", and
        "val". If the optimizer and lr scheduler are provided, then the
        columns will also include "param_group" and "lr". Note that the gradient norm
        is not scaled by the learning rate.
    """
    if optim is not None and lr_scheduler is not None:
        param_groups = optim.param_groups
        lrs = lr_scheduler.get_last_lr()
        if isinstance(lrs, float):
            lrs = [lrs]
    else:
        param_groups = None
        lrs = None

    d = dict(
        epoch=list(),
        step=list(),
        param_name=list(),
        measure_type=list(),
        val=list(),
    )
    if param_groups is not None and lrs is not None:
        d["param_group"] = list()
        d["lr"] = list()

    for name, p in model.named_parameters():
        p_tm1 = prev_model_named_params[name]
        name = param_name_transform_fn(name)
        if exclude_bias_params and "bias" in name:
            continue
        if not p.requires_grad:
            continue

        if p.grad is None:
            grad_norm_val = -0.01
        else:
            grad_norm_val = (
                torch.linalg.vector_norm(p.grad.detach().data.flatten(), norm)
                .cpu()
                .item()
            )
        weights_diff_norm_val = (
            torch.linalg.vector_norm(
                (p.data.detach() - p_tm1.data.detach()).flatten(), norm
            )
            .cpu()
            .item()
        )

        for measure_type, val in zip(
            ["grad_norm", "weight_diff_norm"], [grad_norm_val, weights_diff_norm_val]
        ):
            d["epoch"].append(epoch)
            d["step"].append(step)
            d["param_name"].append(name)
            d["measure_type"].append(measure_type)
            d["val"].append(val)

            if "param_group" in set(d.keys()):
                for i_pg, pg in enumerate(param_groups):
                    if p in set(pg["params"]):
                        d["param_group"].append(i_pg)
                        d["lr"].append(lrs[i_pg])
                        break

    return d


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Taken from RoshanRane at
    <https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10>.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.detach().cpu().abs().mean())
            max_grads.append(p.grad.detach().cpu().abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            matplotlib.lines.Line2D([0], [0], color="c", lw=4),
            matplotlib.lines.Line2D([0], [0], color="b", lw=4),
            matplotlib.lines.Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )

    return plt.gcf()
