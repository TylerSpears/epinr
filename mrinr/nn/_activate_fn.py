# -*- coding: utf-8 -*-
# from typing import Any
import pyrsistent
import torch

__all__ = [
    "Sine",
    "Gaussian",
    "ACTIVATE_FN_CLS_LOOKUP",
    "INPLACE_ACTIVATE_FNS",
]


# Sin activation function, mainly for SIREN-type INRs.
class Sine(torch.nn.Module):
    def __init__(self, a: float, inplace: bool = False):
        super().__init__()
        self._inplace = inplace
        self._a = a

    @property
    def a(self):
        return self._a

    def extra_repr(self) -> str:
        return f"a={self.a}"

    def forward(self, x):
        return torch.sin(self.a * x, out=x if self._inplace else None)


class Gaussian(torch.nn.Module):
    def __init__(self, sigma: float, inplace: bool = False):
        super().__init__()
        self._sigma = sigma
        self._inplace = inplace

    @property
    def sigma(self):
        return self._sigma

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}"

    def forward(self, x):
        # Can be checked against the "Beyond Periodicity" paper implementation:
        # <https://github.com/samgregoost/Beyond_periodicity/blob/c5506a3d906e2c3e3b1df1bde0c5029f687e7d84/run_nerf_helpers.py#L98>
        return torch.exp(
            (-0.5 * x**2) / (self.sigma**2), out=x if self._inplace else None
        )


ACTIVATE_FN_CLS_LOOKUP = pyrsistent.m(
    relu=torch.nn.ReLU,
    tanh=torch.nn.Tanh,
    sigmoid=torch.nn.Sigmoid,
    leakyrelu=torch.nn.LeakyReLU,
    elu=torch.nn.ELU,
    rrelu=torch.nn.RReLU,
    silu=torch.nn.SiLU,
    swish=torch.nn.SiLU,
    celu=torch.nn.CELU,
    logsigmoid=torch.nn.LogSigmoid,
    prelu=torch.nn.PReLU,
    relu6=torch.nn.ReLU6,
    selu=torch.nn.SELU,
    gelu=torch.nn.GELU,
    mish=torch.nn.Mish,
    softplus=torch.nn.Softplus,
    softshrink=torch.nn.Softshrink,
    softsign=torch.nn.Softsign,
    threshold=torch.nn.Threshold,
    glu=torch.nn.GLU,
    hardtanh=torch.nn.Hardtanh,
    sine=Sine,
    gaussian=Gaussian,
    none=torch.nn.Identity,
    identity=torch.nn.Identity,
)

INPLACE_ACTIVATE_FNS = {
    "elu",
    "hardsigmoid",
    "hardtanh",
    "hardswish",
    "leakyrelu",
    "relu",
    "relu6",
    "rrelu",
    "selu",
    "celu",
    "silu",
    "mish",
    "threshold",
    "sine",
    "gaussian",
}
