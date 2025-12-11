# -*- coding: utf-8 -*-
# Module for networks focused on general super-resolution (SR).
from functools import partial
from typing import Any, Callable, Literal

import einops
import einops.layers.torch
import torch

import mrinr

__all__ = ["ICNRUpsample", "ESPCNShuffle"]


# Based on implementations from fastai at <https://github.com/fastai/fastai> and
# torchlayers <https://github.com/szymonmaszke/torchlayers>.
def _icnr_init_(w: torch.Tensor, scale: int, init_fn_: Callable[[torch.Tensor], None]):
    """Initialize weight tensor according to ICNR nearest-neighbor scheme.

    Assumes w has shape 'chan_out x chan_in x spatial_dim_1 (x optional_spatial_dim_2 x ...)'
        and that 'chan_out' is divisible by `scale^N_spatial_dims`.

    Parameters
    ----------
    w : weight Tensor, usually from a convolutional layer.
    scale : int
    init_fn : Callable
    """
    n_spatial_dims = len(w.shape[2:])
    new_shape = (w.shape[0] // (scale**n_spatial_dims),) + tuple(w.shape[1:])
    set_sub_w = torch.zeros(*new_shape, dtype=w.dtype, device=w.device)
    init_fn_(set_sub_w)

    w_nn = einops.repeat(
        set_sub_w,
        "c_lr_out c_lr_in ... -> (c_lr_out repeat) c_lr_in ...",
        repeat=scale**n_spatial_dims,
    )
    w.copy_(w_nn)


@torch.no_grad()
def _conv_icnr_subkernel_init_(
    m: torch.nn.Module,
    scale: int,
    init_fn_=torch.nn.init.kaiming_normal_,
):
    if isinstance(
        m,
        (
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.LazyConv1d,
            torch.nn.LazyConv2d,
            torch.nn.LazyConv3d,
        ),
    ):
        _icnr_init_(m.weight.data, scale=scale, init_fn_=init_fn_)
        # The bias term is typically initialized to 0.
        if m.bias is not None:
            m.bias.data.zero_()


class ESPCNShuffle(einops.layers.torch.Rearrange):
    def __init__(
        self, spatial_dims: Literal[2, 3], out_channels: int, upscale_factor: int
    ):
        self._spatial_dims = spatial_dims
        self._out_channels = out_channels
        self._upscale_factor = upscale_factor

        if self._spatial_dims == 2:
            self._pattern = "b (c r1 r2) x y -> b c (x r1) (y r2)"
            self._rearrange_kwargs = {
                "c": self._out_channels,
                "r1": self._upscale_factor,
                "r2": self._upscale_factor,
            }
        elif self._spatial_dims == 3:
            self._pattern = "b (c r1 r2 r3) x y z -> b c (x r1) (y r2) (z r3)"
            self._rearrange_kwargs = {
                "c": self._out_channels,
                "r1": self._upscale_factor,
                "r2": self._upscale_factor,
                "r3": self._upscale_factor,
            }

        super().__init__(self._pattern, **self._rearrange_kwargs)

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def in_channels(self):
        return self.out_channels * (self._upscale_factor**self.spatial_dims)


class ICNRUpsample(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        blur: bool,
        kernel_size: int = 3,
        conv_padding_mode: Literal[
            "zeros", "reflect", "replicate", "circular"
        ] = "reflect",
        icnr_init: bool = True,
        conv_init_fn_: Callable[[torch.Tensor], None] = torch.nn.init.kaiming_normal_,
    ):
        """Upsampling layer with ICNR initialization and ESPCN shuffling.

        Based on work found in:

            A. Aitken, C. Ledig, L. Theis, J. Caballero, Z. Wang, and W. Shi,
            "Checkerboard artifact free sub-pixel convolution: A note on sub-pixel
            convolution, resize convolution and convolution resize," arXiv:1707.02937
            [cs], Jul. 2017, Accessed: Jan. 06, 2022. [Online].
            Available: http://arxiv.org/abs/1707.02937

            Y. Sugawara, S. Shiota, and H. Kiya, "Super-Resolution Using Convolutional
            Neural Networks Without Any Checkerboard Artifacts,” in 2018 25th
            IEEE International Conference on Image Processing (ICIP), Oct. 2018,
            pp. 66-70. doi: 10.1109/ICIP.2018.8451141.

        and on the `PixelShuffle_ICNR` implementation from fastai at
        https://github.com/fastai/fastai/blob/351f4b9314e2ea23684fb2e19235ee5c5ef8cbfd/fastai/layers.py

        Parameters
        ----------
        spatial_dims : Literal[2, 3]
            Number of spatial dimensions (2D or 3D).
        in_channels : int
        out_channels : int
        upscale_factor : int
        activate_fn : Callable
            Activation function used after the convolutional layer, but before shuffling.
        blur : bool
            Use a 2x2x2 avg pooling after the ESPCN shuffling.
        conv_init_fn_ : Callable, optional
            Function to initialize conv params, by default torch.nn.init.kaiming_normal_
        """

        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        if self.spatial_dims == 2:
            conv_cls = torch.nn.Conv2d
        elif self.spatial_dims == 3:
            conv_cls = torch.nn.Conv3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, but got {spatial_dims}")

        self.pre_conv = conv_cls(
            in_channels=self.in_channels,
            out_channels=self.out_channels * (self.upscale_factor**self.spatial_dims),
            kernel_size=kernel_size,
            padding="same",
            padding_mode=conv_padding_mode,
        )
        # Apply ICNR weight initialization.
        if icnr_init:
            self.pre_conv.apply(
                partial(
                    _conv_icnr_subkernel_init_,
                    scale=self.upscale_factor,
                    init_fn_=conv_init_fn_,
                )
            )

        self.shuffle = ESPCNShuffle(
            self.spatial_dims,
            out_channels=self.out_channels,
            upscale_factor=self.upscale_factor,
        )
        self.activate_fn = mrinr.nn.make_activate_fn_module(activate_fn)

        if blur:
            if self.spatial_dims == 2:
                # p = torch.nn.ReplicationPad2d((1, 0, 1, 0))
                p = torch.nn.ReplicationPad2d((0, 1, 0, 1))
                avg = torch.nn.AvgPool2d(kernel_size=2, stride=1)
            elif self.spatial_dims == 3:
                # p = torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0))
                p = torch.nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))
                avg = torch.nn.AvgPool3d(kernel_size=2, stride=1)
            self.post_blur = torch.nn.Sequential(p, avg)
        else:
            self.post_blur = None

    def forward(self, x):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.shuffle(y)
        if self.post_blur is not None:
            y = self.post_blur(y)

        return y
