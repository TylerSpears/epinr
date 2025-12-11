# -*- coding: utf-8 -*-
import warnings
from typing import Any, Literal

import torch

import mrinr

__all__ = ["CARN"]


class _ResBlock(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        channels: int,
        kernel_size: int | tuple[int, ...],
        activate_fn: str | tuple[str, dict[str, Any]],
        combine_method: Literal["add", "mul", "cat"] = "add",
        **conv_kwargs,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = channels
        self.out_channels = channels
        self.channels = channels
        self._activate_fn_init_obj = activate_fn

        if self.spatial_dims == 2:
            conv_cls = torch.nn.Conv2d
        elif self.spatial_dims == 3:
            conv_cls = torch.nn.Conv3d
        all_conv_kwargs = (
            dict(padding="same")
            | conv_kwargs
            | dict(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=kernel_size,
            )
        )
        self.conv_block = torch.nn.Sequential(
            conv_cls(**all_conv_kwargs),
            mrinr.nn.make_activate_fn_module(init_obj=self._activate_fn_init_obj),
            conv_cls(**all_conv_kwargs),
        )

        self._combine_method = str(combine_method).lower().strip().replace("-", "")
        if "cat" in self._combine_method:
            self._combine_method = "cat"
            self.combiner = conv_cls(
                in_channels=self.channels * 2,
                out_channels=self.channels,
                kernel_size=1,
                padding="same",
                **conv_kwargs,
            )
        elif "add" in self._combine_method:
            self._combine_method = "add"
            self.combiner = None
        elif "mul" in self._combine_method:
            self._combine_method = "mul"
            self.combiner = None
        else:
            raise ValueError(f"Invalid combine method {combine_method}")

        self.activate_fn = mrinr.nn.make_activate_fn_module(
            init_obj=self._activate_fn_init_obj
        )

    def forward(self, x):
        y = self.conv_block(x)
        if self._combine_method == "cat":
            y = self.combiner(torch.cat([x, y], dim=1))
        elif self._combine_method == "add":
            y = x + y
        elif self._combine_method == "mul":
            y = x * y
        y = self.activate_fn(y)

        return y


class _DenseCascadeBlock(torch.nn.Module):
    def __init__(
        self,
        *base_layers,
        spatial_dims: Literal[2, 3],
        channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = channels
        self.out_channels = channels
        # If base_layers was not unpacked, then assume that a sequence of Modules was
        # passed instead.
        if len(base_layers) == 1 and not isinstance(base_layers[0], torch.nn.Module):
            base_layers = base_layers[0]

        self._activate_fn_init_obj = activate_fn
        # Assume the given base layers all have the same input size == output size.
        self.base_layers = torch.nn.ModuleList(base_layers)

        if self.spatial_dims == 2:
            lazy_conv_cls = torch.nn.LazyConv2d
        elif self.spatial_dims == 3:
            lazy_conv_cls = torch.nn.LazyConv3d
        combiner_convs = [
            torch.nn.Sequential(
                *[
                    lazy_conv_cls(
                        self.out_channels, kernel_size=1, stride=1, padding=0
                    ),
                    mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj),
                ]
            )
            for _ in self.base_layers
        ]
        self.combiner_convs = torch.nn.ModuleList(combiner_convs)

    def forward(self, x):
        # Unit (primary layer) input for layer l.
        x_unit_l = x
        # Combiner's input for the previous layer l-1.
        x_cascade_lm1 = x

        for base_layer, combiner in zip(self.base_layers, self.combiner_convs):
            y_unit_l = base_layer(x_unit_l)
            # Concatenate the previous cascaded input with the current base layer's
            # output to form the input for this block's combiner.
            x_cascade_l = torch.cat([x_cascade_lm1, y_unit_l], dim=1)
            y_l = combiner(x_cascade_l)

            # Set up next iteration.
            # Output of combiner at this step is the input to the next step.
            x_unit_l = y_l
            # The unit's output concatenated with the previous cascade inputs becomes
            # the new cascaded input.
            x_cascade_lm1 = x_cascade_l

        return y_l


class CARN(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        interior_channels: int,
        out_channels: int,
        n_dense_units: int,
        n_res_units: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        init_lazy_convs: bool = True,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self._spatial_dims = int(spatial_dims)
        self._in_channels = int(in_channels)
        self._interior_channels = int(interior_channels)
        self._out_channels = int(out_channels)

        self._n_dense_units = int(n_dense_units)
        self._n_res_units = int(n_res_units)
        self._activate_fn_init_obj = activate_fn

        if self.spatial_dims == 2:
            conv_cls = torch.nn.Conv2d
        elif self.spatial_dims == 3:
            conv_cls = torch.nn.Conv3d

        self.pre_conv = torch.nn.Sequential(
            conv_cls(
                self.in_channels,
                self._interior_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj),
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    _ResBlock(
                        spatial_dims=self.spatial_dims,
                        channels=self._interior_channels,
                        kernel_size=3,
                        activate_fn=self._activate_fn_init_obj,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                _DenseCascadeBlock(
                    *res_layers,
                    spatial_dims=self.spatial_dims,
                    channels=self._interior_channels,
                    activate_fn=self._activate_fn_init_obj,
                )
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = _DenseCascadeBlock(
            *top_level_units,
            spatial_dims=self.spatial_dims,
            channels=self._interior_channels,
            activate_fn=self._activate_fn_init_obj,
        )

        self.post_conv = conv_cls(
            self._interior_channels,
            self._out_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )

        if init_lazy_convs:
            if self.spatial_dims == 2:
                x = torch.randn(2, self._interior_channels, 40, 40)
            elif self.spatial_dims == 3:
                x = torch.randn(2, self._interior_channels, 40, 40, 40)
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    # Only cascade has lazy convs.
                    self.cascade(x)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: torch.Tensor):
        y = self.pre_conv(x)
        y = self.cascade(y)
        # Cascade ends with its activation function, so no activation is needed before
        # the post_conv.
        y = self.post_conv(y)

        return y
