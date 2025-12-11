# -*- coding: utf-8 -*-
import collections
import itertools
from typing import Any, Literal, Optional

import monai
import monai.inferers
import monai.networks
import monai.networks.blocks
import numpy as np
import torch
import torch.utils.checkpoint

import mrinr

__all__ = [
    "PlainCNN",
    "ResCNN",
    "ResBlock",
    "ResCNN",
    "ConvUnit",
    "include_bias_given_norm",
    "ConvNet",
    "MonaiAttentionBlock",
    "UNet",
]

_NORM_TYPES_NO_BIAS = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.LazyBatchNorm1d,
    torch.nn.LazyBatchNorm2d,
    torch.nn.LazyBatchNorm3d,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LazyInstanceNorm1d,
    torch.nn.LazyInstanceNorm2d,
    torch.nn.LazyInstanceNorm3d,
)


def _repeat_conv_params(
    n_repeats: int,
    channels: tuple[int, ...] | int,
    activate_fn: tuple[mrinr.typing.NNModuleConstructT, ...]
    | mrinr.typing.NNModuleConstructT,
    kernel_size: tuple[int, ...] | int,
    norm: tuple[mrinr.typing.NNModuleConstructT, ...] | mrinr.typing.NNModuleConstructT,
    end_conv_only: bool,
) -> dict:
    if isinstance(channels, int):
        channels = [channels] * n_repeats
    c = list(channels)

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * n_repeats
    k = list(kernel_size)

    # Check if norm is a Module constructor object/string, or a list of such objects.
    if (
        isinstance(norm, str)
        or (
            isinstance(norm, (list, tuple))
            and len(norm) == 2
            and isinstance(norm[1], dict)
        )
        or (norm is None)
    ):
        norm = [norm] * n_repeats
    n = list(norm)

    if isinstance(activate_fn, str) or (
        isinstance(activate_fn, (list, tuple))
        and len(activate_fn) == 2
        and isinstance(activate_fn[1], dict)
    ):
        activate_fn = [activate_fn] * n_repeats
    a = list(activate_fn)

    if end_conv_only:
        n[-1] = None
        a[-1] = None

    assert len(c) == len(k) == n_repeats, (
        "Invalid length(s) of conv channels or kernel sizes. Expected "
        + f"'{n_repeats}', got '{len(c)}' "
        + f"and '{len(k)}'."
    )
    assert len(n) == len(a) == n_repeats, (
        "Invalid length(s) of norms or activation functions. Expected "
        + f"'{n_repeats}', got '{len(n)}' "
        + f"and '{len(a)}'."
    )

    return {"channels": c, "kernel_size": k, "norm": n, "activate_fn": a}


def include_bias_given_norm(
    norm: Optional[str | tuple[str, dict[str, Any]] | torch.nn.Module],
) -> bool:
    if norm is None:
        r = True
    else:
        if isinstance(norm, torch.nn.Module):
            n_repr = repr(norm)
            if "affine=True" in n_repr:
                r = False
            else:
                r = not isinstance(norm, _NORM_TYPES_NO_BIAS)
        else:
            cls_str, kwargs_from_init_obj = mrinr.nn._parse_module_init_obj(norm)
            try:
                norm_cls = mrinr.nn._NORM_CLS_LOOKUP[cls_str]
            except KeyError:
                raise ValueError(f"Invalid norm layer {norm}")

            if kwargs_from_init_obj.get("affine", False):
                r = False
            else:
                r = norm_cls not in _NORM_TYPES_NO_BIAS
    return r


class ConvUnit(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        activate_fn: mrinr.typing.NNModuleConstructT,
        stride: int | tuple[int, ...] = 1,
        norm: mrinr.typing.NNModuleConstructT = None,
        padding: Optional[int | tuple[int] | str] = "same",
        padding_mode: Optional[str] = "reflect",
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        determine_bias_from_norm: bool = True,
        conv_only: bool = False,
        try_inplace_activate_fn: bool = False,
        is_checkpointed: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims

        # Canonicalize the padding mode. Pytorch has inconsistent naming for some
        # padding modes, so if we get an 'interpolation padding mode' we can convert it
        # to work with conv classes.
        if padding_mode is not None:
            padding_mode = str(padding_mode).strip().lower()
            if "zero" in padding_mode:
                padding_mode = "zeros"
            elif "reflect" in padding_mode:
                padding_mode = "reflect"
            elif (
                ("replicat" in padding_mode)
                or ("border" in padding_mode)
                or ("edge" in padding_mode)
            ):
                padding_mode = "replicate"
            elif (
                ("circ" in padding_mode)
                or ("wrap" in padding_mode)
                or (padding_mode == "dft")
            ):
                padding_mode = "circular"

        if self.spatial_dims == 2:
            Conv = torch.nn.Conv2d
        elif self.spatial_dims == 3:
            Conv = torch.nn.Conv3d
        else:
            raise ValueError(f"Invalid spatial_dims {self.spatial_dims}.")

        # Determine whether to have the bias term in the convolution.
        if determine_bias_from_norm:
            b_from_n = include_bias_given_norm(norm)
            bias = bias and b_from_n  # and (not conv_only)

        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if not conv_only:
            self.norm = mrinr.nn.make_norm_module(
                norm,
                num_channels_or_features=out_channels,
                is_checkpointed=is_checkpointed,
                allow_none=True,
            )
            self.activate_fn = mrinr.nn.make_activate_fn_module(
                activate_fn,
                try_inplace=try_inplace_activate_fn,
                allow_none=True,
            )
        else:
            self.norm = None
            self.activate_fn = None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.norm is not None:
            y = self.norm(y)
        if self.activate_fn is not None:
            y = self.activate_fn(y)
        return y


class ResBlock(torch.nn.Module):
    _is_coord_aware = False

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        channels: int,
        subunits: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        kernel_size: int = 1,
        padding_mode: str = "reflect",
        bias: bool = True,
        determine_bias_from_norm: bool = True,
        norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        last_subunit_conv_only: bool = False,
        end_activate_norm: bool = True,
        is_checkpointed: bool = False,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self.spatial_dims = spatial_dims
        self._channels = channels
        self.in_channels = self._channels
        self.out_channels = self._channels

        convs = list()
        for i in range(subunits):
            convs.append(
                ConvUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=self._channels,
                    out_channels=self._channels,
                    kernel_size=kernel_size,
                    activate_fn=activate_fn,
                    norm=norm,
                    padding="same",
                    padding_mode=padding_mode,
                    bias=bias,
                    determine_bias_from_norm=determine_bias_from_norm,
                    conv_only=(i == subunits - 1) and last_subunit_conv_only,
                    try_inplace_activate_fn=True,
                    is_checkpointed=is_checkpointed,
                )
            )
        self.convs = torch.nn.Sequential(*convs)

        if end_activate_norm:
            if norm is not None:
                self.norm = mrinr.nn.make_norm_module(
                    norm,
                    num_channels_or_features=self._channels,
                    is_checkpointed=is_checkpointed,
                )
            else:
                self.norm = torch.nn.Identity()
            self.activate_fn = mrinr.nn.make_activate_fn_module(
                activate_fn, try_inplace=True
            )
        else:
            self.norm = torch.nn.Identity()
            self.activate_fn = torch.nn.Identity()

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convs(x)
        y = y + x
        y = self.norm(y)
        y = self.activate_fn(y)
        return y


class PlainCNN(torch.nn.Module):
    _is_coord_aware = False

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        interior_channels: list[int] | int,
        kernel_sizes: list[int] | int = 1,
        padding_mode: str = "reflect",
        norms: list[None | str | tuple[str, dict[str, Any]]] | None = None,
        end_with_conv: bool = True,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self.spatial_dims = spatial_dims
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activate_fn_init_obj = activate_fn

        if isinstance(interior_channels, (list, tuple)):
            n_layers = len(interior_channels) + 1
        elif isinstance(kernel_sizes, (list, tuple)):
            n_layers = len(kernel_sizes)
        elif isinstance(norms, (list, tuple)):
            if end_with_conv:
                n_layers = len(norms) + 1
            else:
                n_layers = len(norms)
        else:
            raise ValueError(
                "Invalid input: interior_channels, kernel_sizes, or norms must be a list."
            )

        self._interior_channels = (
            [interior_channels] * (n_layers - 1)
            if isinstance(interior_channels, int)
            else interior_channels
        )
        self._kernel_sizes = (
            [kernel_sizes] * n_layers if isinstance(kernel_sizes, int) else kernel_sizes
        )
        if (
            norms is None
            or isinstance(norms, str)
            or (
                isinstance(norms, (list, tuple))
                and len(norms) == 2
                and isinstance(norms[1], dict)
            )
        ):
            self._norm_init_objs = [norms] * n_layers
        else:
            if end_with_conv:
                self._norm_init_objs = norms + [None]
            else:
                self._norm_init_objs = norms

        assert len(self._kernel_sizes) == len(self._norm_init_objs) == n_layers, (
            "Invalid length(s) of kernels or norms. Expected "
            + f"'{n_layers}', got '{len(self._kernel_sizes)}' "
            + f"and '{len(self._norm_init_objs)}'."
        )
        assert (
            len(self._interior_channels) == n_layers - 1
        ), "Invalid length of interior channels."

        # if self._spatial_dims == 2:
        #     conv_cls = torch.nn.Conv2d
        # elif self._spatial_dims == 3:
        #     conv_cls = torch.nn.Conv3d

        conv_blocks = list()
        for i, (c_in, c_out), kernel_size_i, norm_init_i in zip(
            range(n_layers),
            itertools.pairwise(
                [self._in_channels]
                + list(self._interior_channels)
                + [self._out_channels]
            ),
            self._kernel_sizes,
            self._norm_init_objs,
            strict=True,
        ):
            is_last_layer = i == n_layers - 1
            conv_only = is_last_layer and end_with_conv
            conv_blocks.append(
                ConvUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size_i,
                    activate_fn=self._activate_fn_init_obj,
                    norm=norm_init_i,
                    padding="same",
                    padding_mode=padding_mode,
                    conv_only=conv_only,
                )
            )
            # block_i.append(
            #     conv_cls(
            #         in_channels=c_in,
            #         out_channels=c_out,
            #         kernel_size=kernel_size_i,
            #         padding="same",
            #         padding_mode=padding_mode,
            #     )
            # )
            # if norm_init_i is not None:
            #     block_i.append(
            #         mrinr.nn.make_norm_module(
            #             norm_init_i, num_channels_or_features=c_out
            #         )
            #     )
            # block_i.append(mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj))
            # conv_blocks.append(torch.nn.Sequential(*block_i))

        # # Add the final layer.
        # conv_blocks.append(
        #     conv_cls(
        #         in_channels=self._interior_channels[-1],
        #         out_channels=self._out_channels,
        #         kernel_size=self._kernel_sizes[-1],
        #         padding="same",
        #         padding_mode=padding_mode,
        #     )
        # )

        self.conv_blocks = torch.nn.Sequential(*conv_blocks)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(x)


class ResCNN(torch.nn.Module):
    _is_coord_aware = False

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        block_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        n_blocks: int,
        subunits_per_block: int,
        kernel_size: int = 1,
        padding_mode: str = "reflect",
        norm: None | str | tuple[str, dict[str, Any]] | None = None,
        bias: bool = True,
        determine_bias_from_norm: bool = True,
        end_with_conv: bool = True,
        is_checkpointed: bool = False,
        _chunk_z_infer_chunks: Optional[int] = None,
        _chunk_z_infer_context_edge_size: Optional[int] = None,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self.spatial_dims = spatial_dims
        self._block_channels = block_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._activate_fn_init_obj = activate_fn
        n_units = n_blocks + 2

        self._is_checkpointed = is_checkpointed

        conv_blocks = list()
        for i in range(n_units):
            if i == 0:
                conv_blocks.append(
                    ConvUnit(
                        spatial_dims=self.spatial_dims,
                        in_channels=self.in_channels,
                        out_channels=self._block_channels,
                        kernel_size=kernel_size,
                        activate_fn=self._activate_fn_init_obj,
                        padding="same",
                        bias=bias,
                        determine_bias_from_norm=determine_bias_from_norm,
                        norm=norm,
                        padding_mode=padding_mode,
                        conv_only=False,
                        is_checkpointed=self._is_checkpointed,
                    )
                )
            elif 0 < i < n_units - 1:
                conv_blocks.append(
                    ResBlock(
                        spatial_dims=self.spatial_dims,
                        channels=self._block_channels,
                        subunits=subunits_per_block,
                        kernel_size=kernel_size,
                        activate_fn=self._activate_fn_init_obj,
                        norm=norm,
                        padding_mode=padding_mode,
                        bias=bias,
                        determine_bias_from_norm=determine_bias_from_norm,
                        last_subunit_conv_only=False,
                        end_activate_norm=True,
                        is_checkpointed=self._is_checkpointed,
                    )
                )
            else:
                is_last_layer = i == n_units - 1
                conv_only = is_last_layer and end_with_conv
                conv_blocks.append(
                    ConvUnit(
                        spatial_dims=self.spatial_dims,
                        in_channels=self._block_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel_size,
                        activate_fn=self._activate_fn_init_obj,
                        padding="same",
                        norm=norm,
                        padding_mode=padding_mode,
                        bias=bias,
                        determine_bias_from_norm=determine_bias_from_norm,
                        conv_only=conv_only,
                        is_checkpointed=self._is_checkpointed,
                    )
                )

        self.conv_blocks = torch.nn.Sequential(*conv_blocks)

        #!TMP
        # self._chunk_z_infer_chunks = 5
        # chunk_edge_size = sum(
        #     list(
        #         map(
        #             lambda k_p: ((k_p[1].shape[-1] - 1) // 2)
        #             if "conv.weight" in k_p[0]
        #             else 0,
        #             dict(self.conv_blocks.named_parameters()).items(),
        #         )
        #     )
        # )
        # chunk_edge_size = max(chunk_edge_size, 1)
        # if chunk_edge_size > 1:
        #     chunk_edge_size += 2
        # self._chunk_z_infer_context_edge_size = chunk_edge_size
        #!
        self._chunk_z_infer_chunks = _chunk_z_infer_chunks
        self._chunk_z_infer_context_edge_size = _chunk_z_infer_context_edge_size
        #!

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_checkpointed:
            r = torch.utils.checkpoint.checkpoint(
                self.conv_blocks,
                x,
                use_reentrant=False,
                determinism_check="none",
            )
        # Chunk inference, if allowed.
        elif (
            not self.training
            and self._chunk_z_infer_chunks is not None
            and self._chunk_z_infer_context_edge_size is not None
            and self.spatial_dims == 3
        ):
            r = self._forward_z_chunked(
                x,
                window_edge_len=self._chunk_z_infer_context_edge_size,
                max_chunks=self._chunk_z_infer_chunks,
            )
        else:
            r = self.conv_blocks(x)

        return r

    def _forward_z_chunked(
        self, x: torch.Tensor, window_edge_len: int, max_chunks: int
    ) -> torch.Tensor:
        assert self.spatial_dims == 3
        assert x.ndim == 5
        spatial_shape = tuple(x.shape[-3:])
        dim_size = spatial_shape[-1]
        # Set max chunks to be at most half the size of the last spatial dimension,
        # minus 1.
        max_chunks = min(max_chunks, (dim_size // 2) - 1)
        max_chunks = max(max_chunks, 1)

        # Get the indices for chunking the last spatial dimension.
        # Approximate the size of each chunk with the included context buffer.
        if dim_size % max_chunks == 0:
            chunk_end_idx = np.cumsum(np.array([dim_size // max_chunks] * max_chunks))
        else:
            chunk_end_idx = np.cumsum(
                np.array(
                    [dim_size // max_chunks] * (max_chunks - 1)
                    + [(dim_size // max_chunks) + (dim_size % max_chunks)]
                )
            )
        chunk_start_idx = np.cumsum(
            np.array([0] + [dim_size // max_chunks] * (max_chunks - 1))
        )
        # Extend the end indices by the context length.
        buffered_chunk_end_idx = chunk_end_idx + window_edge_len
        # Reduce the start indices by the context length.
        buffered_chunk_start_idx = chunk_start_idx - window_edge_len

        y = list()
        for in_start_idx, in_end_idx in zip(
            buffered_chunk_start_idx,
            buffered_chunk_end_idx,
        ):
            # Ensure both indices are in bounds.
            x_chunk = x[..., max(in_start_idx, 0) : min(in_end_idx, dim_size)]
            y_chunk = self.conv_blocks(x_chunk)
            # Determine indices to subsample the output such that it can be composed
            # into the correct output.
            # "window edge length, minus the amount of the input that is out of bounds"
            out_start_idx = window_edge_len - abs(in_start_idx - max(in_start_idx, 0))
            # "window edge length from the end, minus the amount of the input that is
            # out of bounds"
            out_end_idx = -window_edge_len + (in_end_idx - min(in_end_idx, dim_size))
            out_end_idx = None if abs(out_end_idx) == 0 else out_end_idx
            y.append(y_chunk[..., out_start_idx:out_end_idx])
            # Debug chunk sizes.
            # print(
            #     in_start_idx,
            #     in_end_idx,
            #     f"({in_end_idx - in_start_idx} | {x_chunk.shape[-1]})",
            #     "-->",
            #     out_start_idx,
            #     out_end_idx,
            #     f"({y[-1].shape[-1]})",
            # )
            # print("------")
        y = torch.cat(y, dim=-1)
        return y


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        inter_channels: list[int],
        out_channels: int,
        activate_fn: mrinr.typing.NNModuleConstructT,
        kernel_sizes: int | list[int],
        padding_mode: str = "reflect",
        padding: str | int | tuple[int, ...] = "same",
        norms: mrinr.typing.NNModuleConstructT
        | list[mrinr.typing.NNModuleConstructT] = None,
        end_conv_only: bool = True,
        bias: bool = True,
        determine_bias_from_norm: bool = True,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self.spatial_dims = spatial_dims
        channels = list(inter_channels) + [out_channels]
        p = _repeat_conv_params(
            len(channels),
            channels=channels,
            activate_fn=activate_fn,
            kernel_size=kernel_sizes,
            norm=norms,
            end_conv_only=end_conv_only,
        )

        conv_layers: list[ConvUnit] = list()
        for (c_in, c_out), kernel_size_i, norm_init_i, activate_fn_i in zip(
            itertools.pairwise([in_channels] + p["channels"]),
            p["kernel_size"],
            p["norm"],
            p["activate_fn"],
        ):
            conv_layers.append(
                ConvUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size_i,
                    activate_fn=activate_fn_i,
                    norm=norm_init_i,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias,
                    determine_bias_from_norm=determine_bias_from_norm,
                )
            )

        self.conv_layers = torch.nn.Sequential(*conv_layers)

    @property
    def in_channels(self):
        return self.conv_layers[0].conv.in_channels

    @property
    def out_channels(self):
        return self.conv_layers[-1].conv.out_channels

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_layers(x)
        return y


class UNet(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        encoder_channels: list[int],
        out_channels: int,
        encoder_downscale_factors: list[int],
        encoder_n_convs_per_level: int | list[int],
        downsampler: Literal["stride", "nearest"],
        upsampler: Literal["nearest", "shuffle"],
        activate_fn: mrinr.typing.NNModuleConstructT,
        kernel_size: int,
        encoder_norms_per_level: mrinr.typing.NNModuleConstructT
        | list[mrinr.typing.NNModuleConstructT] = None,
        downsampler_kwargs: dict[str, Any] = dict(),
        upsampler_kwargs: dict[str, Any] = dict(),
        padding_mode: str = "reflect",
        end_conv_only: bool = True,
        bias: bool = True,
        determine_bias_from_norm: bool = True,
        **kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        n_levels = len(encoder_channels)
        # Repeat params for each level, if not given as a list in the input args.
        scale_factors, n_convs, norms = self._repeat_params_per_level(
            n_levels,
            encoder_downscale_factors=encoder_downscale_factors,
            encoder_n_convs_per_level=encoder_n_convs_per_level,
            encoder_norms_per_level=encoder_norms_per_level,
        )
        # encoder_layers = list(itertools.repeat(None, n_levels))
        # decoder_layers = list(itertools.repeat(None, n_levels))
        encoder_layers = collections.OrderedDict()
        decoder_layers = collections.OrderedDict()

        encoder_in__conv_channels = itertools.pairwise(
            [in_channels] + list(encoder_channels)
        )
        # Build the unet one level at a time
        common_conv_kwargs = {
            "spatial_dims": self.spatial_dims,
            "activate_fn": activate_fn,
            "padding_mode": padding_mode,
            "padding": "same",
            "bias": bias,
            "determine_bias_from_norm": determine_bias_from_norm,
        }
        for level, (enc_c_in, c_conv), downscale_factor_i, n_convs_i, norms_i in zip(
            range(n_levels),
            encoder_in__conv_channels,
            scale_factors,
            n_convs,
            norms,
            strict=True,
        ):
            is_first_level = level == 0
            is_last_level = level == n_levels - 1
            # First level with no resampling.
            if is_first_level:
                # Encoder level
                encoder_layers[f"encode_{level}"] = ConvNet(
                    # encoder_layers[level] = ConvNet(
                    in_channels=enc_c_in,
                    inter_channels=[c_conv] * n_convs_i,
                    out_channels=c_conv,
                    kernel_sizes=kernel_size,
                    norms=norms_i,
                    end_conv_only=False,
                    **common_conv_kwargs,
                )
                # Decoder level with an additional 1-conv at the end.
                dec_in_channels = 2 * encoder_layers[f"encode_{level}"].out_channels
                # dec_in_channels = 2 * encoder_layers[level].out_channels
                dec_inter_channels = [c_conv] * n_convs_i
                dec_out_channels = out_channels
                dec_kernels = ([kernel_size] * n_convs_i) + [1]
                decoder_layers[f"decode_{level}"] = ConvNet(
                    # decoder_layers[level] = ConvNet(
                    in_channels=dec_in_channels,
                    inter_channels=dec_inter_channels,
                    out_channels=dec_out_channels,
                    kernel_sizes=dec_kernels,
                    norms=norms_i,
                    end_conv_only=end_conv_only,
                    **common_conv_kwargs,
                )
            # Intermediate levels with resampling.
            else:
                # Encoder
                enc_downsampler = self._build_downsampler(
                    downsampler=downsampler,
                    in_channels=enc_c_in,
                    out_channels=c_conv,
                    scale_factor=downscale_factor_i,
                    activate_fn=activate_fn,
                    norm=norms_i,
                    bias=bias,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding="same",
                    determine_bias_from_norm=determine_bias_from_norm,
                    downsampler_kwargs=downsampler_kwargs,
                )
                enc_level_conv = ConvNet(
                    in_channels=c_conv,
                    inter_channels=[c_conv] * (n_convs_i - 1),
                    out_channels=c_conv,
                    kernel_sizes=kernel_size,
                    norms=norms_i,
                    end_conv_only=False,
                    **common_conv_kwargs,
                )
                encoder_layers[f"encode_{level}"] = torch.nn.Sequential(
                    # encoder_layers[level] = torch.nn.Sequential(
                    enc_downsampler,
                    enc_level_conv,
                )

                # Decoder
                if not is_last_level:
                    dec_in_channels = 2 * c_conv
                else:
                    dec_in_channels = c_conv
                dec_level_conv = ConvNet(
                    in_channels=dec_in_channels,
                    inter_channels=[c_conv] * (n_convs_i - 1),
                    out_channels=c_conv,
                    kernel_sizes=kernel_size,
                    norms=norms_i,
                    end_conv_only=False,
                    **common_conv_kwargs,
                )
                dec_upsampler = self._build_upsampler(
                    upsampler=upsampler,
                    in_channels=dec_level_conv.out_channels,
                    out_channels=enc_c_in,
                    scale_factor=downscale_factor_i,
                    activate_fn=activate_fn,
                    norm=norms_i,
                    bias=bias,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding="same",
                    determine_bias_from_norm=determine_bias_from_norm,
                    upsampler_kwargs=upsampler_kwargs,
                )
                decoder_layers[f"decode_{level}"] = torch.nn.Sequential(
                    # decoder_layers[level] = torch.nn.Sequential(
                    dec_level_conv,
                    dec_upsampler,
                )

        # self.encoder = torch.nn.ModuleList(encoder_layers)
        # self.decoder = torch.nn.ModuleList(decoder_layers)
        self.encoder = torch.nn.ModuleDict(encoder_layers)
        # Reverse order of the decoder to go bottom-up. This is only for visual clarity
        # as each level is called independently of its order in the dict.
        self.decoder = torch.nn.ModuleDict(
            {k: decoder_layers[k] for k in reversed(decoder_layers.keys())}
        )

    def _build_downsampler(
        self,
        downsampler: str,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        activate_fn,
        norm,
        bias,
        kernel_size,
        padding_mode,
        padding,
        determine_bias_from_norm,
        downsampler_kwargs: dict,
    ):
        downsampler = str(downsampler).lower().strip().replace(" ", "")
        if "stride" in downsampler:
            d = mrinr.nn.StridedConvDownsample(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                stride_downscale_factor=scale_factor,
                conv_padding_as_affine_translate=True,
                **(
                    dict(
                        kernel_size=kernel_size,
                        bias=bias,
                        pad_mode=padding_mode,
                    )
                    | downsampler_kwargs
                ),
            )

            post_downsample = [
                mrinr.nn.make_norm_module(norm, num_channels_or_features=out_channels),
                mrinr.nn.make_activate_fn_module(activate_fn),
            ]
        elif "nearest" in downsampler:
            if self.spatial_dims == 2:
                raise NotImplementedError(
                    "Nearest downsampling not implemented for 2D."
                )
            if scale_factor == 1:
                d = None
            else:
                d = mrinr.nn.NNDownsample3D(
                    downscale_factor=scale_factor,
                    pad_mode=padding_mode,
                    **downsampler_kwargs,
                )
            post_downsample = [
                ConvUnit(
                    self.spatial_dims,
                    in_channels,
                    out_channels,
                    activate_fn=activate_fn,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding=padding,
                    norm=norm,
                    bias=bias,
                    determine_bias_from_norm=determine_bias_from_norm,
                    conv_only=False,
                )
            ]
        else:
            raise ValueError(
                f"Invalid downsampler {downsampler}. Must be 'stride' or 'nearest'."
            )
        n = list()
        for module in [d] + post_downsample:
            if module is not None:
                n.append(module)

        return torch.nn.Sequential(*n)

    def _build_upsampler(
        self,
        upsampler: str,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        activate_fn,
        norm,
        bias,
        kernel_size,
        padding_mode,
        padding,
        determine_bias_from_norm,
        upsampler_kwargs: dict,
    ):
        upsampler = str(upsampler).lower().strip().replace(" ", "")
        if "nearest" in upsampler:
            if scale_factor == 1:
                u = None
            else:
                u = mrinr.nn.NNUpsample(
                    spatial_dims=self.spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    upscale_factor=scale_factor,
                )
            post_upsample = [
                ConvUnit(
                    self.spatial_dims,
                    in_channels,
                    out_channels,
                    activate_fn=activate_fn,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding=padding,
                    norm=norm,
                    bias=bias,
                    determine_bias_from_norm=determine_bias_from_norm,
                    conv_only=False,
                )
            ]

        elif (
            ("shuffle" in upsampler) or ("icnr" in upsampler) or ("espcn" in upsampler)
        ):
            raise NotImplementedError("Upsampling with shuffle not implemented yet.")
            # post_upsample = [
            #     mrinr.nn.make_norm_module(norm, num_channels_or_features=out_channels),
            #     mrinr.nn.make_activate_fn_module(activate_fn),
            # ]

        else:
            raise ValueError(
                f"Invalid upsampler {upsampler}. Must be 'nearest' or 'shuffle'."
            )
        n = list()
        for module in [u] + post_upsample:
            if module is not None:
                n.append(module)

        return torch.nn.Sequential(*n)

    @staticmethod
    def _repeat_params_per_level(
        n_levels: int,
        encoder_downscale_factors: list[int],
        encoder_n_convs_per_level: int | list[int],
        encoder_norms_per_level: mrinr.typing.NNModuleConstructT
        | list[mrinr.typing.NNModuleConstructT] = None,
    ) -> tuple:
        assert (
            len(encoder_downscale_factors) == n_levels - 1
        ), "Invalid length of scale factors."
        if isinstance(encoder_n_convs_per_level, int):
            encoder_n_convs_per_level = [encoder_n_convs_per_level] * n_levels

        norms = encoder_norms_per_level
        if (
            isinstance(norms, str)
            or (
                isinstance(norms, (list, tuple))
                and len(norms) == 2
                and isinstance(norms[1], dict)
            )
            or (norms is None)
        ):
            norms = [norms] * n_levels
        # Append None to the scale factor in the first level.
        return (
            ([None] + list(encoder_downscale_factors)),
            encoder_n_convs_per_level,
            norms,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_outputs = list()
        for level in range(len(self.encoder)):
            x = self.encoder[f"encode_{level}"](x)
            # x = self.encoder[level](x)
            encoder_outputs.append(x)

        # Decoder
        for level in range(len(self.decoder) - 1, -1, -1):
            # reversed(range(len(self.encoder))):
            if level == len(self.decoder) - 1:
                x = self.decoder[f"decode_{level}"](x)
                # x = self.decoder[level](x)
            else:
                x = self.decoder[f"decode_{level}"](
                    torch.cat([x, encoder_outputs[level]], dim=1)
                )
                # x = self.decoder[level](torch.cat([x, encoder_outputs[level]], dim=1))

        return x


class MonaiAttentionBlock(monai.networks.blocks.spatialattention.SpatialAttentionBlock):
    """Perform spatial self-attention on the input tensor.

    The input tensor is reshaped to B x (x_dim * y_dim [ * z_dim]) x C, where C is the number of channels, and then
    self-attention is performed on the reshaped tensor. The output tensor is reshaped back to the original shape.

    Args:
        spatial_dims: number of spatial dimensions, could be 2 or 3.
        num_channels: number of input channels. Must be divisible by num_head_channels.
        norm: normalization layer.
        num_head_channels: number of channels per head.
        attention_dtype: cast attention operations to this dtype.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        norm: list[None | str | tuple[str, dict[str, Any]]] | None = None,
        num_head_channels: int | None = None,
        attention_dtype: Optional[torch.dtype] = None,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__(
            **self._init_kwargs,
            norm_num_groups=1,
            norm_eps=1e-6,
        )

        # Replace self.norm.
        if norm is not None:
            self.norm = mrinr.nn.make_norm_module(
                norm, num_channels_or_features=num_channels
            )
        else:
            self.norm = torch.nn.Identity()
