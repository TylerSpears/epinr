# -*- coding: utf-8 -*-
import itertools
from typing import Any, Literal, Optional

import einops
import numpy as np
import torch

import mrinr

from ._cont_coders import DenseCoordSpace

__all__ = [
    "CoordAwareIdentityModuleWrapper",
    "IdentityResampler",
    "InterpolationResampler",
    "TranslationOnlyResampler",
    "FConvEncoder",
    "FConvDecoder",
]


class CoordAwareIdentityModuleWrapper(torch.nn.Module):
    _is_coord_aware: bool = True

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self,
        x: torch.Tensor,
        *args,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        return_coord_space: bool = False,
        **kwargs,
    ):
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )

        y = self.module(x, *args, **kwargs)

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=x_coords, affine=affine_x_el2coords
            )
        else:
            r = y

        return r


class IdentityResampler(torch.nn.Module):
    _is_coord_aware: bool = True

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        *args,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        return_coord_space: bool = False,
        **kwargs,
    ):
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )

        y = x

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=x_coords, affine=affine_x_el2coords
            )
        else:
            r = y

        return r


class InterpolationResampler(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_features: int,
        out_features: int,
        mode_or_interpolation,
        padding_mode_or_bound,
        interp_lib: Literal["torch", "interpol"] = "torch",
        resample_kwargs: dict[str, Any] = dict(),
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=kwargs, warn=True
        )

        super().__init__()

        self._spatial_dims = spatial_dims
        self._in_features = in_features
        self._out_features = out_features
        self._in_channels = self._in_features
        self._out_channels = self._out_features
        self._interp_lib = interp_lib
        self._mode_or_interpolation = mode_or_interpolation
        self._padding_mode_or_bound = padding_mode_or_bound
        self._extra_resample_kwargs = resample_kwargs

        if self.in_channels != self.out_channels:
            # Set up conv layer to create a learned linear combination in
            # the channel dimension, while interp does the combination in the spatial
            # dimension.
            conv_cls = torch.nn.Conv2d if self.spatial_dims == 2 else torch.nn.Conv3d
            self.channel_conv = conv_cls(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
            )
        else:
            self.channel_conv = None

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    @property
    def in_channels(self):
        return self.in_features

    @property
    def out_channels(self):
        return self.out_features

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def extra_repr(self) -> str:
        return (
            "(interpolation_kwargs): ("
            + " ".join(
                [
                    f"spatial_dims={self.spatial_dims},",
                    f"in_channels={self.in_channels},",
                    f"out_channels={self.out_channels},",
                    f"interp_lib='{self._interp_lib}',",
                    f"mode_or_interpolation='{self._mode_or_interpolation}',",
                    f"padding_mode_or_bound='{self._padding_mode_or_bound}'",
                ]
            )
            + ")"
        )

    def forward(
        self,
        x: mrinr.typing.Volume | mrinr.typing.Image,
        x_coords: Any,  # Ignored, for compatibility with INR resamplers
        query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        affine_query_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        *args,
        return_coord_space: bool = False,
        **kwargs,
    ) -> mrinr.typing.Volume | mrinr.typing.Image | DenseCoordSpace:
        y = mrinr.grid_resample(
            x=x,
            affine_x_el2coords=affine_x_el2coords,
            sample_coords=query_coords,
            interp_lib=self._interp_lib,
            mode_or_interpolation=self._mode_or_interpolation,
            padding_mode_or_bound=self._padding_mode_or_bound,
            **self._extra_resample_kwargs,
        )
        if self.channel_conv is not None:
            y = self.channel_conv(y)

        if return_coord_space:
            ret = DenseCoordSpace(
                values=y, coords=query_coords, affine=affine_query_el2coords
            )
        else:
            ret = y

        return ret


class TranslationOnlyResampler(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_features: int,
        out_features: int,
        padding_mode_or_bound="reflection",
        interp_lib: Literal["torch", "interpol"] = "torch",
        resample_kwargs: dict[str, Any] = dict(),
        spacing_tol: float = 1e-5,
        rotation_mat_tol: float = 1e-5,
        # spacing_tol: float = 5e-2,  #!
        # rotation_mat_tol: float = 5e-2,  #!
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=kwargs, warn=True
        )

        super().__init__()

        self._spatial_dims = spatial_dims
        self._in_features = in_features
        self._out_features = out_features

        # Input channels must equal output channels.
        if self._in_features != self._out_features:
            raise ValueError(
                f"Input channels ({self._in_features}) must equal output channels "
                f"({self._out_features})."
            )

        self.resampler = InterpolationResampler(
            spatial_dims=self.spatial_dims,
            in_features=self.in_features,
            out_features=self.out_features,
            mode_or_interpolation="nearest",
            padding_mode_or_bound=padding_mode_or_bound,
            interp_lib=interp_lib,
            **resample_kwargs,
        )
        self._spacing_tol = spacing_tol
        self._rotation_mat_tol = rotation_mat_tol

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    @property
    def in_channels(self):
        return self.in_features

    @property
    def out_channels(self):
        return self.out_features

    _TRANSLATION_ERROR_MSG = (
        "Resampling would not result in a translation-only transformation."
    )

    def _verify_affine_translation_only(
        self,
        x_affine: mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D,
        q_affine: mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D,
    ) -> bool:
        with torch.no_grad():
            x_spacing = mrinr.coords.spacing(x_affine)
            q_spacing = mrinr.coords.spacing(q_affine)

            if not torch.allclose(x_spacing, q_spacing, atol=self._spacing_tol):
                raise ValueError(
                    f"Input and query spacing are not equal: {x_spacing} != {q_spacing}. "
                    + self._TRANSLATION_ERROR_MSG
                )

            # Divide the rotation+scaling+shearing part of the affine by the spacing to get
            # only rotation+shearing. Assume shearing to be negligible, but the tolerance
            # will still be checked against it.
            homog_x_spacing = torch.nn.functional.pad(
                x_spacing, (0, 1), value=1.0, mode="constant"
            )
            x_scale_aff = torch.diag_embed(homog_x_spacing)
            x_rot_mat = mrinr.coords.combine_affines(
                x_affine,
                mrinr.coords.inv_affine(x_scale_aff),
                transform_order_left_to_right=False,
            )[..., :-1, :-1]

            homog_q_spacing = torch.nn.functional.pad(
                q_spacing, (0, 1), value=1.0, mode="constant"
            )
            q_scale_aff = torch.diag_embed(homog_q_spacing)
            q_rot_mat = mrinr.coords.combine_affines(
                q_affine,
                mrinr.coords.inv_affine(q_scale_aff),
                transform_order_left_to_right=False,
            )[..., :-1, :-1]

            if not torch.allclose(x_rot_mat, q_rot_mat, atol=self._rotation_mat_tol):
                raise ValueError(
                    "Input and query rotation matrices are not equal:\n"
                    + f"{x_rot_mat}\n!=\n{q_rot_mat}.\n"
                    + self._TRANSLATION_ERROR_MSG
                )

        return True

    def forward(
        self,
        x: mrinr.typing.Volume | mrinr.typing.Image,
        x_coords: Any,
        query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        affine_query_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        *args,
        return_coord_space: bool = False,
        **kwargs,
    ) -> (
        mrinr.typing.Volume
        | mrinr.typing.Image
        | DenseCoordSpace
        | tuple[
            mrinr.typing.Volume | mrinr.typing.Image | DenseCoordSpace, torch.Generator
        ]
    ):
        # Verify that the input and query affines only contain translations, no scaling
        # or rotation.
        self._verify_affine_translation_only(affine_x_el2coords, affine_query_el2coords)

        # Resample input into the query/target space.
        r = self.resampler(
            x,
            x_coords=x_coords,
            query_coords=query_coords,
            affine_x_el2coords=affine_x_el2coords,
            affine_query_el2coords=affine_query_el2coords,
            return_coord_space=return_coord_space,
            **kwargs,
        )

        return r


class FConvEncoder(torch.nn.Module):
    _is_coord_aware = True

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        downscale_factors: tuple[int],
        block_channels: tuple[int] | int,
        block_subunits: int,
        conv_type: Literal["plain", "res"] = "plain",
        downscale_method: Literal["stride"] = "stride",  # Only strides for now
        downscale_kernel_sizes: tuple[int] | int = 3,
        block_kernel_sizes: tuple[int] | int = 3,
        padding_mode: str = "reflect",
        block_norms: str
        | tuple[str, dict[str, Any]]
        | tuple[None | str | tuple[str, dict[str, Any]]]
        | None = None,
        res_subunits_per_block: Optional[int] = None,
        append_input_spacing: bool = False,
        is_checkpointed: Optional[bool] = None,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self._spatial_dims = spatial_dims
        self._append_input_spacing = append_input_spacing
        if self._append_input_spacing:
            self._in_channels = in_channels + spatial_dims
        else:
            self._in_channels = in_channels
        self._out_channels = out_channels
        self._activate_fn_init_obj = activate_fn
        self._downscale_factors = tuple(downscale_factors)
        self._conv_type = (
            str(conv_type).lower().strip().replace("_", "").replace("-", "")
        )
        if "plain" in self._conv_type:
            self._conv_type = "plain"
        elif "res" in self._conv_type:
            self._conv_type = "res"
            assert (
                res_subunits_per_block is not None
            ), "If using residual blocks, `res_subunits_per_block` must be given."

        n_downscales = len(self._downscale_factors)
        n_blocks = len(self._downscale_factors) + 1
        n_units_per_block = int(block_subunits)

        # Match the block channels, norms, and kernel sizes to the number of blocks.
        if isinstance(block_channels, int):
            block_ch = [block_channels] * n_blocks
        else:
            block_ch = list(block_channels)
        if isinstance(block_kernel_sizes, int):
            block_k = [block_kernel_sizes] * n_blocks
        else:
            block_k = list(block_kernel_sizes)

        if block_norms is None:
            block_n = [None] * n_blocks
        elif isinstance(block_norms, str):
            block_n = [block_norms] * n_blocks
        elif (
            isinstance(block_norms, (list, tuple))
            and len(block_norms) == 2
            and isinstance(block_norms[0], str)
            and isinstance(block_norms[1], dict)
        ):
            block_n = [block_norms] * n_blocks
        else:
            block_n = list(block_norms)

        assert len(block_ch) == len(block_k) == len(block_n) == n_blocks

        # Match downscale kernel sizes to the number of downscales.
        if isinstance(downscale_kernel_sizes, int):
            downscale_k = [downscale_kernel_sizes] * n_downscales
        else:
            downscale_k = downscale_kernel_sizes
        assert len(downscale_k) == n_downscales

        encoder = list()
        for i_block_and_scale, downscale_factor_i, (c_in, c_block) in zip(
            range(n_downscales),
            downscale_factors,
            itertools.pairwise([self.in_channels] + block_ch[:-1]),
            strict=True,
        ):
            block_i = torch.nn.ModuleDict()
            k_i = block_k[i_block_and_scale]
            norm_i = block_n[i_block_and_scale]
            is_first_block = i_block_and_scale == 0

            if is_first_block:
                if self._conv_type == "plain":
                    c = mrinr.nn.PlainCNN(
                        self.spatial_dims,
                        in_channels=c_in,
                        interior_channels=[c_block] * n_units_per_block,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        kernel_sizes=k_i,
                        norms=[norm_i] * (n_units_per_block + 1),
                        padding_mode=padding_mode,
                        end_with_conv=False,
                    )
                elif self._conv_type == "res":
                    c = mrinr.nn.ResCNN(
                        spatial_dims=self.spatial_dims,
                        in_channels=c_in,
                        block_channels=c_block,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        n_blocks=n_units_per_block,
                        subunits_per_block=res_subunits_per_block,
                        kernel_size=k_i,
                        norm=norm_i,
                        padding_mode=padding_mode,
                        end_with_conv=False,
                        determine_bias_from_norm=True,
                        is_checkpointed=bool(is_checkpointed),
                    )
            else:
                if self._conv_type == "plain":
                    c = mrinr.nn.PlainCNN(
                        self.spatial_dims,
                        in_channels=c_in,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        interior_channels=[c_block] * (n_units_per_block - 1),
                        kernel_sizes=k_i,
                        norms=norm_i,
                        padding_mode=padding_mode,
                        end_with_conv=False,
                    )
                elif self._conv_type == "res":
                    c = mrinr.nn.ResCNN(
                        spatial_dims=self.spatial_dims,
                        in_channels=c_in,
                        block_channels=c_block,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        n_blocks=n_units_per_block,
                        subunits_per_block=res_subunits_per_block,
                        kernel_size=k_i,
                        norm=norm_i,
                        padding_mode=padding_mode,
                        end_with_conv=False,
                        determine_bias_from_norm=True,
                        is_checkpointed=bool(is_checkpointed),
                    )
            block_i["conv_block"] = (
                mrinr.nn.spatial_coders.CoordAwareIdentityModuleWrapper(c)
            )
            # Create downsampler module.
            downsampler_i = mrinr.nn.spatial_coders.StridedConvDownsample(
                self.spatial_dims,
                in_channels=c_block,
                out_channels=c_block,
                stride_downscale_factor=downscale_factor_i,
                kernel_size=downscale_k[i_block_and_scale],
                pad_mode=padding_mode if padding_mode != "zeros" else "constant",
                pad_value=0.0 if padding_mode == "zeros" else None,
                # Use norm to determine if bias should be included.
                bias=mrinr.nn.include_bias_given_norm(norm_i),
            )
            block_i["downsampler"] = downsampler_i
            # Create module for calling after the downsampler
            post_downsample = list()
            # Add normalization, if given.
            # if downscale_n[i_block_and_scale] is not None:
            if norm_i is not None:
                post_downsample.append(
                    mrinr.nn.make_norm_module(
                        # downscale_n[i_block_and_scale],
                        norm_i,
                        num_channels_or_features=block_i["downsampler"].out_channels,
                    )
                )
            # Add activation function.
            post_downsample.append(
                mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj)
            )
            # If there are multiple modules, combine into a Sequential.
            if len(post_downsample) > 1:
                block_i["post_downsampler"] = torch.nn.Sequential(*post_downsample)
            # Otherwise, just store the activation function Module alone.
            else:
                block_i["post_downsampler"] = post_downsample[0]

            encoder.append(block_i)
        # Create final conv block, after the final downsampler.
        if self._conv_type == "plain":
            final_c = mrinr.nn.PlainCNN(
                self.spatial_dims,
                in_channels=encoder[-1]["downsampler"].out_channels,
                out_channels=self.out_channels,
                activate_fn=self._activate_fn_init_obj,
                interior_channels=[block_ch[-1]] * (n_units_per_block),
                kernel_sizes=block_k[-1],
                norms=([block_n[-1]] * n_units_per_block),
                padding_mode=padding_mode,
                end_with_conv=True,
            )
        elif self._conv_type == "res":
            final_c = mrinr.nn.ResCNN(
                self.spatial_dims,
                in_channels=encoder[-1]["downsampler"].out_channels,
                block_channels=block_ch[-1],
                out_channels=self.out_channels,
                activate_fn=self._activate_fn_init_obj,
                n_blocks=n_units_per_block,
                subunits_per_block=res_subunits_per_block,
                kernel_size=block_k[-1],
                norm=block_n[-1],
                padding_mode=padding_mode,
                end_with_conv=True,
                determine_bias_from_norm=True,
                is_checkpointed=bool(is_checkpointed),
            )
        encoder.append(
            torch.nn.ModuleDict(
                {
                    "conv_block": mrinr.nn.spatial_coders.CoordAwareIdentityModuleWrapper(
                        final_c
                    )
                }
            )
        )

        self.encoder = torch.nn.ModuleList(encoder)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def out_channels(self):
        return self._out_channels

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        affine_x_el2coords: torch.Tensor,
        *,
        return_coord_space: bool = False,
        **kwargs,
    ):
        if self._append_input_spacing:
            if self.spatial_dims == 2:
                s = einops.repeat(
                    mrinr.coords.spacing(affine_x_el2coords),
                    "b c -> b c x y",
                    b=x.shape[0],
                    c=self.spatial_dims,
                    x=x.shape[2],
                    y=x.shape[3],
                )
            elif self.spatial_dims == 3:
                s = einops.repeat(
                    mrinr.coords.spacing(affine_x_el2coords),
                    "b c -> b c x y z",
                    b=x.shape[0],
                    c=self.spatial_dims,
                    x=x.shape[2],
                    y=x.shape[3],
                    z=x.shape[4],
                )
            y = torch.cat([x, s], dim=1)
        else:
            y = x
        y_coords = x_coords
        affine_y_el2coords = affine_x_el2coords

        for i, block in enumerate(self.encoder):
            is_last_block = i == len(self.encoder) - 1
            y, y_coords, affine_y_el2coords = block["conv_block"](
                y,
                x_coords=y_coords,
                affine_x_el2coords=affine_y_el2coords,
                return_coord_space=True,
            )
            if not is_last_block:
                y, y_coords, affine_y_el2coords = block["downsampler"](
                    x=y,
                    x_coords=y_coords,
                    affine_x_el2coords=affine_y_el2coords,
                    return_coord_space=True,
                )
                # The activation function and normalization should never change any
                # spatial data, so coordinates are not passed to them.
                y = block["post_downsampler"](y)

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=y_coords, affine=affine_y_el2coords
            )
        else:
            r = y

        return r


class FConvDecoder(torch.nn.Module):
    _is_coord_aware = True
    needs_coord_alignment = True

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        upscale_factors: tuple[int],
        block_channels: tuple[int] | int,
        block_subunits: int,
        upscale_method: str | Literal["icnr", "nearest", "fixed-nearest"],
        block_kernel_sizes: tuple[int] | int = 3,
        conv_type: Literal["plain", "res"] = "plain",
        padding_mode: str = "reflect",
        block_norms: str
        | tuple[str, dict[str, Any]]
        | tuple[None | str | tuple[str, dict[str, Any]]]
        | None = None,
        res_subunits_per_block: Optional[int] = None,
        is_checkpointed: Optional[bool] = None,
        append_norm_coords: bool = False,
        **upscale_kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self._spatial_dims = spatial_dims
        self._append_norm_input_coords = append_norm_coords
        if self._append_norm_input_coords:
            self._in_channels = in_channels + spatial_dims
        else:
            self._in_channels = in_channels
        self._out_channels = out_channels
        self._activate_fn_init_obj = activate_fn
        self._upscale_factors = tuple(upscale_factors)

        self._conv_type = (
            str(conv_type).lower().strip().replace("_", "").replace("-", "")
        )
        if "plain" in self._conv_type:
            self._conv_type = "plain"
        elif "res" in self._conv_type:
            self._conv_type = "res"
            assert (
                res_subunits_per_block is not None
            ), "If using residual blocks, `res_subunits_per_block` must be given."

        m = str(upscale_method).lower().strip().replace("_", "")
        if ("nearest" == m) or (m == "nn"):
            self._upscale_method = "nearest"
        # Multiple names could match ICNR shuffling.
        elif ("shuffle" in m) or ("espcn" in m) or ("icnr" in m):
            self._upscale_method = "icnr"
        elif ("fixed-nn" == m) or ("fixed-nearest" == m):
            self._upscale_method = "fixed-nearest"
        else:
            raise ValueError(
                f"Invalid upscale method: {upscale_method}. "
                + "Must be 'nearest', 'fixed-nearest', or 'icnr'."
            )
        n_upscales = len(self._upscale_factors)
        n_blocks = len(self._upscale_factors) + 1
        n_units_per_block = int(block_subunits)

        # Match the block channels, norms, and kernel sizes to the number of blocks.
        if isinstance(block_channels, int):
            block_ch = [block_channels] * n_blocks
        else:
            block_ch = list(block_channels)
        if isinstance(block_kernel_sizes, int):
            block_k = [block_kernel_sizes] * n_blocks
        else:
            block_k = list(block_kernel_sizes)

        if block_norms is None:
            block_n = [None] * n_blocks
        elif isinstance(block_norms, str):
            block_n = [block_norms] * n_blocks
        elif (
            isinstance(block_norms, (list, tuple))
            and len(block_norms) == 2
            and isinstance(block_norms[0], str)
            and isinstance(block_norms[1], dict)
        ):
            block_n = [block_norms] * n_blocks
        else:
            block_n = list(block_norms)

        assert len(block_ch) == len(block_k) == len(block_n) == n_blocks
        # Select class for upsampler, depending on the selected method.
        if self._upscale_method == "nearest":
            Upsampler = mrinr.nn.spatial_coders.NearestNeighborUpsample
        elif self._upscale_method == "icnr":
            Upsampler = mrinr.nn.spatial_coders.ICNRUpsample
            # Also add the activation function as an upsampler kwarg, as it should
            # likely match the activation for the rest of the decoder.
            upscale_kwargs["activate_fn"] = self._activate_fn_init_obj
        elif self._upscale_method == "fixed-nearest":
            Upsampler = mrinr.nn.spatial_coders.NNUpsample

        if self._append_norm_input_coords:
            self.input_coord_norm = mrinr.nn.CoordNorm(
                spatial_dims=self.spatial_dims, affine=True, track_running_stats=True
            )
        else:
            self.input_coord_norm = None
        # Build the decoder.
        decoder = list()
        for i_block_and_scale, upscale_factor_i, (c_in, c_block) in zip(
            range(n_upscales),
            upscale_factors,
            itertools.pairwise([self.in_channels] + block_ch[:-1]),
            strict=True,
        ):
            block_i = torch.nn.ModuleDict()
            k_i = block_k[i_block_and_scale]
            norm_i = block_n[i_block_and_scale]
            is_first_block = i_block_and_scale == 0

            if is_first_block:
                if self._conv_type == "plain":
                    c = mrinr.nn.PlainCNN(
                        self.spatial_dims,
                        in_channels=c_in,
                        interior_channels=[c_block] * n_units_per_block,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        kernel_sizes=k_i,
                        norms=[norm_i] * (n_units_per_block + 1),
                        padding_mode=padding_mode,
                        end_with_conv=False,
                    )
                elif self._conv_type == "res":
                    c = mrinr.nn.ResCNN(
                        spatial_dims=self.spatial_dims,
                        in_channels=c_in,
                        block_channels=c_block,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        n_blocks=n_units_per_block,
                        subunits_per_block=res_subunits_per_block,
                        kernel_size=k_i,
                        norm=norm_i,
                        padding_mode=padding_mode,
                        end_with_conv=False,
                        determine_bias_from_norm=True,
                        is_checkpointed=bool(is_checkpointed),
                    )
            else:
                if self._conv_type == "plain":
                    c = mrinr.nn.PlainCNN(
                        self.spatial_dims,
                        in_channels=c_in,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        interior_channels=[c_block] * (n_units_per_block - 1),
                        kernel_sizes=k_i,
                        norms=norm_i,
                        padding_mode=padding_mode,
                        end_with_conv=False,
                    )
                elif self._conv_type == "res":
                    c = mrinr.nn.ResCNN(
                        spatial_dims=self.spatial_dims,
                        in_channels=c_in,
                        block_channels=c_block,
                        out_channels=c_block,
                        activate_fn=self._activate_fn_init_obj,
                        n_blocks=n_units_per_block,
                        subunits_per_block=res_subunits_per_block,
                        kernel_size=k_i,
                        norm=norm_i,
                        padding_mode=padding_mode,
                        end_with_conv=False,
                        determine_bias_from_norm=True,
                        is_checkpointed=bool(is_checkpointed),
                    )
            block_i["conv_block"] = (
                mrinr.nn.spatial_coders.CoordAwareIdentityModuleWrapper(c)
            )
            # Create the upsampler module.
            block_i["upsampler"] = Upsampler(
                spatial_dims=self.spatial_dims,
                in_channels=c_block,
                out_channels=c_block,
                upscale_factor=upscale_factor_i,
                **upscale_kwargs,
            )

            # Create module for calling after the upsampler
            post_upsample = list()
            # Add normalization, if given.
            if norm_i is not None:
                post_upsample.append(
                    mrinr.nn.make_norm_module(
                        # downscale_n[i_block_and_scale],
                        norm_i,
                        num_channels_or_features=block_i["upsampler"].out_channels,
                    )
                )
            # Add activation function.
            post_upsample.append(
                mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj)
            )
            # If there are multiple modules, combine into a Sequential.
            if len(post_upsample) > 1:
                block_i["post_upsampler"] = torch.nn.Sequential(*post_upsample)
            # Otherwise, just store the activation function Module alone.
            else:
                block_i["post_upsampler"] = post_upsample[0]

            decoder.append(block_i)
        # Create final conv block, after the final upsampler.
        if self._conv_type == "plain":
            final_c = mrinr.nn.PlainCNN(
                self.spatial_dims,
                in_channels=decoder[-1]["upsampler"].out_channels,
                out_channels=self.out_channels,
                activate_fn=self._activate_fn_init_obj,
                interior_channels=[block_ch[-1]] * (n_units_per_block),
                kernel_sizes=block_k[-1],
                norms=([block_n[-1]] * n_units_per_block),
                padding_mode=padding_mode,
                end_with_conv=True,
            )
        elif self._conv_type == "res":
            final_c = mrinr.nn.ResCNN(
                self.spatial_dims,
                in_channels=decoder[-1]["upsampler"].out_channels,
                block_channels=block_ch[-1],
                out_channels=self.out_channels,
                activate_fn=self._activate_fn_init_obj,
                n_blocks=n_units_per_block,
                subunits_per_block=res_subunits_per_block,
                kernel_size=block_k[-1],
                norm=block_n[-1],
                padding_mode=padding_mode,
                end_with_conv=True,
                determine_bias_from_norm=True,
                is_checkpointed=bool(is_checkpointed),
            )
        decoder.append(
            torch.nn.ModuleDict(
                {
                    "conv_block": mrinr.nn.spatial_coders.CoordAwareIdentityModuleWrapper(
                        final_c
                    )
                }
            )
        )

        self.decoder = torch.nn.ModuleList(decoder)

        # Also create a translation resampler, in the event that decoder outputs need
        # alignment to a target coordinate space.
        self.output_resampler = mrinr.nn.spatial_coders.TranslationOnlyResampler(
            spatial_dims=self.spatial_dims,
            in_features=self.out_channels,
            out_features=self.out_channels,
        )

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def out_channels(self):
        return self._out_channels

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def align_target_coords_to_input(
        self, target_affine_el2coords, target_coords
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the target coordinates, determine the input coordinates that will
        produce the target coordinates when given to the decoder, +/- some translation.
        """
        target_spatial_shape = np.asarray(
            target_coords.shape[:-1][-self.spatial_dims :]
        )
        upscale_factor = np.prod(self._upscale_factors)
        input_spatial_shape = tuple(target_spatial_shape // upscale_factor)

        # Cast to double precision, as small numerical errors may invalidate later
        # model inputs.
        input_affine, input_coords = mrinr.coords.resize_affine(
            affine_x_el2coords=target_affine_el2coords.double(),
            in_spatial_shape=target_spatial_shape,
            target_spatial_shape=input_spatial_shape,
            centered=True,
            return_coord_grid=True,
        )

        return (
            input_affine.to(target_affine_el2coords.dtype),
            input_coords.to(target_affine_el2coords.dtype),
        )

    def align_output_to_target(
        self,
        y: mrinr.typing.Volume | mrinr.typing.Image,
        y_coords: Any,
        affine_y: mrinr.typing.HomogeneousAffine3D | mrinr.typing.HomogeneousAffine2D,
        affine_target: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        target_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        *,
        return_coord_space: bool = False,
    ):
        # Align model output to the target space.
        r = self.output_resampler(
            x=y,
            x_coords=y_coords,
            query_coords=target_coords,
            affine_x_el2coords=affine_y,
            affine_query_el2coords=affine_target,
            return_coord_space=return_coord_space,
        )
        # r = self.output_resampler(
        #     x=y.double(),
        #     x_coords=y_coords.double(),
        #     query_coords=target_coords.double(),
        #     affine_x_el2coords=affine_y.double(),
        #     affine_query_el2coords=affine_target.double(),
        #     return_coord_space=return_coord_space,
        # )
        if return_coord_space:
            r = [t.to(y.dtype) for t in r]
        else:
            r = r.to(y.dtype)
        return r

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        affine_x_el2coords: torch.Tensor,
        *,
        return_coord_space: bool = False,
        **kwargs,
    ):
        if self._append_norm_input_coords:
            y = torch.cat(
                [
                    x,
                    mrinr.nn.coords_as_channels(
                        self.input_coord_norm(x_coords), has_batch_dim=True
                    ),
                ],
                dim=1,
            )
        else:
            y = x

        y_coords = x_coords
        affine_y_el2coords = affine_x_el2coords

        for i, block in enumerate(self.decoder):
            is_last_block = i == len(self.decoder) - 1
            y, y_coords, affine_y_el2coords = block["conv_block"](
                y,
                x_coords=y_coords,
                affine_x_el2coords=affine_y_el2coords,
                return_coord_space=True,
            )
            if not is_last_block:
                y, y_coords, affine_y_el2coords = block["upsampler"](
                    x=y,
                    x_coords=y_coords,
                    affine_x_el2coords=affine_y_el2coords,
                    return_coord_space=True,
                )
                # The activation function and normalization should never change any
                # spatial data, so coordinates are not passed to them.
                y = block["post_upsampler"](y)

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=y_coords, affine=affine_y_el2coords
            )
        else:
            r = y

        return r
