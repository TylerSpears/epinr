# -*- coding: utf-8 -*-
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import torch

import mrinr

from ._sr import ICNRUpsample

__all__ = [
    "IdentityResampler",
    "InterpolationResampler",
    "TranslationOnlyResampler",
    "StridedConvDownsample",
    "ConvShuffleUpsample",
    "NNUpsample",
    "NNDownsample3D",
]


class AffineReturn(NamedTuple):
    x: mrinr.typing.Image | mrinr.typing.Volume
    affine: mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D


class IdentityResampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        affine: Optional[torch.Tensor] = None,
        return_affine: bool = False,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        if return_affine and affine is None:
            raise ValueError("If `return_affine` is True, `affine` must be provided.")

        y = x

        if return_affine:
            r = AffineReturn(y, affine)
        else:
            r = y

        return r


class InterpolationResampler(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_features: int,
        out_features: int,
        mode_or_interpolation: str,
        padding_mode_or_bound: str,
        interp_lib: Literal["torch", "interpol"] = "torch",
        resample_kwargs: dict[str, Any] = dict(),
        channel_conv: Optional[bool] = None,
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

        make_conv = (
            channel_conv
            if channel_conv is not None
            else self.in_channels != self.out_channels
        )

        if (self.in_channels != self.out_channels) or make_conv:
            if not make_conv:
                raise ValueError(
                    "Input and output channels must match if a conv layer is not allowed."
                )
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
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine_x: mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D,
        target_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_target: Optional[
            mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D
        ] = None,
        return_affine: bool = False,
        **kwargs,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        y = mrinr.grid_resample(
            x=x,
            affine_x_el2coords=affine_x,
            sample_coords=target_coords,
            interp_lib=self._interp_lib,
            mode_or_interpolation=self._mode_or_interpolation,
            padding_mode_or_bound=self._padding_mode_or_bound,
            **self._extra_resample_kwargs,
        )
        if self.channel_conv is not None:
            y = self.channel_conv(y)

        if return_affine:
            ret = AffineReturn(y, affine_target)
        else:
            ret = y

        return ret


class TranslationOnlyResampler(InterpolationResampler):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_features: int,
        out_features: int,
        padding_mode_or_bound: str = "replicate",
        interp_lib: Literal["torch", "interpol"] = "interpol",
        resample_kwargs: dict[str, Any] = dict(),
        spacing_tol: float = 1e-5,
        rotation_mat_tol: float = 1e-5,
        **kwargs,
    ):
        _init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=kwargs, warn=True
        )

        super().__init__(
            spatial_dims=spatial_dims,
            in_features=in_features,
            out_features=out_features,
            mode_or_interpolation="nearest",
            padding_mode_or_bound=padding_mode_or_bound,
            interp_lib=interp_lib,
            resample_kwargs=resample_kwargs,
            channel_conv=False,
            **kwargs,
        )
        self._init_kwargs = _init_kwargs

        self._spacing_tol = spacing_tol
        self._rotation_mat_tol = rotation_mat_tol

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
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine_x: mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D,
        target_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_target: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        return_affine: bool = False,
        **kwargs,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        # Verify that the input and query affines only contain translations, no scaling
        # or rotation.
        self._verify_affine_translation_only(affine_x, affine_target)

        return super().forward(
            x=x,
            affine_x=affine_x,
            target_coords=target_coords,
            affine_target=affine_target,
            return_affine=return_affine,
            **kwargs,
        )


class StridedConvDownsample(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        stride_downscale_factor: int,
        kernel_size: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
        pad_value: Optional[float] = None,
        device=None,
        dtype=None,
        conv_padding_as_affine_translate: bool = False,
    ):
        super().__init__()

        self._spatial_dims = spatial_dims
        if self._spatial_dims == 2:
            conv_cls = torch.nn.Conv2d
        elif self._spatial_dims == 3:
            conv_cls = torch.nn.Conv3d
        else:
            raise ValueError("Spatial dims must be 2 or 3.")
        # Canonicalize the padding mode. Pytorch has inconsistent naming for some
        # padding modes, so if we get an 'interpolation padding mode' we can convert it
        # to work with conv classes.
        if pad_mode is not None:
            pad_mode = str(pad_mode).strip().lower()
            if "zero" in pad_mode:
                pad_mode = "zeros"
            elif "reflect" in pad_mode:
                pad_mode = "reflect"
            elif (
                ("replicat" in pad_mode)
                or ("border" in pad_mode)
                or ("edge" in pad_mode)
            ):
                pad_mode = "replicate"
            elif ("circ" in pad_mode) or ("wrap" in pad_mode) or (pad_mode == "dft"):
                pad_mode = "circular"
        self._downscale_factor = int(stride_downscale_factor)
        k = int(kernel_size)
        assert (k % 2) == 1, "Kernel size must be odd."

        conv_padding = (k - 1) // 2 if conv_padding_as_affine_translate else 0
        conv_padding_mode = (
            "zeros"
            if (pad_mode == "constant" or not conv_padding_as_affine_translate)
            else pad_mode
        )
        self.conv = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=self._downscale_factor,
            padding=conv_padding,
            padding_mode=conv_padding_mode,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._pad_mode = pad_mode
        self._pad_value = pad_value

        if conv_padding_as_affine_translate:
            self._kernel_padding_low = np.array(((k - 1) // 2,) * self.spatial_dims)
        else:
            self._kernel_padding_low = np.zeros(self.spatial_dims, dtype=int)
        self._kernel_padding_high = self._kernel_padding_low.copy()

        self._apply_conv_padding_to_affine = conv_padding_as_affine_translate

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine: Optional[
            mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D
        ] = None,
        return_affine: bool = False,
        **kwargs,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        # Check if x needs to be padded to be divisible by the stride amount.
        spatial_shape = np.array(tuple(x.shape)[-self._spatial_dims :])
        # Start with padding that would be performed by the conv layer.
        affine_pad_low = self._kernel_padding_low
        affine_pad_high = self._kernel_padding_high
        # If the input shape is not divisible by the stride amount, then we need to
        # pad the input.
        if np.any(spatial_shape % self._downscale_factor):
            # Put padding on the "high" end of each spatial dimension, to potentially
            # avoid translation of the origin in the affine matrix.
            stride_factor_pad_high = spatial_shape % self._downscale_factor
            affine_pad_high += stride_factor_pad_high
        else:
            stride_factor_pad_high = np.zeros(self.spatial_dims, dtype=int)
        # If any padding is needed, then pad outside of the conv layer.
        if np.any(stride_factor_pad_high):
            # No padding on batch or channel dimensions.
            pad_low_high = np.stack(
                [tuple([0] * self.spatial_dims), tuple(stride_factor_pad_high)],
                axis=1,
                dtype=int,
            )
            # Flip and flatten to get the padding in the order expected by PyTorch.
            pytorch_pads = tuple(np.flip(pad_low_high, axis=0).flatten().tolist())
            y = torch.nn.functional.pad(
                x,
                pytorch_pads,
                mode=self._pad_mode,
                value=self._pad_value,
            )
        else:
            y = x
        # May or may not be padded from x.
        in_spatial_shape = tuple(y.shape[-self.spatial_dims :])

        y = self.conv(y)

        if not return_affine:
            r = y
        else:
            if affine is None:
                raise ValueError("Affine must be provided if return_affine=True.")
            out_spatial_shape = tuple(y.shape[-self.spatial_dims :])
            # Translate the affine by the pad_low amount. The upper padding does not
            # influence the affine matrix.
            if np.any(affine_pad_low):
                affine_y_el2coords = mrinr.coords.pad_affine(
                    affine=affine, pad_low=affine_pad_low
                )
            else:
                affine_y_el2coords = affine
            # Cast to double precision to reduce floating point errors.
            affine_y_el2coords = mrinr.coords.resize_affine(
                affine_x_el2coords=affine_y_el2coords.double(),
                in_spatial_shape=in_spatial_shape,
                target_spatial_shape=out_spatial_shape,
                centered=True,
                return_coord_grid=False,
            )
            r = AffineReturn(y, affine_y_el2coords.to(affine.dtype))

        return r


class NNDownsample3D(torch.nn.Module):
    def __init__(
        self,
        downscale_factor: int | tuple[int, int, int],
        pad_mode: str = "constant",
        pad_value: Optional[float] = None,
    ):
        super().__init__()
        self.spatial_dims = 3
        self.downscale_factor = downscale_factor
        if isinstance(downscale_factor, int):
            self.downscale_factor = (downscale_factor,) * self.spatial_dims
        self.scale_factor = (1 / np.asarray(self.downscale_factor)).tolist()

        self._pad_mode = pad_mode
        self._pad_value = pad_value

    def forward(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine: Optional[
            mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D
        ] = None,
        return_affine: bool = False,
        **kwargs,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        # Check if x needs to be padded to be divisible by the stride amount.
        spatial_shape = np.array(tuple(x.shape)[-self.spatial_dims :])
        # Start with padding that would be performed by the conv layer.
        pad_low = np.zeros(self.spatial_dims, dtype=int)
        pad_high = np.zeros(self.spatial_dims, dtype=int)
        # If the input shape is not divisible by the stride amount, then we need to
        # pad the input.
        if np.any(spatial_shape % self.downscale_factor):
            # Put padding on the "high" end of each spatial dimension, to potentially
            # avoid translation of the origin in the affine matrix.
            factor_pad_high = spatial_shape % self.downscale_factor
            pad_high += factor_pad_high

        # If any padding is needed, then pad outside of the conv layer.
        if np.any(pad_low) or np.any(pad_high):
            # No padding on batch or channel dimensions.
            pad_low_high = np.stack(
                [tuple(pad_low), tuple(pad_high)], axis=1, dtype=int
            )
            # Flip and flatten to get the padding in the order expected by PyTorch.
            pytorch_pads = tuple(np.flip(pad_low_high, axis=0).flatten().tolist())
            y = torch.nn.functional.pad(
                x,
                pytorch_pads,
                mode=self._pad_mode,
                value=self._pad_value,
            )
        else:
            y = x
        # May or may not be padded from x.
        in_spatial_shape = tuple(y.shape[-self.spatial_dims :])
        out_spatial_shape = tuple(np.asarray(in_spatial_shape) // self.downscale_factor)

        y = torch.nn.functional.interpolate(
            y,
            size=out_spatial_shape,
            mode="nearest-exact",
        )

        if not return_affine:
            r = y
        else:
            if affine is None:
                raise ValueError("Affine must be provided if return_affine=True.")
            # Translate the affine by the pad_low amount. The upper padding does not
            # influence the affine matrix.
            if np.any(pad_low):
                affine_y_el2coords = mrinr.coords.pad_affine(
                    affine=affine, pad_low=pad_low
                )
            else:
                affine_y_el2coords = affine
            # Cast to double precision to reduce floating point errors.
            affine_y_el2coords = mrinr.coords.resize_affine(
                affine_x_el2coords=affine.double(),
                in_spatial_shape=in_spatial_shape,
                target_spatial_shape=out_spatial_shape,
                centered=True,
                return_coord_grid=False,
            )
            r = AffineReturn(y, affine_y_el2coords.to(affine.dtype))

        return r


class ConvShuffleUpsample(ICNRUpsample):
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
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict(), warn=True
        )
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
            activate_fn=activate_fn,
            blur=blur,
            kernel_size=kernel_size,
            conv_padding_mode=conv_padding_mode,
            icnr_init=icnr_init,
        )

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine: Optional[
            mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D
        ] = None,
        return_affine: bool = False,
        **kwargs,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        y = super().forward(x)

        if return_affine:
            if affine is None:
                raise ValueError("Affine must be provided if return_affine=True.")
            x_spatial_shape = np.asarray(x.shape[-self.spatial_dims :])
            y_spatial_shape = x_spatial_shape * self.upscale_factor
            # Cast to double precision to reduce floating point errors.
            y_affine = mrinr.coords.resize_affine(
                affine_x_el2coords=affine.double(),
                in_spatial_shape=tuple(x_spatial_shape),
                target_spatial_shape=tuple(y_spatial_shape),
                centered=True,
                return_coord_grid=False,
            )
            r = AffineReturn(y, y_affine.to(affine.dtype))
        else:
            r = y
        return r


class NNUpsample(torch.nn.Module):
    _is_coord_aware: bool = True

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict(), warn=True
        )
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert (
            self.in_channels == self.out_channels
        ), "Input and output channels must match."

        self._upscale_factor = int(upscale_factor)
        self.upsampler = torch.nn.Upsample(
            scale_factor=self._upscale_factor,
            mode="nearest",
        )

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    # def extra_repr(self) -> str:
    #     return self.upsampler.extra_repr().replace("Upsample", "NNUpsample")

    def forward(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine: Optional[
            mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D
        ] = None,
        return_affine: bool = False,
        **kwargs,
    ) -> mrinr.typing.Image | mrinr.typing.Volume | AffineReturn:
        y = self.upsampler(x)

        if return_affine:
            if affine is None:
                raise ValueError("Affine must be provided if return_affine=True.")
            x_spatial_shape = np.asarray(x.shape[-self.spatial_dims :])
            y_spatial_shape = x_spatial_shape * self._upscale_factor
            # Cast to double precision to reduce floating point errors.
            y_affine = mrinr.coords.resize_affine(
                affine_x_el2coords=affine.double(),
                in_spatial_shape=tuple(x_spatial_shape),
                target_spatial_shape=tuple(y_spatial_shape),
                centered=True,
                return_coord_grid=False,
            )
            r = AffineReturn(y, y_affine.to(affine.dtype))
        else:
            r = y
        return r
