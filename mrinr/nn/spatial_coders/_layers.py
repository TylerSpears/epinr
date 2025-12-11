# -*- coding: utf-8 -*-
from typing import Any, Literal, Optional, Union

import einops.layers.torch
import numpy as np
import torch

import mrinr

from .. import _sr as sr
from ._nets import InterpolationResampler

__all__ = [
    "CoordsAsChannels",
    "ChannelsAsCoords",
    "StridedConvDownsample",
    "ICNRUpsample",
    "NearestNeighborUpsample",
    "NNUpsample",
    "NNDownsample3D",
]


class CoordsAsChannels(einops.layers.torch.Rearrange):
    def __init__(self, spatial_dims: int):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        if spatial_dims == 2:
            super().__init__("b x y coord -> b coord x y", coord=spatial_dims)
        elif spatial_dims == 3:
            super().__init__("b x y z coord -> b coord x y z", coord=spatial_dims)
        else:
            raise ValueError(f"Invalid spatial_dims {spatial_dims}")
        self.spatial_dims = spatial_dims

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return


class ChannelsAsCoords(einops.layers.torch.Rearrange):
    def __init__(self, spatial_dims: int):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )

        if spatial_dims == 2:
            super().__init__("b coord x y -> b x y coord", coord=spatial_dims)
        elif spatial_dims == 3:
            super().__init__("b coord x y z -> b x y z coord", coord=spatial_dims)
        else:
            raise ValueError(f"Invalid spatial_dims {spatial_dims}")
        self.spatial_dims = spatial_dims

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return


class StridedConvDownsample(torch.nn.Module):
    _is_coord_aware: bool = True

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
    ):
        super().__init__()

        self._spatial_dims = spatial_dims
        if self._spatial_dims == 2:
            conv_cls = torch.nn.Conv2d
        elif self._spatial_dims == 3:
            conv_cls = torch.nn.Conv3d
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

        self.conv = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=self._downscale_factor,
            padding=(k - 1) // 2,
            padding_mode="zeros" if pad_mode == "constant" else pad_mode,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._pad_mode = pad_mode
        self._pad_value = pad_value

        self.downscale_factor = stride_downscale_factor
        # If padding needs to be done to make the input size divisible by the stride,
        # then padding is only performed on the "high" (up, right, etc.) side of the
        # input. This way there is no "low" side padding, which would require a
        # translation of the origin in the affine matrix.
        self._pad_low = (0,) * self._spatial_dims

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
        x: torch.Tensor,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
    ) -> Union[torch.Tensor, "mrinr.nn.spatial_coders.DenseCoordSpace"]:
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )

        # Check if x needs to be padded to be divisible by the stride amount.
        spatial_shape = np.array(tuple(x.shape)[-self._spatial_dims :])
        if np.any(spatial_shape % self.downscale_factor):
            # Put all padding on the "high" end of each spatial dimension.
            # No change to the affine is needed with only high padding.
            pad_high = spatial_shape % self.downscale_factor
            # No padding on batch or channel dimensions.
            spatial_pad_low_high = np.stack(
                [self._pad_low, tuple(pad_high)], axis=1, dtype=int
            )
            # Flip and flatten to get the padding in the order expected by PyTorch.
            pytorch_pads = tuple(
                np.flip(spatial_pad_low_high, axis=0).flatten().tolist()
            )
            y = torch.nn.functional.pad(
                x,
                pytorch_pads,
                mode=self._pad_mode,
                value=self._pad_value,
            )
        else:
            y = x
        # May or may not be padded from x.
        in_spatial_shape = tuple(y.shape)[-self._spatial_dims :]
        y = self.conv(y)

        if not return_coord_space:
            r = y
        else:
            out_spatial_shape = tuple(
                (np.asarray(in_spatial_shape) // self.downscale_factor).tolist()
            )
            if out_spatial_shape != tuple(y.shape)[-self.spatial_dims :]:
                raise ValueError(
                    f"Output spatial shape {tuple(y.shape)[-self.spatial_dims :]} "
                    + f"does not match the expected shape {out_spatial_shape}."
                )
            # Cast to double precision to reduce floating point errors.
            affine_y_el2coords, y_coords = mrinr.coords.resize_affine(
                affine_x_el2coords=affine_x_el2coords.double(),
                in_spatial_shape=in_spatial_shape,
                target_spatial_shape=out_spatial_shape,
                centered=True,
                return_coord_grid=True,
            )
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y,
                coords=y_coords.to(affine_x_el2coords.dtype),
                affine=affine_y_el2coords.to(affine_x_el2coords.dtype),
            )
        return r


class NNDownsample3D(torch.nn.Module):
    def __init__(self, downscale_factor: int | tuple[int, int, int]):
        super().__init__()
        self.spatial_dims = 3
        self.downscale_factor = downscale_factor
        if isinstance(downscale_factor, int):
            self.downscale_factor = (downscale_factor,) * 3
        self.scale_factor = (1 / np.asarray(self.downscale_factor)).tolist()

    def forward(
        self,
        x: mrinr.typing.ScalarVolume,
        affine: Optional[mrinr.typing.HomogeneousAffine3D] = None,
        return_affine: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="nearest-exact",
            recompute_scale_factor=True,
        )
        in_shape = tuple(x.shape[-3:])
        out_shape = tuple(y.shape[-3:])

        if return_affine:
            assert affine is not None, "Affine must be provided if return_affine=True."
            # Adjust the affine matrix to reflect the new shape.
            # Cast to double precision to reduce floating point errors.
            affine_y = mrinr.coords.resize_affine(
                affine.double(),
                in_spatial_shape=in_shape,
                target_spatial_shape=out_shape,
                centered=True,
                return_coord_grid=False,
                return_transform_mat=False,
            ).to(dtype=affine.dtype)
            r = y, affine_y
        else:
            r = y
        return r


class ICNRUpsample(sr.ICNRUpsample):
    _is_coord_aware: bool = True

    def forward(
        self,
        x: torch.Tensor,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
    ) -> Union[torch.Tensor, "mrinr.nn.spatial_coders.DenseCoordSpace"]:
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )

        y = super().forward(x)

        if not return_coord_space:
            r = y
        else:
            in_spatial_shape = tuple(x.shape)[-self.spatial_dims :]
            out_spatial_shape = tuple(
                (np.asarray(in_spatial_shape) * self.upscale_factor).tolist()
            )
            if out_spatial_shape != tuple(y.shape)[-self.spatial_dims :]:
                raise ValueError(
                    f"Output spatial shape {tuple(y.shape)[-self.spatial_dims :]} "
                    + f"does not match the expected shape {out_spatial_shape}."
                )
            if in_spatial_shape != out_spatial_shape:
                # Cast to double precision to reduce floating point errors.
                affine_y_el2coords, y_coords = mrinr.coords.resize_affine(
                    affine_x_el2coords=affine_x_el2coords.double(),
                    in_spatial_shape=in_spatial_shape,
                    target_spatial_shape=out_spatial_shape,
                    centered=True,
                    return_coord_grid=True,
                )
            else:
                y_coords = x_coords
                affine_y_el2coords = affine_x_el2coords
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y,
                coords=y_coords.to(affine_x_el2coords.dtype),
                affine=affine_y_el2coords.to(affine_x_el2coords.dtype),
            )

        return r


class NearestNeighborUpsample(InterpolationResampler):
    _is_coord_aware: bool = True

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        # mode_or_interpolation,
        padding_mode: Literal["zeros", "border", "reflection"] = "reflection",
        interp_lib: str = "torch",
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_features=in_channels,
            out_features=out_channels,
            mode_or_interpolation="nearest",
            padding_mode_or_bound=padding_mode,
            interp_lib=interp_lib,
        )
        # Overwrite the init_kwargs dict.
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict(), warn=True
        )

        self._upscale_factor = upscale_factor

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(
        self,
        x: torch.Tensor,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
    ) -> Union[torch.Tensor, "mrinr.nn.spatial_coders.DenseCoordSpace"]:
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )
        x_spatial_shape = np.asarray(x.shape[-self.spatial_dims :])
        y_spatial_shape = x_spatial_shape * self._upscale_factor
        # Cast to double precision to reduce floating point errors.
        y_affine, y_coords = mrinr.coords.resize_affine(
            affine_x_el2coords=affine_x_el2coords.double(),
            in_spatial_shape=tuple(x_spatial_shape),
            target_spatial_shape=tuple(y_spatial_shape),
            centered=True,
            return_coord_grid=True,
        )

        return super().forward(
            x=x,
            x_coords=x_coords,
            affine_x_el2coords=affine_x_el2coords,
            query_coords=y_coords.to(affine_x_el2coords.dtype),
            affine_query_el2coords=y_affine.to(affine_x_el2coords.dtype),
            return_coord_space=return_coord_space,
            return_rng=False,
        )


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

    def forward(
        self,
        x: torch.Tensor,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
    ) -> Union[torch.Tensor, "mrinr.nn.spatial_coders.DenseCoordSpace"]:
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )
        x_spatial_shape = np.asarray(x.shape[-self.spatial_dims :])
        y_spatial_shape = x_spatial_shape * self._upscale_factor
        # Cast to double precision to reduce floating point errors.
        y_affine, y_coords = mrinr.coords.resize_affine(
            affine_x_el2coords=affine_x_el2coords.double(),
            in_spatial_shape=tuple(x_spatial_shape),
            target_spatial_shape=tuple(y_spatial_shape),
            centered=True,
            return_coord_grid=True,
        )

        y = self.upsampler(x)
        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y,
                coords=y_coords.to(affine_x_el2coords),
                affine=y_affine.to(affine_x_el2coords),
            )
        else:
            r = y
        return r

        # return super().forward(
        #     x=x,
        #     x_coords=x_coords,
        #     affine_x_el2coords=affine_x_el2coords,
        #     query_coords=y_coords.to(affine_x_el2coords.dtype),
        #     affine_query_el2coords=y_affine.to(affine_x_el2coords.dtype),
        #     return_coord_space=return_coord_space,
        #     return_rng=False,
        # )
