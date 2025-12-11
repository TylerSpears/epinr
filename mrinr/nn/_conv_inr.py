# -*- coding: utf-8 -*-
# Module for INRs built with convolutional layers instead of MLPs.

import collections
from typing import Any, Literal, Optional, Union

import einops
import einops.layers.torch
import numpy as np
import torch

import mrinr

from ._conv_nets import ResCNN
from ._inr import _SpatialINRBase

__all__ = [
    "LTEConv",
    "CoordNorm",
    "_ResConvINR",
    "_ensemble_coords_in_x",
    "_sample_with_ensemble",
    "_ensemble_delta__linear_weights",
]


def _get_chunked_inference_features(
    x: torch.Tensor,
    max_chunks: int,
    spatial_indices: tuple[int],
) -> tuple[tuple[torch.Tensor, ...], int]:
    spatial_shapes = np.asarray([x.shape[i] for i in spatial_indices], dtype=int)
    # Ensure that the number of chunks does not exceed 2*dim_size - 1, as any more
    # chunks would produce at least one chunk with a size of 1 in that dimension, which
    # may produce different outputs in some functions.
    max_spatial_chunks = max((np.max(spatial_shapes).item() // 2) - 1, 1)

    n_chunks = min(max_chunks, max_spatial_chunks)
    chunk_dim = spatial_indices[np.argmax(spatial_shapes).item()]
    x_chunked = torch.chunk(x, n_chunks, dim=chunk_dim)

    return x_chunked, chunk_dim


def _get_dim_chunk_indices_affines(
    dim_size: int,
    window_edge_len: int,
    max_chunks: int,
    affine: torch.Tensor,
    affine_translate_dim: int,
) -> tuple[np.ndarray, np.ndarray, tuple[torch.Tensor, ...]]:
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
    chunk_end_idx += window_edge_len
    # Reduce the start indices by the context length.
    chunk_start_idx -= window_edge_len
    # Ensure both are in bounds.
    chunk_start_idx = np.clip(chunk_start_idx, 0, dim_size)
    chunk_end_idx = np.clip(chunk_end_idx, 0, dim_size)

    # Translate the affines such that the element->coordinate translation remains
    # valid for all chunks. Otherwise, functions that rely on the shape of the
    # image/volume for grid sampling will have incorrect coordinate grids.
    spatial_dims = affine.shape[-1] - 1
    affines = list()
    for start_idx in chunk_start_idx.tolist():
        # Create translation matrix applied to the element coordinates in a single
        # spatial dimension.
        t = torch.eye(spatial_dims + 1, dtype=affine.dtype, device=affine.device)
        # Only translate the last column of the spatial dimension being translated.
        t[..., affine_translate_dim, -1] = start_idx
        # Apply translation in (pi|vo)xel coordinates, then combine with the original.
        chunk_affine = mrinr.coords.combine_affines(
            t.to(torch.float64),
            affine.to(torch.float64),
            transform_order_left_to_right=True,
        ).to(affine)
        affines.append(chunk_affine)

    return chunk_start_idx, chunk_end_idx, tuple(affines)


class CoordNorm(torch.nn.Module):
    def __init__(self, spatial_dims: Literal[2, 3], **inst_norm_kwargs):
        super().__init__()
        self.spatial_dims = spatial_dims
        coord2channel = einops.layers.torch.Rearrange(
            "b x ... dim -> b 1 (dim x) ...", dim=self.spatial_dims
        )
        if self.spatial_dims == 2:
            norm = torch.nn.InstanceNorm2d(num_features=1, **inst_norm_kwargs)
        elif self.spatial_dims == 3:
            norm = torch.nn.InstanceNorm3d(num_features=1, **inst_norm_kwargs)
        else:
            raise ValueError(f"Invalid spatial_dims: {self.spatial_dims}")

        channel2coord = einops.layers.torch.Rearrange(
            "b 1 (dim x) ... -> b x ... dim", dim=self.spatial_dims
        )

        self.coord_norm = torch.nn.Sequential(
            collections.OrderedDict(
                coord2channel=coord2channel, norm=norm, channel2coord=channel2coord
            )
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.coord_norm(c)


class _ResConvINR(ResCNN):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        block_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        n_blocks: int,
        subunits_per_block: int,
        norm: None | str | tuple[str, dict[str, Any]] | None = None,
        end_with_conv: bool = True,
        zero_final_conv: bool = False,
        max_inference_chunks: Optional[int] = None,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            block_channels=block_channels,
            out_channels=out_channels,
            activate_fn=activate_fn,
            n_blocks=n_blocks,
            subunits_per_block=subunits_per_block,
            end_with_conv=end_with_conv,
            kernel_size=1,
            # padding_mode=None,
            norm=norm,
            determine_bias_from_norm=True,
        )
        self._max_chunks = max_inference_chunks
        if isinstance(self._max_chunks, int) and self._max_chunks == 1:
            self._max_chunks = None

        if zero_final_conv:
            self.conv_blocks[-1].conv.weight.data.zero_()
            if self.conv_blocks[-1].conv.bias is not None:
                self.conv_blocks[-1].conv.bias.data.zero_()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if not self.training and not x.requires_grad and self._max_chunks is not None:
    #         # Because all convs have kernel size 1 (stride 1, no dilation, etc), we can
    #         # arbitrarily split the input along any spatial and batch dimensions and
    #         # recombine for the exact same output. This is useful for reducing memory
    #         # usage during inference with large volumetric data (there would be no
    #         # memory savings if gradients were being stored).
    #         spatial_indices = (2, 3) if self.spatial_dims == 2 else (2, 3, 4)
    #         x_chunked, chunk_dim = _get_chunked_inference_features(
    #             x, self._max_chunks, spatial_indices=spatial_indices
    #         )
    #         y = list()
    #         for x_chunk in x_chunked:
    #             y.append(super().forward(x_chunk))
    #         y = torch.cat(y, dim=chunk_dim)
    #     else:
    #         y = super().forward(x)

    #     return y


def _ensemble_coords_in_x(
    query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
    affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
    | mrinr.typing.HomogeneousAffine2D,
) -> torch.Tensor:
    spatial_dims = query_coords.shape[-1]

    # Bring the query coordinates into the el space of x and get the "bottom" (left,
    # lower, backwards) corner to create the starting point for the ensemble.
    q_el = (
        mrinr.coords.transform_coords(
            query_coords.double(),
            mrinr.coords.inv_affine(affine_x_el2coords.double()),
            broadcast_batch=False,
        )
        .round(decimals=6)
        .to(query_coords)
    ).floor()

    # Expand the query el coordinates in el space to create the ensemble, then bring
    # those coordinates back into the x coordinate space.
    ensemble_el_offsets = torch.Tensor([0.0, 1.0]).to(query_coords)
    # Shape (2^spatial_dims, spatial_dims)
    ensemble_el_offsets = torch.cartesian_prod(*([ensemble_el_offsets] * spatial_dims))
    if spatial_dims == 2:
        ensemble_el_offsets = einops.rearrange(
            ensemble_el_offsets,
            "(lh_a lh_b) dim -> 1 lh_a lh_b 1 1 dim",
            lh_a=2,
            lh_b=2,
        )
        q_el = (
            einops.rearrange(q_el, "b x y dim -> b 1 1 x y dim") + ensemble_el_offsets
        )
        # Fold the a and b axes into the x and y axes for coordinate transformation,
        # then unfold them for the output.
        ensemble_coords = mrinr.coords.transform_coords(
            einops.rearrange(
                q_el, "b lh_a lh_b x y dim -> b (lh_a x) (lh_b y) dim", lh_a=2, lh_b=2
            ),
            affine_x_el2coords,
            broadcast_batch=False,
        )
        ensemble_coords = einops.rearrange(
            ensemble_coords,
            "b (lh_a x) (lh_b y) dim -> b lh_a lh_b x y dim",
            lh_a=2,
            lh_b=2,
        )
    elif spatial_dims == 3:
        ensemble_el_offsets = einops.rearrange(
            ensemble_el_offsets,
            "(lh_a lh_b lh_c) dim -> 1 lh_a lh_b lh_c 1 1 1 dim",
            lh_a=2,
            lh_b=2,
            lh_c=2,
        )
        q_el = (
            einops.rearrange(q_el, "b x y z dim -> b 1 1 1 x y z dim")
            + ensemble_el_offsets
        )
        # Fold the a, b, and c axes into the x, y, z axes for coordinate transformation,
        # then unfold them for the output.
        ensemble_coords = mrinr.coords.transform_coords(
            einops.rearrange(
                q_el,
                "b lh_a lh_b lh_c x y z dim -> b (lh_a x) (lh_b y) (lh_c z) dim",
                lh_a=2,
                lh_b=2,
                lh_c=2,
            ),
            affine_x_el2coords,
            broadcast_batch=False,
        )
        ensemble_coords = einops.rearrange(
            ensemble_coords,
            "b (lh_a x) (lh_b y) (lh_c z) dim -> b lh_a lh_b lh_c x y z dim",
            lh_a=2,
            lh_b=2,
            lh_c=2,
        )
    else:
        raise ValueError(f"Invalid spatial_dims: {spatial_dims}")

    return ensemble_coords


def _sample_with_ensemble(
    spatial_dims: Literal[2, 3],
    x: mrinr.typing.Volume | mrinr.typing.Image,
    affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
    | mrinr.typing.HomogeneousAffine2D,
    x_ensemble_coords: torch.Tensor,
    padding_mode_or_bound: str,
    **resample_kwargs,
) -> torch.Tensor:
    resample_kwargs = resample_kwargs | dict(
        padding_mode_or_bound=padding_mode_or_bound
    )
    if spatial_dims == 2:
        x_ensemble = mrinr.grid_resample(
            x,
            affine_x_el2coords=affine_x_el2coords,
            sample_coords=einops.rearrange(
                x_ensemble_coords, "b lh_a lh_b x y dim -> b (lh_a x) (lh_b y) dim"
            ),
            mode_or_interpolation="nearest",
            **resample_kwargs,
        )
        x_ensemble = einops.rearrange(
            x_ensemble, "b c (lh_a x) (lh_b y) -> b c lh_a lh_b x y", lh_a=2, lh_b=2
        )
    elif spatial_dims == 3:
        x_ensemble = mrinr.grid_resample(
            x,
            affine_x_el2coords=affine_x_el2coords,
            sample_coords=einops.rearrange(
                x_ensemble_coords,
                "b lh_a lh_b lh_c x y z dim -> b (lh_a x) (lh_b y) (lh_c z) dim",
            ),
            mode_or_interpolation="nearest",
            **resample_kwargs,
        )
        x_ensemble = einops.rearrange(
            x_ensemble,
            "b c (lh_a x) (lh_b y) (lh_c z) -> b c lh_a lh_b lh_c x y z",
            lh_a=2,
            lh_b=2,
            lh_c=2,
        )
    else:
        raise ValueError(f"Invalid spatial_dims: {spatial_dims}")

    return x_ensemble


def _ensemble_delta__linear_weights(
    query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
    x_ensemble_coords: torch.Tensor,
    affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
    | mrinr.typing.HomogeneousAffine2D,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculate the delta between the query and ensemble coordinates in element space.
    # At the same time, calculate the linear weights for each ensemble coordinate.
    # Both feature maps share similar logic, so they are calculated in one function.
    spatial_dims: Literal[2, 3] = affine_x_el2coords.shape[-1] - 1
    x_edge_lengths = mrinr.coords.spacing(affine_x_el2coords)
    if spatial_dims == 2:
        delta = (
            einops.rearrange(query_coords, "b x y dim -> b 1 1 x y dim")
            - x_ensemble_coords
        )
        # Find the side length of the rectangle opposite the ensemble coordinate (a, b)
        # in unit/el space, then scale each edge by the grid size to get the real-world
        # side length.
        x_edge_lengths = einops.rearrange(
            x_edge_lengths, "b dim -> b 1 1 1 1 dim", dim=spatial_dims
        )

        q_el = einops.rearrange(
            mrinr.coords.transform_coords(
                query_coords, mrinr.coords.inv_affine(affine_x_el2coords)
            ),
            "b x y dim -> b 1 1 x y dim",
        )
        # 'lh' stands for "low-high" in that ensemble dimension.
        ab_el = einops.rearrange(
            mrinr.coords.transform_coords(
                einops.rearrange(
                    x_ensemble_coords, "b lh_a lh_b x y dim -> b (lh_a x) (lh_b y) dim"
                ),
                mrinr.coords.inv_affine(affine_x_el2coords),
            ),
            "b (lh_a x) (lh_b y) dim -> b lh_a lh_b x y dim",
            lh_a=2,
            lh_b=2,
        )
    elif spatial_dims == 3:
        delta = (
            einops.rearrange(query_coords, "b x y z dim -> b 1 1 1 x y z dim")
            - x_ensemble_coords
        )
        # Find the side length of the rectangle opposite the ensemble coordinate (a, b, c)
        # in unit/el space, then scale each edge by the grid size to get the real-world
        # side length.
        x_edge_lengths = einops.rearrange(
            x_edge_lengths, "b dim -> b 1 1 1 1 1 1 dim", dim=spatial_dims
        )

        q_el = einops.rearrange(
            mrinr.coords.transform_coords(
                query_coords, mrinr.coords.inv_affine(affine_x_el2coords)
            ),
            "b x y z dim -> b 1 1 1 x y z dim",
        )
        # 'lh' stands for "low-high" in that ensemble dimension.
        ab_el = einops.rearrange(
            mrinr.coords.transform_coords(
                einops.rearrange(
                    x_ensemble_coords,
                    "b lh_a lh_b lh_c x y z dim -> b (lh_a x) (lh_b y) (lh_c z) dim",
                ),
                mrinr.coords.inv_affine(affine_x_el2coords),
            ),
            "b (lh_a x) (lh_b y) (lh_c z) dim -> b lh_a lh_b lh_c x y z dim",
            lh_a=2,
            lh_b=2,
            lh_c=2,
        )
    else:
        raise ValueError(f"Invalid spatial_dims: {spatial_dims}")

    x_grid_area = torch.prod(x_edge_lengths, dim=-1)
    # Operations are stuffed into one step, as the intermediate tensors may be
    # (very) large during inference. The order of operations is:
    # - Get each ensemble position's side length in element space
    # - Find the side lengths of the opposite rectangle
    # - Scale the lengths by the coordinate-space grid size
    # - Calculate the area of the ensemble's rectangle in coordinate space
    # - Normalize each ensemble's area weight by the total area of the cell in
    #   coordinate space
    ensemble_linear_weights = (
        torch.prod((1 - torch.abs(q_el - ab_el)) * x_edge_lengths, dim=-1) / x_grid_area
    )

    return delta, ensemble_linear_weights


class LTEConv(_SpatialINRBase):
    _ENSEMBLE_SAMPLE_PADDING_MODE = "border"

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_spatial_features: int,
        out_features: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        k_freqs: int,
        norm: None | str | tuple[str, dict[str, Any]] | None = None,
        norm_ensemble_coords: bool = False,
        zero_final_inr_layer: bool = True,
        allow_approx_infer: bool = False,
        _coarse_z_infer_chunks: Optional[int] = None,
        _coarse_z_infer_context_edge_size: Optional[int] = None,
        **res_conv_inr_kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(locals())
        torch.nn.Module.__init__(self)
        self._allow_approx_infer = allow_approx_infer
        self._spatial_dims = spatial_dims
        self._out_features = out_features
        self._k_freqs = k_freqs
        # Store the init object for creating activation function modules.
        self._activate_fn_init_obj = activate_fn

        self.low_freq_interp = mrinr.nn.spatial_coders.InterpolationResampler(
            spatial_dims=self._spatial_dims,
            in_features=in_spatial_features,
            out_features=self._out_features,
            interp_lib=self._MLP_INTERP_SKIP_LIB,
            mode_or_interpolation=self._MLP_INTERP_SKIP_MODE,
            padding_mode_or_bound=self._MLP_INTERP_SKIP_PADDING_MODE,
        )

        self._in_spatial_features = in_spatial_features

        # self.pre_conv = None
        freq_amp_spatial_channels = self._in_spatial_features

        self._normalize_ensemble_coords = norm_ensemble_coords

        self.amplitude_layer = mrinr.nn.ConvUnit(
            spatial_dims=self.spatial_dims,
            in_channels=freq_amp_spatial_channels,
            out_channels=self._k_freqs,
            kernel_size=3,
            padding="same",
            padding_mode="border",
            activate_fn="none",
            bias=True,
        )

        self.freq_coef_layer = mrinr.nn.ConvUnit(
            spatial_dims=self.spatial_dims,
            in_channels=freq_amp_spatial_channels,
            out_channels=self.spatial_dims * self._k_freqs,
            kernel_size=3,
            padding="same",
            padding_mode="border",
            activate_fn="none",
            bias=True,
        )
        self.phase_layer = torch.nn.Linear(self.spatial_dims, self._k_freqs, bias=False)

        inr_in_channels = self._k_freqs * 2

        self.conv_inr = _ResConvINR(
            spatial_dims=self.spatial_dims,
            in_channels=inr_in_channels,
            out_channels=self.out_features,
            activate_fn=self._activate_fn_init_obj,
            norm=norm,
            end_with_conv=True,
            zero_final_conv=zero_final_inr_layer,
            **res_conv_inr_kwargs,
        )

        if (
            self._allow_approx_infer
            and self.spatial_dims == 3
            and _coarse_z_infer_chunks is not None
            and _coarse_z_infer_context_edge_size is not None
        ):
            self._use_coarse_z_infer = True
            self._coarse_z_infer_chunks = _coarse_z_infer_chunks
            self._coarse_z_infer_context_size = _coarse_z_infer_context_edge_size
        else:
            self._use_coarse_z_infer = False
            self._coarse_z_infer_chunks = None
            self._coarse_z_infer_context_size = None

    @property
    def inr_mlp(self):
        # Compatibility with the INR expected parameters
        return self.conv_inr

    @property
    def interpolate_skip(self):
        # Compatibility with the INR expected parameters
        return self.low_freq_interp

    @property
    def out_channels(self) -> int:
        return self.out_features

    def _coarse_z_chunk_forward(
        self,
        x: mrinr.typing.Volume,
        x_coords: mrinr.typing.CoordGrid3D,
        query_coords: mrinr.typing.CoordGrid3D,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine3D,
        affine_query_el2coords: mrinr.typing.HomogeneousAffine3D,
        x_coord_normalizer_params: "mrinr.nn.spatial_coders.NormalizerParams",
        query_coord_normalizer_params: "mrinr.nn.spatial_coders.NormalizerParams",
        return_coord_space: bool = False,
    ) -> Union[
        mrinr.typing.Volume,
        "mrinr.nn.spatial_coders.DenseCoordSpace",
    ]:
        # Chop up x, x_coords, and query_coords into chunks along the z axis, then
        # combine. Ratios of x chunk size to query chunk size should be similar to the
        # input shapes.
        x_size = x.shape[-1]
        q_size = query_coords.shape[-2]

        max_chunks = min(
            self._coarse_z_infer_chunks, (x_size // 2) - 1, (q_size // 2) - 1
        )
        max_chunks = max(max_chunks, 1)
        x_start_idx, x_end_idx, x_chunk_affines = _get_dim_chunk_indices_affines(
            dim_size=x_size,
            window_edge_len=self._coarse_z_infer_context_size,
            max_chunks=max_chunks,
            affine=affine_x_el2coords,
            affine_translate_dim=2,
        )
        q_start_idx, q_end_idx, q_chunk_affines = _get_dim_chunk_indices_affines(
            dim_size=q_size,
            window_edge_len=0,
            max_chunks=max_chunks,
            affine=affine_query_el2coords,
            affine_translate_dim=2,
        )

        assert len(q_start_idx) == len(x_start_idx)

        y = list()
        for (q_start, q_end, affine_q), (x_start, x_end, affine_x) in zip(
            zip(q_start_idx, q_end_idx, q_chunk_affines),
            zip(x_start_idx, x_end_idx, x_chunk_affines),
        ):
            x_chunk = x[..., x_start:x_end]
            x_coords_chunk = x_coords[..., x_start:x_end, :]
            q_coords_chunk = query_coords[..., q_start:q_end, :]
            y.append(
                self.forward(
                    x=x_chunk,
                    x_coords=x_coords_chunk,
                    query_coords=q_coords_chunk,
                    affine_x_el2coords=affine_x,
                    affine_query_el2coords=affine_q,
                    x_coord_normalizer_params=x_coord_normalizer_params,
                    query_coord_normalizer_params=query_coord_normalizer_params,
                    max_q_chunks=None,
                    return_coord_space=False,
                    allow_infer_chunking=False,
                    _forward_coarse_z_infer=False,  # Set to False to avoid infinite recursion
                )
            )

        y = torch.cat(y, dim=-1)

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=query_coords, affine=affine_query_el2coords
            )
        else:
            r = y

        return r

    def forward(
        self,
        x: mrinr.typing.Volume | mrinr.typing.Image,
        x_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        affine_query_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        x_coord_normalizer_params: "mrinr.nn.spatial_coders.NormalizerParams",
        query_coord_normalizer_params: "mrinr.nn.spatial_coders.NormalizerParams",
        *,
        max_q_chunks: Optional[int] = None,
        return_coord_space: bool = False,
        allow_infer_chunking: bool = False,
        _forward_coarse_z_infer: Optional[bool] = None,
        **kwargs,
    ) -> Union[
        mrinr.typing.Volume,
        mrinr.typing.Image,
        "mrinr.nn.spatial_coders.DenseCoordSpace",
    ]:
        if (
            _forward_coarse_z_infer is None
            and self._use_coarse_z_infer
            and allow_infer_chunking
            and not self.training
        ):
            return self._coarse_z_chunk_forward(
                x=x,
                x_coords=x_coords,
                query_coords=query_coords,
                affine_x_el2coords=affine_x_el2coords,
                affine_query_el2coords=affine_query_el2coords,
                x_coord_normalizer_params=x_coord_normalizer_params,
                query_coord_normalizer_params=query_coord_normalizer_params,
                return_coord_space=return_coord_space,
            )
        x_features = x

        # Encode x as amplitude and frequency.
        x_amplitude = self.amplitude_layer(x_features)
        x_freq_coef = self.freq_coef_layer(x_features)
        # Encode phase.
        query_grid_sizes = mrinr.coords.spacing(affine_query_el2coords)

        if self._normalize_ensemble_coords:
            x_grid_sizes = mrinr.coords.spacing(affine_x_el2coords)
            query_grid_sizes = query_grid_sizes / x_grid_sizes
            # if self.spatial_dims == 2:
            #     query_grid_sizes = self.spacing_phase_norm(
            #         einops.rearrange(query_grid_sizes, "b dim -> b 1 1 dim")
            #     ).squeeze(1, 2)
            # elif self.spatial_dims == 3:
            #     query_grid_sizes = self.spacing_phase_norm(
            #         einops.rearrange(query_grid_sizes, "b dim -> b 1 1 1 dim")
            #     ).squeeze(1, 2, 3)
        phase_q = self.phase_layer(query_grid_sizes)

        if (
            not self.training
            and allow_infer_chunking
            and not x.requires_grad
            and max_q_chunks is not None
        ):
            # If performing inference without grads, then we can split up the query
            # coordinates into smaller independent chunks to reduce peak memory usage.
            # Chunk the query coordinates, but treat it as a spatial feature map for
            # keeping track of the correct concatenation dimension (the output will be
            # a spatial feature map, not a set of coordinates).
            q, cat_dim = _get_chunked_inference_features(
                mrinr.nn.coords_as_channels(query_coords, has_batch_dim=True),
                max_chunks=max_q_chunks,
                spatial_indices=(2, 3) if self.spatial_dims == 2 else (2, 3, 4),
            )
            y = list()
            for q_chunk in q:
                y.append(
                    self._ensemble_forward(
                        x=x,
                        x_amplitude=x_amplitude,
                        x_freq_coef=x_freq_coef,
                        phase_q=phase_q,
                        query_coords=mrinr.nn.channels_as_coords(
                            q_chunk, has_batch_dim=True
                        ),
                        affine_query_el2coords=affine_query_el2coords,
                        affine_x_el2coords=affine_x_el2coords,
                    )
                )
            y = torch.cat(y, dim=cat_dim)
        # If training, just perform a forward pass as normal.
        else:
            y = self._ensemble_forward(
                x=x,
                x_amplitude=x_amplitude,
                x_freq_coef=x_freq_coef,
                phase_q=phase_q,
                query_coords=query_coords,
                affine_query_el2coords=affine_query_el2coords,
                affine_x_el2coords=affine_x_el2coords,
            )

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=query_coords, affine=affine_query_el2coords
            )
        else:
            r = y
        return r

    def _ensemble_forward(
        self,
        x: mrinr.typing.Volume | mrinr.typing.Image,
        x_amplitude: mrinr.typing.Volume | mrinr.typing.Image,
        x_freq_coef: mrinr.typing.Volume | mrinr.typing.Image,
        phase_q: torch.Tensor,
        query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        affine_query_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
    ) -> mrinr.typing.Volume | mrinr.typing.Image:
        if self.spatial_dims == 2:
            axes_lengths = {
                "b": query_coords.shape[0],
                "lh_a": 2,
                "lh_b": 2,
                "x": query_coords.shape[-3],
                "y": query_coords.shape[-2],
            }
        elif self.spatial_dims == 3:
            axes_lengths = {
                "b": query_coords.shape[0],
                "lh_a": 2,
                "lh_b": 2,
                "lh_c": 2,
                "x": query_coords.shape[-4],
                "y": query_coords.shape[-3],
                "z": query_coords.shape[-2],
            }

        # Find coordinates in x that correspond to the ensemble surrounding each point
        # in q.
        x_ensemble_coords = _ensemble_coords_in_x(query_coords, affine_x_el2coords)
        # Sample the amplitude and frequency spatial maps with the ensemble around q.
        x_amplitude = _sample_with_ensemble(
            spatial_dims=self.spatial_dims,
            x=x_amplitude,
            affine_x_el2coords=affine_x_el2coords,
            x_ensemble_coords=x_ensemble_coords,
            padding_mode_or_bound=self._ENSEMBLE_SAMPLE_PADDING_MODE,
            # The interpol nearest-neighbor sampling requires less memory than
            # pytorch grid_sample().
            interp_lib="interpol",
        )
        x_freq_coef = _sample_with_ensemble(
            spatial_dims=self.spatial_dims,
            x=x_freq_coef,
            affine_x_el2coords=affine_x_el2coords,
            x_ensemble_coords=x_ensemble_coords,
            padding_mode_or_bound=self._ENSEMBLE_SAMPLE_PADDING_MODE,
            interp_lib="interpol",
        )

        (
            x_ensemble_coord_delta,
            ensemble_linear_weights,
        ) = _ensemble_delta__linear_weights(
            query_coords=query_coords,
            x_ensemble_coords=x_ensemble_coords,
            affine_x_el2coords=affine_x_el2coords,
        )

        # Concatenate all ensemble features and merge ensemble axes into the batch.
        # *2D*
        if self.spatial_dims == 2:
            x_amplitude = einops.rearrange(
                x_amplitude,
                "b c lh_a lh_b x y -> b c (lh_a x) (lh_b y)",
                **axes_lengths,
            )
            # The frequency and the coordinate delta have an inner product on the
            # spatial dimension axis, so the frequency map is reduced from k x D to k.
            x_freq_coef = einops.rearrange(
                x_freq_coef,
                "b (dim k) lh_a lh_b x y -> b k (lh_a x) (lh_b y) dim",
                k=self._k_freqs,
                dim=self.spatial_dims,
                **axes_lengths,
            )
            # Reshape for the inner product with the frequency features.
            x_ensemble_coord_delta = einops.rearrange(
                x_ensemble_coord_delta,
                "b lh_a lh_b x y dim -> b 1 (lh_a x) (lh_b y) dim",
                **axes_lengths,
            )
            if self._normalize_ensemble_coords:
                x_spacing = mrinr.coords.spacing(affine_x_el2coords)
                x_ensemble_coord_delta = (
                    x_ensemble_coord_delta / x_spacing[:, None, None, None, :]
                )
                # x_ensemble_coord_delta = self.ensemble_coord_delta_norm(
                #     x_ensemble_coord_delta.squeeze(1)
                # ).unsqueeze(1)
            # Broadcast over spatial shapes.
            phase_q = einops.rearrange(phase_q, "b k -> b k 1 1", k=self._k_freqs)

            # Calculate ensemble features.
            # Use an inner product over spatial dimension, then squeeze the singleton
            # dimension (singletons and reshaping not yet supported by einops.einsum).
            # Dot product shown in LTE paper Figure 2.
            x_freq_coef = einops.einsum(
                x_freq_coef,
                x_ensemble_coord_delta,
                "b k ... dim, b singleton ... dim -> b singleton k ...",
            ).squeeze(1)

            # LTE paper Eq. 7.
            ensemble_feats = einops.rearrange(
                [
                    x_amplitude * torch.cos(torch.pi * (x_freq_coef + phase_q)),
                    x_amplitude * torch.sin(torch.pi * (x_freq_coef + phase_q)),
                ],
                "cos_sin b k (lh_a x) (lh_b y) -> b (cos_sin k) (lh_a x) (lh_b y)",
                cos_sin=2,
                k=self._k_freqs,
                **axes_lengths,
            )

        # *3D*
        elif self.spatial_dims == 3:
            x_amplitude = einops.rearrange(
                x_amplitude,
                "b c lh_a lh_b lh_c x y z -> b c (lh_a x) (lh_b y) (lh_c z)",
                **axes_lengths,
            )
            # The frequency and the coordinate delta have an inner product on the
            # spatial dimension axis, so the frequency map is reduced from k x D to k.
            x_freq_coef = einops.rearrange(
                x_freq_coef,
                "b (dim k) lh_a lh_b lh_c x y z -> b k (lh_a x) (lh_b y) (lh_c z) dim",
                k=self._k_freqs,
                dim=self.spatial_dims,
                **axes_lengths,
            )
            # Reshape for the inner product with the frequency features.
            x_ensemble_coord_delta = einops.rearrange(
                x_ensemble_coord_delta,
                "b lh_a lh_b lh_c x y z dim -> b 1 (lh_a x) (lh_b y) (lh_c z) dim",
                **axes_lengths,
            )
            if self._normalize_ensemble_coords:
                x_spacing = mrinr.coords.spacing(affine_x_el2coords)
                x_ensemble_coord_delta = (
                    x_ensemble_coord_delta / x_spacing[:, None, None, None, None, :]
                )
                # x_ensemble_coord_delta = self.ensemble_coord_delta_norm(
                #     x_ensemble_coord_delta.squeeze(1)
                # ).unsqueeze(1)
            # Broadcast over spatial shapes.
            phase_q = einops.rearrange(phase_q, "b k -> b k 1 1 1", k=self._k_freqs)

            # Calculate ensemble features.
            # Use an inner product over spatial dimension, then squeeze the singleton
            # dimension (singletons and reshaping not yet supported by einops.einsum).
            # Dot product shown in LTE paper Figure 2.
            x_freq_coef = einops.einsum(
                x_freq_coef,
                x_ensemble_coord_delta,
                "b k ... dim, b singleton ... dim -> b singleton k ...",
            ).squeeze(1)

            # LTE paper Eq. 7.
            ensemble_feats = einops.rearrange(
                [
                    x_amplitude * torch.cos(torch.pi * (x_freq_coef + phase_q)),
                    x_amplitude * torch.sin(torch.pi * (x_freq_coef + phase_q)),
                ],
                "cos_sin b k (lh_a x) (lh_b y) (lh_c z) -> b (cos_sin k) (lh_a x) (lh_b y) (lh_c z)",
                cos_sin=2,
                k=self._k_freqs,
                **axes_lengths,
            )

        else:
            raise ValueError(f"Invalid spatial_dims: {self.spatial_dims}")

        # If performing inference, free intermediate tensors to save memory in the event
        # of a large input.
        if not self.training:
            del x_ensemble_coords, x_ensemble_coord_delta

        # Forward pass through the conv INR.
        y = self.conv_inr(ensemble_feats)

        # Free intermediate tensors if performing inference.
        if not self.training:
            del ensemble_feats
            del x_freq_coef, x_amplitude

        # Weigh the output by linear interpolation weights.
        # Move the ensemble dimensions into the last dimension for weighting + summation.
        if self.spatial_dims == 2:
            ensemble_linear_weights = einops.rearrange(
                ensemble_linear_weights,
                "b lh_a lh_b x y -> b 1 x y (lh_a lh_b)",
                **axes_lengths,
            )
            y = einops.rearrange(
                y, "b c (lh_a x) (lh_b y) -> b c x y (lh_a lh_b)", **axes_lengths
            )
        elif self.spatial_dims == 3:
            ensemble_linear_weights = einops.rearrange(
                ensemble_linear_weights,
                "b lh_a lh_b lh_c x y z -> b 1 x y z (lh_a lh_b lh_c)",
                **axes_lengths,
            )
            y = einops.rearrange(
                y,
                "b c (lh_a x) (lh_b y) (lh_c z) -> b c x y z (lh_a lh_b lh_c)",
                **axes_lengths,
            )

        # Weigh by linear interpolation weights, then sum over each element in the
        # ensemble.
        y = torch.linalg.vecdot(y, ensemble_linear_weights, dim=-1)

        if not self.training:
            del ensemble_linear_weights

        # Add low-frequency linear interpolation of x.
        y += self.low_freq_interp(
            x=x,
            x_coords=None,
            query_coords=query_coords,
            affine_x_el2coords=affine_x_el2coords,
            affine_query_el2coords=affine_query_el2coords,
        )

        return y
