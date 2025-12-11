# -*- coding: utf-8 -*-
# Module for classes and functions that build implicit neural representations (INR).

import collections
import itertools
from functools import partial
from typing import Any, Iterator, Literal, NamedTuple, Optional, Union

import einops
import numpy as np
import pandas as pd
import torch
from jaxtyping import Integer

import mrinr

__all__ = [
    "spatial_ensemble_3d_coord_features_generator",
    "spatial_ensemble_2d_coord_features_generator",
    "pos_encoding",
    "PositionalCoordEncoding",
    "gaussian_encoding",
    "GaussianCoordEncoding",
    "GaussianCoordEncodingPreSampled",
    "EnsembleSpatialINR",
    "NeighborhoodSpatialINR",
    "LTE",
    "_ElementwiseResINRMLP",
    "_DenseResINRMLP",
    "_SpatialINRBase",
]


def pos_encoding(v: torch.Tensor, sigma_scale: float, m_num_freqs: int) -> torch.Tensor:
    """From the "Positional Encoding" method found in section 6.1 [1].

    [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.

    Parameters
    ----------
    v : torch.Tensor, shape '[batch x] coord'
    sigma_scale : float
    m_num_freqs : int

    Returns
    -------
    torch.Tensor
    """

    if v.ndim == 1:
        p = v.unsqueeze(0)
    else:
        p = v
    # 'batch x coord x 1'
    p = p.unsqueeze(-1)

    coeffs = (
        2
        * torch.pi
        * sigma_scale
        ** (torch.arange(0, m_num_freqs, device=p.device, dtype=p.dtype) / m_num_freqs)
    )

    # 'batch x coord x m_freqs'
    theta = coeffs * p

    # Calculate cosine and sine components, reshape to give to an MLP as
    # `b x in_features`.
    y = einops.rearrange(
        [torch.cos(theta), torch.sin(theta)], "cos_sin b coord m -> b (m cos_sin coord)"
    )
    # Reshape to be compatible with the `positional_encoding` function.
    # y = einops.rearrange(
    #     [torch.cos(theta), torch.sin(theta)], "cos_sin b coord m -> b (coord m cos_sin)"
    # )

    if v.ndim == 1:
        y.squeeze_(0)

    return y


class PositionalCoordEncoding(torch.nn.Module):
    def __init__(
        self, spatial_dims: Literal[2, 3], sigma_scale: float, m_num_freqs: int
    ):
        """From the "Positional Encoding" method found in section 6.1 [1].

        [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.
        """
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self._spatial_dims = int(spatial_dims)
        self.sigma_scale = float(sigma_scale)
        self.m_num_freqs = int(m_num_freqs)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def extra_repr(self) -> str:
        return ", ".join(
            [
                str(self.in_features),
                str(self.out_features),
                f"spatial_dims={self._spatial_dims}",
                f"sigma_scale={self.sigma_scale}",
                f"m_num_freqs={self.m_num_freqs}",
            ]
        )

    @property
    def in_features(self):
        return self._spatial_dims

    @property
    def out_features(self):
        return self.m_num_freqs * self._spatial_dims * 2

    def forward(
        self, x: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D
    ) -> mrinr.typing.Image | mrinr.typing.Volume:
        """Encode input coordinate vector v.

        Parameters
        ----------
        v : torch.Tensor
            Input coordinate grid.

        Returns
        -------
        torch.Tensor
            Encoded coordinate features.
        """
        # Reshape to '[batch x] coord'
        if self._spatial_dims == 2:
            v = einops.rearrange(x, "b x y coord-> (b x y) coord")
        elif self._spatial_dims == 3:
            v = einops.rearrange(x, "b x y z coord -> (b x y z) coord")
        else:
            raise ValueError(
                f"Invalid spatial dimensions '{self._spatial_dims}'. Must be 2 or 3."
            )

        y = pos_encoding(v, sigma_scale=self.sigma_scale, m_num_freqs=self.m_num_freqs)
        # Reshape into spatial data layout 'batch x features x X x Y [x Z]', flattening
        # the 'spatial_dims x m x 2' dimensions.
        # Some shapes are not necessary to specify, but doing so clarifies the
        # dimensions and does a sanity-check on sizes.
        if self._spatial_dims == 2:
            y = einops.rearrange(
                y,
                "(b x y) (m sin_cos coord) -> b (m sin_cos coord) x y",
                b=x.shape[0],
                x=x.shape[1],
                y=x.shape[2],
                coord=self._spatial_dims,
                m=self.m_num_freqs,
                sin_cos=2,
            )
        elif self._spatial_dims == 3:
            y = einops.rearrange(
                y,
                "(b x y z) (m sin_cos coord) -> b (m sin_cos coord) x y z",
                b=x.shape[0],
                x=x.shape[1],
                y=x.shape[2],
                z=x.shape[3],
                coord=self._spatial_dims,
                m=self.m_num_freqs,
                sin_cos=2,
            )

        return y


def gaussian_encoding(
    v: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    m_num_freqs: int,
    rng: torch.Generator,
    *,
    fork_rng: bool = True,
    return_rng: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Generator]:
    """From the "Gaussian" method found in section 6.1 [1].

    [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.

    Parameters
    ----------
    v : torch.Tensor, shape '[batch x] coord'
    sigma : float or Tensor
    m_num_freqs : int
    rng : Generator

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Generator]
    """

    # 'batch x coord'
    if v.ndim == 1:
        p = v.unsqueeze(0)
    else:
        p = v

    rng_ = mrinr.utils.fork_rng(rng) if fork_rng else rng

    d = p.shape[1]
    batch_size = p.shape[0]
    B = torch.normal(
        mean=0,
        std=sigma,
        size=(batch_size, d, m_num_freqs),
        dtype=p.dtype,
        device=p.device,
        generator=rng_,
    )
    Bp = einops.einsum(B, p, "... i j, ... i -> ... j")
    cos_mapping = torch.cos(2 * torch.pi * Bp)
    sin_mapping = torch.sin(2 * torch.pi * Bp)
    # batch x spatial_dims x m_num_freqs x 2
    y = einops.rearrange([cos_mapping, sin_mapping], "cos_sin b m -> b (m cos_sin)")

    if v.ndim == 1:
        y.squeeze_(0)
    if return_rng:
        r = (y, rng_)
    else:
        r = y
    return r


def _pre_sampled_gaussian_encoding(
    v: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """From the "Gaussian" method found in section 6.1 [1].

    [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.

    Parameters
    ----------
    v : torch.Tensor, shape '[batch x] coord'
    B : torch.Tensor, shape '[batch x] coord x m_num_freqs'

    Returns
    -------
    torch.Tensor
    """

    # 'batch x coord'
    if v.ndim == 1:
        p = v.unsqueeze(0)
    else:
        p = v
    if B.ndim == 2:
        B = B.unsqueeze(0)

    Bp = einops.einsum(B, p, "... i j, ... i -> ... j")
    cos_mapping = torch.cos(2 * torch.pi * Bp)
    sin_mapping = torch.sin(2 * torch.pi * Bp)
    # batch x spatial_dims x m_num_freqs x 2
    y = einops.rearrange([cos_mapping, sin_mapping], "cos_sin b m -> b (m cos_sin)")

    if v.ndim == 1:
        y.squeeze_(0)
    return y


class GaussianCoordEncoding(torch.nn.Module):
    def __init__(self, spatial_dims: Literal[2, 3], sigma: float, m_num_freqs: int):
        """From the "Gaussian" method found in section 6.1 [1].

        [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.
        """
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self._spatial_dims = int(spatial_dims)
        self.sigma = float(sigma)
        self.m_num_freqs = int(m_num_freqs)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def extra_repr(self) -> str:
        return ", ".join(
            [
                str(self.in_features),
                str(self.out_features),
                f"spatial_dims={self._spatial_dims}",
                f"sigma={self.sigma}",
                f"m_num_freqs={self.m_num_freqs}",
            ]
        )

    @property
    def in_features(self):
        return self._spatial_dims

    @property
    def out_features(self):
        return self.m_num_freqs * 2

    def forward(
        self,
        x: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        rng: Literal["default"] | torch.Generator = "default",
        *,
        fork_rng: bool = False,
        return_rng: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Generator]:
        if rng == "default":
            if x.device.type == "cpu":
                rng_ = torch.default_generator
            elif x.device.type == "cuda":
                rng_ = None
                # rng_ = torch.cuda.default_generators[x.device.index]
            else:
                raise ValueError(f"Unsupported device type '{x.device.type}'.")
        else:
            rng_ = rng
        # Reshape to '[batch x] coord'
        if self._spatial_dims == 2:
            v = einops.rearrange(x, "b x y coord-> (b x y) coord")
        elif self._spatial_dims == 3:
            v = einops.rearrange(x, "b x y z coord -> (b x y z) coord")
        else:
            raise ValueError(
                f"Invalid spatial dimensions '{self._spatial_dims}'. Must be 2 or 3."
            )

        y = gaussian_encoding(
            v,
            sigma=self.sigma,
            m_num_freqs=self.m_num_freqs,
            rng=rng_,
            fork_rng=fork_rng,
            return_rng=return_rng,
        )
        if return_rng:
            y, rng_ = y

        # Reshape into spatial data layout 'batch x features x X x Y [x Z]', flattening
        # the 'spatial_dims x m x 2' dimensions.
        # Some shapes are not necessary to specify, but doing so clarifies the
        # dimensions and does a sanity-check on sizes.
        if self._spatial_dims == 2:
            y = einops.rearrange(
                y,
                "(b x y) (m sin_cos) -> b (m sin_cos) x y",
                b=x.shape[0],
                x=x.shape[1],
                y=x.shape[2],
                m=self.m_num_freqs,
                sin_cos=2,
            )
        elif self._spatial_dims == 3:
            y = einops.rearrange(
                y,
                "(b x y z) (m sin_cos) -> b (m sin_cos) x y z",
                b=x.shape[0],
                x=x.shape[1],
                y=x.shape[2],
                z=x.shape[3],
                m=self.m_num_freqs,
                sin_cos=2,
            )

        if return_rng:
            r = (y, rng_)
        else:
            r = y
        return r


class GaussianCoordEncodingPreSampled(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        sigma: float,
        m_num_freqs: int,
        rng: Literal["default"] | torch.Generator = "default",
    ):
        """From the "Gaussian" method found in section 6.1 [1].

        [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.
        """
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self._spatial_dims = int(spatial_dims)
        self.sigma = float(sigma)
        self.m_num_freqs = int(m_num_freqs)
        if rng == "default":
            rng_ = torch.default_generator
        else:
            rng_ = rng

        B = torch.normal(
            mean=0,
            std=sigma,
            size=(self._spatial_dims, self.m_num_freqs),
            generator=rng_,
        )
        self.register_buffer("B", B)
        self.B: torch.Tensor

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def extra_repr(self) -> str:
        return ", ".join(
            [
                str(self.in_features),
                str(self.out_features),
                f"spatial_dims={self._spatial_dims}",
                f"sigma={self.sigma}",
                f"m_num_freqs={self.m_num_freqs}",
            ]
        )

    @property
    def in_features(self):
        return self._spatial_dims

    @property
    def out_features(self):
        return self.m_num_freqs * 2

    def forward(
        self,
        x: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Generator]:
        # Reshape to '[batch x] coord'
        if self._spatial_dims == 2:
            v = einops.rearrange(x, "b x y coord-> (b x y) coord")
        elif self._spatial_dims == 3:
            v = einops.rearrange(x, "b x y z coord -> (b x y z) coord")
        else:
            raise ValueError(
                f"Invalid spatial dimensions '{self._spatial_dims}'. Must be 2 or 3."
            )

        y = _pre_sampled_gaussian_encoding(
            v, self.B
        )  # self.B.expand(v.shape[0], -1, -1))

        # Reshape into spatial data layout 'batch x features x X x Y [x Z]', flattening
        # the 'spatial_dims x m x 2' dimensions.
        # Some shapes are not necessary to specify, but doing so clarifies the
        # dimensions and does a sanity-check on sizes.
        if self._spatial_dims == 2:
            y = einops.rearrange(
                y,
                "(b x y) (m sin_cos) -> b (m sin_cos) x y",
                b=x.shape[0],
                x=x.shape[1],
                y=x.shape[2],
                m=self.m_num_freqs,
                sin_cos=2,
            )
        elif self._spatial_dims == 3:
            y = einops.rearrange(
                y,
                "(b x y z) (m sin_cos) -> b (m sin_cos) x y z",
                b=x.shape[0],
                x=x.shape[1],
                y=x.shape[2],
                z=x.shape[3],
                m=self.m_num_freqs,
                sin_cos=2,
            )

        return y


class _SpatialEnsembleElement(NamedTuple):
    """Spatially-ordered ensemble features for 2D/3D LIIF-like generators.

    *Note*: There are 3 different spaces involved in calculating the ensemble features:

    1. Coordinate/real space - defined by the associated affine transformation matrix.
    Points in this space are denoted `x, y` in 2D and `x, y, z` in 3D.

    2. Pixel/voxel space of the input - integers that can be used to index into the
    input volume(s). Points in this space are denoted 'i, j' in 2D and `i, j, k` in 3D.

    3. Local ensemble space - space surrounding the query point bounded by its 2x2[x2]
    nearest neighbors. Points in this space are denoted `a, b` in 2D (or
    `a, b, c` in 3D), and all points must be in the range [0, 1]. The corner points
        $a, b \\in {0, 1} x {0, 1}$
        or
        $a, b, c \\in {0, 1} x {0, 1} x {0, 1}$
    are points/pixels/voxels in the input pixel/voxel space, so the only (possibly)
    non-integer coordinate in this space should be the query point itself.
    """

    input_el_idx: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D
    diff_ensemble_corner_to_ensemble_q: (
        mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D
    )
    spatial_weight: mrinr.typing.ScalarImage | mrinr.typing.ScalarVolume
    ensemble_corner: Integer[torch.Tensor, "coords"]  # noqa: F821
    ensemble_corner_label: str


def spatial_ensemble_2d_coord_features_generator(
    query_coords: mrinr.typing.CoordGrid2D,
    affine_pixel2coords: mrinr.typing.HomogeneousAffine2D,
) -> Iterator[_SpatialEnsembleElement]:
    # raise NotImplementedError()
    ####### matrix inverse and muliplication implementation for q_vox
    # Slower and less precise than linalg.solve, but does not have the matrix size bug.
    q_pixel = (
        mrinr.coords.transform_coords(
            query_coords.double(), mrinr.coords.inv_affine(affine_pixel2coords.double())
        )
        .round(decimals=6)
        .to(query_coords)
    )

    # Get the (a=0, b=0) pixel coordinate of the ensemble.
    q_pixel_bottom = q_pixel.floor()

    # Coordinates of the query in its own local coordinate system.
    q_local_ensemble = q_pixel - q_pixel_bottom.to(q_pixel)

    # Build the ensemble one voxel index at a time.
    # Each ensemble is a [0, 1] voxel coordinate system local to the query point,
    # where the origin is the input voxel that is "lower" in all dimensions
    # than the query coordinate (i.e., pixel a=0, b=0, in ensemble
    # coordinates).
    ab_ensemble = q_pixel_bottom.new_zeros(1, 1, 1, 2)
    for (
        a_ensemble,
        b_ensemble,
    ) in itertools.product((0, 1), (0, 1)):
        # Rebuild indexing tuple for each element of the sub-window
        ab_ensemble[..., 0] = a_ensemble
        ab_ensemble[..., 1] = b_ensemble
        input_index_ij = q_pixel_bottom + ab_ensemble

        # delta between the current abc and the query, in local ensemble
        # coordinates. Will be in range [-1, 1].
        diff_local_ensemble_to_q = ab_ensemble - q_local_ensemble

        # Find the volume of the opposing square corner (1-a, 1-b) to find the
        # output weighting.
        w_ab = einops.reduce(
            1 - torch.abs(diff_local_ensemble_to_q),
            "b x y coord -> b 1 x y",
            reduction="prod",
        )

        yield _SpatialEnsembleElement(
            input_el_idx=input_index_ij,
            diff_ensemble_corner_to_ensemble_q=diff_local_ensemble_to_q,
            spatial_weight=w_ab,
            ensemble_corner=ab_ensemble.flatten(),
            ensemble_corner_label=f"{a_ensemble}{b_ensemble}",
        )


def spatial_ensemble_3d_coord_features_generator(
    query_coords: mrinr.typing.CoordGrid3D,
    affine_vox2coords: mrinr.typing.HomogeneousAffine3D,
) -> Iterator[_SpatialEnsembleElement]:
    ####### matrix inverse and muliplication implementation for q_vox
    # Slower and less precise than linalg.solve, but does not have the matrix size bug.
    q_vox = (
        mrinr.coords.transform_coords(
            query_coords.double(), mrinr.coords.inv_affine(affine_vox2coords.double())
        )
        .round(decimals=6)
        .to(query_coords)
    )

    # Get the (a=0, b=0, c=0) vox coordinate of the ensemble.
    q_vox_bottom = q_vox.floor()

    # Coordinates of the query in its own local coordinate system.
    q_local_ensemble = q_vox - q_vox_bottom.to(q_vox)

    # Build the ensemble one voxel index at a time.
    # Each ensemble is a [0, 1] voxel coordinate system local to the query point,
    # where the origin is the input voxel that is "lower" in all dimensions
    # than the query coordinate (i.e., voxel a=0, b=0, c=0, in ensemble
    # coordinates).
    abc_ensemble = q_vox_bottom.new_zeros(1, 1, 1, 1, 3)
    for (
        a_ensemble,
        b_ensemble,
        c_ensemble,
    ) in itertools.product((0, 1), (0, 1), (0, 1)):
        # Rebuild indexing tuple for each element of the sub-window
        abc_ensemble[..., 0] = a_ensemble
        abc_ensemble[..., 1] = b_ensemble
        abc_ensemble[..., 2] = c_ensemble
        input_vox_index_ijk = q_vox_bottom + abc_ensemble

        # delta between the current abc and the query, in local ensemble
        # coordinates. Will be in range [-1, 1].
        diff_local_ensemble_to_q = abc_ensemble - q_local_ensemble

        # Find the volume of the opposing cube corner (1-a, 1-b, 1-c) to find the
        # output weighting.
        w_abc = einops.reduce(
            1 - torch.abs(diff_local_ensemble_to_q),
            "b x y z coord -> b 1 x y z",
            reduction="prod",
        )

        yield _SpatialEnsembleElement(
            input_el_idx=input_vox_index_ijk,
            diff_ensemble_corner_to_ensemble_q=diff_local_ensemble_to_q,
            spatial_weight=w_abc,
            ensemble_corner=abc_ensemble.flatten(),
            ensemble_corner_label=f"{a_ensemble}{b_ensemble}{c_ensemble}",
        )


@torch.no_grad
def init_layer_glorot_bengio_uniform_(
    layer: torch.nn.Linear | torch.nn.Conv1d, gain=1.0, generator=None
) -> None:
    # Similar to xavier uniform, but inits a layer instead of a Tensor.
    torch.nn.init.xavier_uniform_(layer.weight, gain=gain, generator=generator)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


class _DenseMLPInputCombiner(torch.nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.combiner = torch.nn.Conv1d(
            in_channels=n_inputs, out_channels=1, kernel_size=1
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # Stack inputs as channels.
        y = torch.stack(inputs, dim=1)
        y = self.combiner(y)
        return y.squeeze(1)


class _DenseResINRMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        internal_layer_features: list[int],
        activate_fn: str | tuple[str, dict[str, Any]],
        res_connections_out_layer2in_layer_idx: Optional[
            dict[int, tuple[int] | int | Literal["dense"]]
        ] = None,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self._activate_fn_init_obj = activate_fn
        # Never allow inplace activations due to dense residual connections.
        activate_fn_init_callable = partial(
            mrinr.nn.make_activate_fn_module,
            init_obj=self._activate_fn_init_obj,
            inplace=False,
        )

        if isinstance(activate_fn_init_callable(), mrinr.nn.Sine):
            raise NotImplementedError("SIREN layers not implemented.")

        self._internal_layer_features = list(internal_layer_features)
        self._n_blocks = len(self._internal_layer_features) + 1

        # Map out residual connection scheme.
        layer_mappings = list()
        if res_connections_out_layer2in_layer_idx is None:
            res_connections_out_layer2in_layer_idx = dict()
        # Check for "dense" option. This will assume that *all* layers have compatible
        # feature shapes!
        elif isinstance(res_connections_out_layer2in_layer_idx, str):
            m = str(res_connections_out_layer2in_layer_idx).lower().strip()
            if "dense" in m:
                res_connections_out_layer2in_layer_idx = dict()
                for i_layer in range(self._n_blocks - 1):
                    # Assign each layer output to a tuple of all subsequent layers.
                    res_connections_out_layer2in_layer_idx[i_layer] = tuple(
                        range(i_layer + 2, self._n_blocks)
                    )
            else:
                raise ValueError(f"Invalid residual connection scheme '{m}'. ")
        # Populate skip connections.
        for i_layer, j_layer in res_connections_out_layer2in_layer_idx.items():
            if isinstance(j_layer, (tuple, list)):
                for k_layer in tuple(j_layer):
                    # Pass on the non-skip connections, as they will always be present.
                    if k_layer == i_layer + 1:
                        continue
                    layer_mappings.append((int(i_layer), int(k_layer)))
            else:
                if j_layer == i_layer + 1:
                    continue
                layer_mappings.append((int(i_layer), int(j_layer)))

        # Residual connections should be in the form of a DAG with no self-loops.
        self._res_connections = pd.DataFrame(
            layer_mappings, columns=["from_layer_idx_output", "to_layer_idx_input"]
        )
        self._res_connections = self._res_connections.sort_values(
            "from_layer_idx_output"
        ).reset_index(drop=True)
        assert (self._res_connections.from_layer_idx_output >= 0).all()
        assert (self._res_connections.from_layer_idx_output < self._n_blocks - 1).all()
        # A dense connection cannot be made between subsequent layers, as the output
        # will already be going there.
        assert (
            self._res_connections.from_layer_idx_output
            < (self._res_connections.to_layer_idx_input - 1)
        ).all()
        assert (self._res_connections.to_layer_idx_input < self._n_blocks).all()

        # Construct linear layer blocks.
        res_linear_blocks = list()
        for i, (in_f, out_f) in enumerate(
            itertools.pairwise(
                [self.in_features] + self._internal_layer_features + [self.out_features]
            )
        ):
            block_i = collections.OrderedDict()
            # If there are any residual connections going into layer i, then add a
            # combiner layer.
            if (self._res_connections.to_layer_idx_input == i).any():
                # Ensure all dense connections going into layer i have the same
                # number of features.
                if not all(
                    map(
                        lambda f: f == in_f,
                        [
                            self._internal_layer_features[a]
                            for a in (
                                self._res_connections[
                                    self._res_connections.to_layer_idx_input == i
                                ]
                            ).from_layer_idx_output
                        ],
                    )
                ):
                    raise ValueError(
                        f"All dense connections into layer {i} must have the same "
                        + "number of output features. "
                        + f"Expected all features sizes to be {in_f}, got ["
                        + ", ".join(
                            [
                                str(self._internal_layer_features[a])
                                for a in (
                                    self._res_connections[
                                        self._res_connections.to_layer_idx_input == i
                                    ]
                                ).from_layer_idx_output
                            ]
                            + [str(in_f)]
                        )
                        + "]"
                    )
                combiner_name = "combiner_from__" + "_".join(
                    [
                        str(s)
                        for s in self._res_connections[
                            self._res_connections.to_layer_idx_input == i
                        ].from_layer_idx_output
                    ]
                    + [str(i - 1)]
                )
                block_i[combiner_name] = _DenseMLPInputCombiner(
                    n_inputs=(self._res_connections.to_layer_idx_input == i).sum() + 1
                )
                block_i["combiner_activate_fn"] = activate_fn_init_callable()
            block_i["linear"] = torch.nn.Linear(in_f, out_f)
            if i < self._n_blocks - 1:
                block_i["activate_fn"] = activate_fn_init_callable()

            res_linear_blocks.append(torch.nn.Sequential(block_i))
        self.linear_blocks = torch.nn.ModuleList(res_linear_blocks)

        if isinstance(activate_fn_init_callable(), mrinr.nn.Gaussian):
            # Initialize layers with the Xavier Uniform initialization shown to be
            # effective in Ramasinghe and Lucey, 2022.
            gain = 1.0
            for m in self.linear_blocks.modules():
                if isinstance(
                    m,
                    (torch.nn.Linear, torch.nn.Conv1d),
                ):
                    init_layer_glorot_bengio_uniform_(m, gain=gain)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        tmp_block_outputs = dict()
        for i, block in enumerate(self.linear_blocks):
            # If this block needs to combine any inputs, then pull those from the tmp
            # outputs storage.
            if (self._res_connections.to_layer_idx_input == i).any():
                combine_from_idx = sorted(
                    self._res_connections[
                        self._res_connections.to_layer_idx_input == i
                    ].from_layer_idx_output.tolist()
                )
                # Combine designated residual inputs plus the output from the previous
                # block.
                block_input = [tmp_block_outputs[i] for i in combine_from_idx] + [y]
            else:
                block_input = y

            y = block(block_input)

            # If block i is to be used as an input to another block, store its output.
            if (self._res_connections.from_layer_idx_output == i).any():
                tmp_block_outputs[i] = y

        return y


class _ElementwiseResMLPBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        subunits: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        elementwise_op: Literal["add", "mul"] = "add",
        last_subunit_lin_only: bool = False,
        end_activate: bool = True,
        bias: bool = True,
        init_bias_to_zero: bool = True,
    ):
        super().__init__()

        self.features = int(features)
        self.in_features = self.features
        self.out_features = self.features
        self._activate_fn_init_obj = activate_fn
        self.elementwise_op = str(elementwise_op).strip().lower()
        if "add" in self.elementwise_op:
            self.elementwise_op = "add"
        elif "mul" in self.elementwise_op:
            self.elementwise_op = "mul"

        lins = list()
        for i in range(subunits - 1):
            l = torch.nn.Linear(features, features, bias=bias)
            if init_bias_to_zero:
                l.bias.data.zero_()
            lins.append(l)
            lins.append(
                mrinr.nn.make_activate_fn_module(
                    self._activate_fn_init_obj, try_inplace=True
                )
            )
        # Final layer.
        l = torch.nn.Linear(features, features, bias=bias)
        if init_bias_to_zero:
            l.bias.data.zero_()
        lins.append(l)
        if not last_subunit_lin_only:
            lins.append(
                mrinr.nn.make_activate_fn_module(
                    self._activate_fn_init_obj, try_inplace=True
                )
            )

        self.lins = torch.nn.Sequential(*lins)

        if end_activate:
            self.activate_fn = mrinr.nn.make_activate_fn_module(
                self._activate_fn_init_obj, try_inplace=True
            )
        else:
            self.activate_fn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lins(x)
        if self.elementwise_op == "add":
            y = y + x
        elif self.elementwise_op == "mul":
            y = y * x

        if self.activate_fn is not None:
            y = self.activate_fn(y)

        return y


class _ElementwiseResINRMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        internal_features: int,
        out_features: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        n_blocks: int,
        subunits_per_block: int,
        elementwise_op: Literal["add", "mul"] = "add",
        bias: bool = True,
        end_with_lin: bool = True,
        init_bias_to_zero: bool = True,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()

        self.in_features = in_features
        self._internal_features = internal_features
        self.out_features = out_features
        self._activate_fn_init_obj = activate_fn

        pre_lin = torch.nn.Linear(self.in_features, self._internal_features, bias=bias)
        pre_activate_fn = mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj)
        if isinstance(pre_activate_fn, mrinr.nn.Sine):
            raise NotImplementedError("SIREN layers not implemented.")

        lin_blocks = [pre_lin, pre_activate_fn]
        for i in range(n_blocks):
            lin_blocks.append(
                _ElementwiseResMLPBlock(
                    features=self._internal_features,
                    subunits=subunits_per_block,
                    activate_fn=self._activate_fn_init_obj,
                    elementwise_op=elementwise_op,
                    end_activate=True,
                    last_subunit_lin_only=False,
                    bias=bias,
                    init_bias_to_zero=init_bias_to_zero,
                )
            )

        if end_with_lin:
            post_lin = torch.nn.Linear(
                self._internal_features, self.out_features, bias=bias
            )
            lin_blocks.append(post_lin)
        else:
            post_lin = None
        self.linear_blocks = torch.nn.Sequential(*lin_blocks)
        # Select MLP weight initialization scheme.
        for m in self.linear_blocks.modules():
            if isinstance(
                m,
                (torch.nn.Linear, torch.nn.Conv1d),
            ):
                if isinstance(pre_activate_fn, mrinr.nn.Gaussian):
                    # Initialize layers with the Xavier Uniform initialization shown to
                    # be effective in Ramasinghe and Lucey, 2022.
                    gain = 1.0
                    init_layer_glorot_bengio_uniform_(m, gain=gain)
                elif init_bias_to_zero:
                    m.bias.data.zero_()

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_blocks(x)


class _SpatialINRBase(torch.nn.Module):
    _MLP_INTERP_SKIP_MODE = "linear"
    _MLP_INTERP_SKIP_PADDING_MODE = "border"
    _MLP_INTERP_SKIP_LIB = "torch"

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_spatial_features: int,
        out_features: int,
        interpolate_skip_connect: bool,
        activate_fn: str | tuple[str, dict[str, Any]],
        *,
        use_norm_input_coords: bool,
        use_input_grid_sizes: bool,
        use_query_grid_sizes: bool,
        coord_encoding: Optional[Literal["positional", "gaussian"]],
        use_encode_input_coords: bool,
        use_encode_query_coords: bool,
        coord_encoding_kwargs: dict[str, Any] = dict(),
        interpolate_skip_combine_method: Optional[
            Literal["concat", "add", "mul"]
        ] = "add",
        res_mlp_combine_method: Optional[Literal["concat", "add", "mul"]] = None,
        **mlp_kwargs,
    ):
        raise NotImplementedError("Subclasses must implement this method.")
        # Subclasses should call some form of these methods in their __init__:

        # self._init_kwargs = mrinr.nn.get_module_init_kwargs(
        #     locals(), warn=True
        # )
        # super().__init__()
        # self._set_common_properties(**self._init_kwargs)

    def _set_common_properties(
        self,
        *,
        spatial_dims,
        in_spatial_features,
        out_features,
        activate_fn,
        interpolate_skip_connect=None,
        interpolate_skip_combine_method=None,
        res_mlp_combine_method=None,
        use_norm_input_coords=None,
        use_norm_query_coords: bool = True,
        use_input_grid_sizes=None,
        use_query_grid_sizes=None,
        coord_encoding=None,
        coord_encoding_kwargs=None,
        use_encode_input_coords=None,
        use_encode_query_coords=None,
        **kwargs,
    ):
        # Network shape parameters.
        self._spatial_dims: int = spatial_dims
        self._in_spatial_features: int = in_spatial_features
        self._out_features: int = out_features
        self._interpolate_skip_connect: bool = bool(interpolate_skip_connect)
        # Input feature flags/parameters.
        self._use_norm_input_coords: bool = bool(use_norm_input_coords)
        # Always use the query coordinates as an input feature. Otherwise, how would
        # the network know what to predict?
        self._use_norm_query_coords: bool = bool(use_norm_query_coords)
        self._use_input_grid_sizes: bool = bool(use_input_grid_sizes)
        self._use_query_grid_sizes: bool = bool(use_query_grid_sizes)
        # Coordinate encoding.
        self._coord_encoding = (
            coord_encoding
            if coord_encoding is None
            else str(coord_encoding).strip().lower()
        )
        if self._coord_encoding not in {None, "positional", "gaussian"}:
            raise ValueError(f"Invalid coord encoding {self._coord_encoding}")
        self._coord_encoding_kwargs = coord_encoding_kwargs
        if self._coord_encoding is None:
            self.coord_encoder = None
        elif self._coord_encoding == "positional":
            self.coord_encoder = PositionalCoordEncoding(
                spatial_dims=self._spatial_dims, **self._coord_encoding_kwargs
            )
        elif self._coord_encoding == "gaussian":
            self.coord_encoder = GaussianCoordEncoding(
                spatial_dims=self._spatial_dims, **self._coord_encoding_kwargs
            )
        self._use_encode_input_coords: bool = bool(use_encode_input_coords)
        self._use_encode_query_coords: bool = bool(use_encode_query_coords)
        if self.coord_encoder is None and (
            self._use_encode_input_coords or self._use_encode_query_coords
        ):
            raise ValueError(
                "Cannot encode coordinates without a coordinate encoding method."
            )

        # Store the init object for creating activation function modules.
        self._activate_fn_init_obj = activate_fn

        # Determine the residual MLP connection method.
        if res_mlp_combine_method is not None:
            m = str(interpolate_skip_combine_method).lower().strip()
            if m in {"concat", "cat"} or "conv" in m:
                self._res_mlp_combine_method = "concat"
            elif m in {"add", "sum", "addition"}:
                self._res_mlp_combine_method = "add"
            elif m in {"mul", "mult", "multiply"}:
                self._res_mlp_combine_method = "mul"
            else:
                raise ValueError("Invalid Residual MLP combine method " + f"{m}")
        else:
            self._res_mlp_combine_method = None

        # Interpolation skip connection.
        if self._interpolate_skip_connect:
            self.interpolate_skip = mrinr.nn.spatial_coders.InterpolationResampler(
                spatial_dims=self._spatial_dims,
                in_features=self._in_spatial_features,
                out_features=self._out_features,
                interp_lib=self._MLP_INTERP_SKIP_LIB,
                mode_or_interpolation=self._MLP_INTERP_SKIP_MODE,
                padding_mode_or_bound=self._MLP_INTERP_SKIP_PADDING_MODE,
            )
            # Determine the method for merging the interpolation skip connection with
            # the MLP output.
            if interpolate_skip_combine_method is None:
                raise ValueError(
                    "Interpolation skip connection method must be indicated."
                )
            self._interpolate_skip_combine_method = (
                str(interpolate_skip_combine_method).lower().strip()
            )
            if (
                self._interpolate_skip_combine_method in {"concat", "cat"}
                or "conv" in self._interpolate_skip_combine_method
            ):
                self._interpolate_skip_combine_method = "concat"
                if self._spatial_dims == 2:
                    self.merge_interpolate_skip_mlp_conv = torch.nn.Conv2d(
                        in_channels=self._out_features
                        + self.interpolate_skip.out_features,
                        out_channels=self._out_features,
                        kernel_size=1,
                    )
                elif self._spatial_dims == 3:
                    self.merge_interpolate_skip_mlp_conv = torch.nn.Conv3d(
                        in_channels=self._out_features
                        + self.interpolate_skip.out_features,
                        out_channels=self._out_features,
                        kernel_size=1,
                    )
                # Have an identity transform for the interpolation output at
                # initialization, and 0 to the MLP output.
                self.merge_interpolate_skip_mlp_conv.weight.data[
                    :, : self._out_features
                ].zero_()
                # Create identity matrix in the second half of the weight matrix.
                I = torch.eye(self._out_features).to(
                    self.merge_interpolate_skip_mlp_conv.weight.data
                )
                # Reshape to fit the kernel size.
                I = I.reshape(
                    *self.merge_interpolate_skip_mlp_conv.weight.data[
                        :, : self._out_features
                    ].shape
                )
                self.merge_interpolate_skip_mlp_conv.weight.data[
                    :, self._out_features :
                ] = I
                # Zero out bias vector for merge conv layer.
                self.merge_interpolate_skip_mlp_conv.bias.data.zero_()
            elif self._interpolate_skip_combine_method in {"add", "sum", "addition"}:
                self._interpolate_skip_combine_method = "add"
            elif self._interpolate_skip_combine_method in {"mul", "mult", "multiply"}:
                self._interpolate_skip_combine_method = "mul"
            else:
                raise ValueError(
                    "Invalid interpolation skip connection method "
                    + f"{interpolate_skip_combine_method}"
                )
        else:
            self.interpolate_skip = None
            self.merge_interpolate_skip_mlp_conv = None
            self._interpolate_skip_combine_method = None
        self._mlp_out_weight = 1.0
        # Cached feature ordering, with potential feature splits for multi-branch
        # networks.
        self._feature_order: tuple[str, ...] | tuple[tuple[str, ...], ...] | None
        self._feature_sizes: tuple[int, ...] | tuple[tuple[int, ...], ...] | None
        self._inr_feature_names: set | None
        # Only set to None if they were not defined in the subclass.
        for a in {"_feature_order", "_feature_sizes", "_inr_feature_names"}:
            try:
                getattr(self, a)
            except AttributeError:
                setattr(self, a, None)

    def _get_feature_order_size(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        raise NotImplementedError("Subclasses must implement this method.")

    def reorder_and_split_features(
        self, cat_dim: int = 1, **feat_names_to_tensors
    ) -> torch.Tensor | tuple[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    @property
    def spatial_dims(self) -> Literal[2, 3]:
        return self._spatial_dims

    @property
    def out_features(self) -> int:
        return self._out_features

    # def disable_mlp_output(self):
    #     if not self._interpolate_skip_connect:
    #         raise ValueError("Cannot disable MLP output without an interpolation skip.")
    #     self._mlp_out_weight = 0.0

    # def enable_mlp_output(self):
    #     self._mlp_out_weight = 1.0

    def rearrange_coords_as_channels(self, coords: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2:
            r = einops.rearrange(
                coords, "b x y coord -> b coord x y", coord=self.spatial_dims
            )
        elif self.spatial_dims == 3:
            r = einops.rearrange(
                coords, "b x y z coord -> b coord x y z", coord=self.spatial_dims
            )

        return r

    def _apply_interpolate_skip_connect(
        self,
        x: mrinr.typing.Volume | mrinr.typing.Image,
        query_coords: mrinr.typing.CoordGrid3D | mrinr.typing.CoordGrid2D,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        affine_query_el2coords: mrinr.typing.HomogeneousAffine3D
        | mrinr.typing.HomogeneousAffine2D,
        y_mlp_output: mrinr.typing.Volume | mrinr.typing.Image,
    ) -> mrinr.typing.Volume | mrinr.typing.Image:
        if self.interpolate_skip is None:
            r = y_mlp_output
        else:
            y_interp = self.interpolate_skip(
                x,
                x_coords=None,
                query_coords=query_coords,
                affine_x_el2coords=affine_x_el2coords,
                affine_query_el2coords=affine_query_el2coords,
                return_coord_space=False,
            )
            # y_mlp_output = y_mlp_output * self._mlp_out_weight
            if self._interpolate_skip_combine_method == "concat":
                # Concatenate the interpolated features with the MLP output.
                r = self.merge_interpolate_skip_mlp_conv(
                    torch.cat([y_mlp_output, y_interp], dim=1)
                )
            elif self._interpolate_skip_combine_method == "add":
                if self.training:
                    r = y_mlp_output + y_interp
                else:
                    y_mlp_output += y_interp
                    r = y_mlp_output
            elif self._interpolate_skip_combine_method == "mul":
                if self.training:
                    r = y_mlp_output * y_interp
                else:
                    y_mlp_output *= y_interp
                    r = y_mlp_output
        return r

    def _get_chunked_query_features(
        self,
        max_q_chunks: int,
        coords_to_chunk: list[torch.Tensor | None],
        spatial_data_to_chunk: list[mrinr.typing.Image | mrinr.typing.Volume | None],
    ) -> tuple[list[tuple[torch.Tensor, ...]], int, int]:
        spatial_shapes = [
            tuple(c.shape[1:-1])
            for c in filter(lambda c: c is not None, coords_to_chunk)
        ] + [
            tuple(d.shape[2:])
            for d in filter(lambda s: s is not None, spatial_data_to_chunk)
        ]
        spatial_shapes = np.asarray(spatial_shapes, dtype=int)
        # All spatial shapes should be the same.
        assert (spatial_shapes == spatial_shapes[0][None]).all()

        # If only 1 chunk is requested, then just return the inputs.
        if max_q_chunks is None or int(max_q_chunks) <= 1:
            # Choose arbitrary concatenation dimensions, as only one tensor will be
            # concatenated, here the "x" spatial dimension.
            coords_chunk_dim = 1
            data_chunk_dim = 2
            chunks = [tuple(coords_to_chunk) + tuple(spatial_data_to_chunk)]
        else:
            spatial_shape = tuple(spatial_shapes[0])
            # Find the largest spatial dimension for both coordinate tensors and data
            # (image or volume) tensors.
            max_spatial_chunks, coords_chunk_dim, data_chunk_dim = max(
                zip(
                    spatial_shape,
                    range(1, self.spatial_dims + 1),
                    range(2, self.spatial_dims + 2),
                ),
                key=lambda s: s[0],
            )
            # Decide on the number of chunks.
            n_chunks = min(max_spatial_chunks, max_q_chunks)
            # Chunks for all data, but with the outer index being the tensor and the
            # inner index being the chunk index.
            chunks__tensor_chnk = [
                (
                    torch.chunk(c, chunks=n_chunks, dim=coords_chunk_dim)
                    if c is not None
                    else itertools.repeat(None, n_chunks)
                )
                for c in coords_to_chunk
            ] + [
                (
                    torch.chunk(d, chunks=n_chunks, dim=data_chunk_dim)
                    if d is not None
                    else itertools.repeat(None, n_chunks)
                )
                for d in spatial_data_to_chunk
            ]
            # The None repeats may have more elements than the actual returned chunks,
            # so rely on zip to truncate to the number of actual chunks from the data.
            # Transpose to have the outer index be the chunk index.
            chunks = tuple(zip(*chunks__tensor_chnk))

        return (chunks, coords_chunk_dim, data_chunk_dim)

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
        x_grid_sizes: Optional[torch.Tensor],
        query_grid_sizes: Optional[torch.Tensor],
        *,
        max_q_chunks: Optional[int] = None,
        return_coord_space: bool = False,
        **kwargs,
    ) -> Union[
        mrinr.typing.Volume,
        mrinr.typing.Image,
        "mrinr.nn.spatial_coders.DenseCoordSpace",
    ]:
        raise NotImplementedError("Subclasses must implement this method.")


class EnsembleSpatialINR(_SpatialINRBase):
    _ENSEMBLE_SAMPLE_PADDING_MODE = "border"

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_spatial_features: int,
        out_features: int,
        interpolate_skip_connect: bool,
        activate_fn: str | tuple[str, dict[str, Any]],
        *,
        use_norm_input_coords: bool,
        use_input_grid_sizes: bool,
        use_query_grid_sizes: bool,
        coord_encoding: Optional[Literal["positional", "gaussian"]],
        use_encode_input_coords: bool,
        use_encode_ensemble_coords_diff: bool,
        use_encode_query_coords: bool,
        coord_encoding_kwargs: dict[str, Any] = dict(),
        interpolate_skip_combine_method: Optional[
            Literal["concat", "add", "mul"]
        ] = "add",
        res_mlp_combine_method: Optional[Literal["concat", "add", "mul"]] = None,
        allow_approx_infer: bool = False,
        **mlp_kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(locals())
        torch.nn.Module.__init__(self)
        self._set_common_properties(**self._init_kwargs)
        self._inr_feature_names = {
            "x",
            "norm_input_coords",
            "norm_query_coords",
            "input_grid_sizes",
            "query_grid_sizes",
            "input_coords_encoding",
            "query_coords_encoding",
            "norm_ensemble_coords_diff",
            "ensemble_coords_diff_encoding",
        }
        self._use_encode_ensemble_coords_diff = use_encode_ensemble_coords_diff
        # Always use the normalized ensemble coordinate diffs as input.
        self._use_norm_ensemble_coords_diff = True
        # Set all feature flags before calculating feature order/size.
        self._feature_order, self._feature_sizes = self._get_feature_order_size()

        # Construct the MLP according to the type of ResMLP indicated.
        if self._res_mlp_combine_method in {None, "concat"}:
            self.inr_mlp = _DenseResINRMLP(
                in_features=sum(self._feature_sizes),
                out_features=self._out_features,
                activate_fn=self._activate_fn_init_obj,
                **mlp_kwargs,
                # internal_layer_features=mlp_internal_layer_features,
                # res_connections_out_layer2in_layer_idx=mlp_res_connect_mapping,
            )
        elif self._res_mlp_combine_method in {"add", "mul"}:
            self.inr_mlp = _ElementwiseResINRMLP(
                in_features=sum(self._feature_sizes),
                out_features=self._out_features,
                activate_fn=self._activate_fn_init_obj,
                elementwise_op=self._res_mlp_combine_method,
                end_with_lin=True,
                **mlp_kwargs,
            )
        #!TESTING
        # Zero-out final linear layer if the MLP + interp will be combined.
        if self._interpolate_skip_combine_method in {"add", "mul"}:
            self.inr_mlp.linear_blocks[-1].weight.data.zero_()
            self.inr_mlp.linear_blocks[-1].bias.data.zero_()

    # def mlp_parameters(self):
    #     p = self.inr_mlp.parameters()
    #     if getattr(self, "merge_interpolate_skip_mlp_conv", None) is not None:
    #         p = itertools.chain(p, self.merge_interpolate_skip_mlp_conv.parameters())
    #     return p

    def _get_feature_order_size(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        # Order by type of feature: spatial features, then coordinate features.
        # Then, order coordinates by input, ensemble, and query coordinates.
        all_features = collections.OrderedDict(
            x=True,
            norm_input_coords=self._use_norm_input_coords,
            input_coords_encoding=self._use_encode_input_coords,
            input_grid_sizes=self._use_input_grid_sizes,
            norm_ensemble_coords_diff=self._use_norm_ensemble_coords_diff,
            ensemble_coords_diff_encoding=self._use_encode_ensemble_coords_diff,
            norm_query_coords=self._use_norm_query_coords,
            query_coords_encoding=self._use_encode_query_coords,
            query_grid_sizes=self._use_query_grid_sizes,
        )
        all_feature_sizes = collections.OrderedDict(
            x=self._in_spatial_features,
            norm_input_coords=self.spatial_dims,
            input_coords_encoding=self.coord_encoder.out_features
            if self.coord_encoder is not None
            else None,
            input_grid_sizes=self.spatial_dims,
            norm_ensemble_coords_diff=self.spatial_dims,
            ensemble_coords_diff_encoding=self.coord_encoder.out_features
            if self.coord_encoder is not None
            else None,
            norm_query_coords=self.spatial_dims,
            query_coords_encoding=self.coord_encoder.out_features
            if self.coord_encoder is not None
            else None,
            query_grid_sizes=self.spatial_dims,
        )

        assert set(all_features.keys()) == self._inr_feature_names
        assert set(all_features.keys()) == set(all_feature_sizes.keys())

        features = list()
        feature_sizes = list()
        # Only include features that are enabled.
        for k in all_features.keys():
            if all_features[k]:
                features.append(k)
                feature_sizes.append(all_feature_sizes[k])
        return tuple(features), tuple(feature_sizes)

    def reorder_and_split_features(
        self, cat_dim: int = 1, **feat_names_to_tensors
    ) -> torch.Tensor | tuple[torch.Tensor]:
        # Reorder features to match the expected order.
        # Ensure all expected features are present.
        assert set(feat_names_to_tensors.keys()) == set(self._feature_order)
        feat_tensors = [feat_names_to_tensors[k] for k in self._feature_order]
        # Concatenate along the feature dimension.
        y = torch.cat(feat_tensors, dim=cat_dim)
        return y

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
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_grid_sizes: Optional[torch.Tensor] = None,
        *,
        max_q_chunks: Optional[int] = None,
        return_coord_space: bool = False,
        **kwargs,
    ) -> Union[
        mrinr.typing.Volume,
        mrinr.typing.Image,
        "mrinr.nn.spatial_coders.DenseCoordSpace",
    ]:
        # Gather query coordinate features for chunking.
        q_spatial_dim_labels = {"x": query_coords.shape[1], "y": query_coords.shape[2]}
        if self.spatial_dims == 3:
            q_spatial_dim_labels = q_spatial_dim_labels | {"z": query_coords.shape[3]}
        norm_q_coords = (
            query_coords - query_coord_normalizer_params.min_shift
        ) / query_coord_normalizer_params.size_scale
        if self._use_encode_query_coords:
            q_coords_encoding = self.coord_encoder(norm_q_coords)
        else:
            q_coords_encoding = None
        # Repeat query grid sizes to fill the image/volume shape.
        if self._use_query_grid_sizes or self._use_input_grid_sizes:
            pattern = (
                "b coord -> b coord x y"
                if self.spatial_dims == 2
                else "b coord -> b coord x y z"
            )
            q_grid_sizes = (
                einops.repeat(query_grid_sizes, pattern, **q_spatial_dim_labels)
                if self._use_query_grid_sizes
                else None
            )
            in_grid_sizes = (
                einops.repeat(x_grid_sizes, pattern, **q_spatial_dim_labels)
                if self._use_input_grid_sizes
                else None
            )
        else:
            q_grid_sizes = None
            in_grid_sizes = None

        batch_size = x.shape[0]
        batch_id_affine = torch.eye(
            self.spatial_dims + 1,
            dtype=affine_x_el2coords.dtype,
            device=affine_x_el2coords.device,
        ).repeat(batch_size, 1, 1)

        ##### Iterate over chunks of query features.
        q_chunks, _, spatial_data_cat_dim = self._get_chunked_query_features(
            1 if max_q_chunks is None else max_q_chunks,
            coords_to_chunk=[query_coords, norm_q_coords],
            spatial_data_to_chunk=[q_coords_encoding, q_grid_sizes, in_grid_sizes],
        )
        y = list()
        for (
            q_coords_i,
            norm_q_coords_i,
            q_coords_enc_i,
            q_grid_sizes_i,
            in_grid_sizes_i,
        ) in q_chunks:
            in_features_i = dict()
            if self._use_norm_query_coords:
                in_features_i["norm_query_coords"] = self.rearrange_coords_as_channels(
                    norm_q_coords_i
                )
            if self._use_query_grid_sizes:
                in_features_i["query_grid_sizes"] = q_grid_sizes_i
            if self._use_input_grid_sizes:
                in_features_i["input_grid_sizes"] = in_grid_sizes_i
            if self._use_encode_query_coords:
                in_features_i["query_coords_encoding"] = q_coords_enc_i

            ###### Loop over elements in the ensemble surrounding q.
            if self.spatial_dims == 2:
                ensemble_iter = spatial_ensemble_2d_coord_features_generator(
                    query_coords=q_coords_i, affine_pixel2coords=affine_x_el2coords
                )
            elif self.spatial_dims == 3:
                ensemble_iter = spatial_ensemble_3d_coord_features_generator(
                    query_coords=q_coords_i, affine_vox2coords=affine_x_el2coords
                )
            y_i = None
            for ensemble_feats_j in ensemble_iter:
                in_features_ij = in_features_i.copy()
                # Sample the ensemble in x surrounding the query point by using nearest-
                # neighbor sampling. The `input_el_idx` is already in (pix/vox)el
                # coordinates, so the affine transform is just identity.
                x_ensemble = mrinr.grid_resample(
                    x,
                    sample_coords=ensemble_feats_j.input_el_idx,
                    affine_x_el2coords=batch_id_affine,
                    interp_lib="torch",
                    mode_or_interpolation="nearest",
                    padding_mode_or_bound=self._ENSEMBLE_SAMPLE_PADDING_MODE,
                )
                in_features_ij["x"] = x_ensemble

                # Ensemble diff features.
                if (
                    self._use_norm_ensemble_coords_diff
                    or self._use_encode_ensemble_coords_diff
                ):
                    # Normalize from [-1, 1] to [0, 1]
                    norm_delta_ensemble_x_q = (
                        ensemble_feats_j.diff_ensemble_corner_to_ensemble_q + 1
                    ) / 2
                    if self._use_norm_ensemble_coords_diff:
                        in_features_ij["norm_ensemble_coords_diff"] = (
                            self.rearrange_coords_as_channels(norm_delta_ensemble_x_q)
                        )
                    if self._use_encode_ensemble_coords_diff:
                        in_features_ij["ensemble_coords_diff_encoding"] = (
                            self.coord_encoder(norm_delta_ensemble_x_q)
                        )
                # Input coordinate features.
                if self._use_norm_input_coords or self._use_encode_input_coords:
                    input_coords_ij = mrinr.coords.transform_coords(
                        ensemble_feats_j.input_el_idx, affine_a2b=affine_x_el2coords
                    )
                    norm_input_coords_ij = (
                        input_coords_ij - x_coord_normalizer_params.min_shift
                    ) / x_coord_normalizer_params.size_scale
                    if self._use_norm_input_coords:
                        in_features_ij["norm_input_coords"] = (
                            self.rearrange_coords_as_channels(norm_input_coords_ij)
                        )
                    if self._use_encode_input_coords:
                        in_features_ij["input_coords_encoding"] = self.coord_encoder(
                            norm_input_coords_ij
                        )

                # Concatenate features and pass through the MLP.
                x_features_ij = self.reorder_and_split_features(**in_features_ij)
                if self.spatial_dims == 2:
                    x_features_ij = einops.rearrange(
                        x_features_ij, "b c x y -> (b x y) c"
                    )
                elif self.spatial_dims == 3:
                    x_features_ij = einops.rearrange(
                        x_features_ij, "b c x y z -> (b x y z) c"
                    )

                pred_ij = self.inr_mlp(x_features_ij)

                if self.spatial_dims == 2:
                    pred_ij = einops.rearrange(
                        pred_ij,
                        "(b x y) c -> b c x y",
                        x=q_coords_i.shape[1],
                        y=q_coords_i.shape[2],
                    )
                elif self.spatial_dims == 3:
                    pred_ij = einops.rearrange(
                        pred_ij,
                        "(b x y z) c -> b c x y z",
                        x=q_coords_i.shape[1],
                        y=q_coords_i.shape[2],
                        z=q_coords_i.shape[3],
                    )
                # Initialize y_i here, easier than determining the prediction shapes
                # beforehand.
                if y_i is None:
                    y_i = torch.zeros_like(pred_ij)
                # Accumulate the prediction for each ensemble element, weighted by
                # the opposite area/volume of the chosen ensemble corner.
                y_i += pred_ij * ensemble_feats_j.spatial_weight
            y.append(y_i)
        # If the query was not chunked, then there is no need to concatenate the result.
        if len(q_chunks) == 1:
            y = y[0]
        else:
            # Concatenate on the chunk dimension.
            y = torch.cat(y, dim=spatial_data_cat_dim)

        if self._interpolate_skip_connect:
            y = self._apply_interpolate_skip_connect(
                x=x,
                query_coords=query_coords,
                affine_x_el2coords=affine_x_el2coords,
                affine_query_el2coords=affine_query_el2coords,
                y_mlp_output=y,
            )

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=query_coords, affine=affine_query_el2coords
            )
        else:
            r = y
        return r


class LTE(_SpatialINRBase):
    # Match the vanilla ensemble INR without inheritance.
    _ENSEMBLE_SAMPLE_PADDING_MODE = EnsembleSpatialINR._ENSEMBLE_SAMPLE_PADDING_MODE

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_spatial_features: int,
        out_features: int,
        k_freqs: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        res_mlp_combine_method: Optional[Literal["concat", "add", "mul"]] = None,
        interpolate_skip_combine_method: Literal["concat", "add", "mul"] | None = "add",
        allow_approx_infer: bool = False,
        **mlp_kwargs,
    ):
        """Local Texture Estimator (LTE)

        From:
        J. Lee and K. H. Jin, “Local Texture Estimator for Implicit Representation
        Function,” presented at the Proceedings of the IEEE/CVF Conference on Computer
        Vision and Pattern Recognition, 2022, pp. 1929-1938. Accessed: Jan. 10, 2024.
        [Online]. Available:
        https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Local_Texture_Estimator_for_Implicit_Representation_Function_CVPR_2022_paper.html

        Parameters
        ----------
        spatial_dims : Literal[2, 3]
            Spatial dimensions (2D or 3D).
        in_spatial_features : int
            Number of channels given in the input.
        out_features : int
            Number of output features from the network.
        k_freqs : int
            Number of frequencies to expand within the model.
        activate_fn : str | tuple[str, dict[str, Any]]
            Activation function for the INR MLP.
        interpolate_skip_combine_method : Literal[&quot;concat&quot;, &quot;add&quot;, &quot;mul&quot;], optional
            Method for combining interp skip connection to mlp output, by default "add"
        """
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(locals(), warn=True)
        torch.nn.Module.__init__(self)
        self._set_common_properties(
            spatial_dims=spatial_dims,
            in_spatial_features=in_spatial_features,
            out_features=out_features,
            activate_fn=activate_fn,
            interpolate_skip_connect=True,
            interpolate_skip_combine_method=interpolate_skip_combine_method,
            use_input_grid_sizes=True,
            use_query_grid_sizes=True,
            res_mlp_combine_method=res_mlp_combine_method,
        )
        self._k_freqs = k_freqs
        if self.spatial_dims == 2:
            conv_cls = torch.nn.Conv2d
        elif self.spatial_dims == 3:
            conv_cls = torch.nn.Conv3d

        self.amplitude_layer = conv_cls(
            in_channels=self._in_spatial_features,
            out_channels=self._k_freqs * self.spatial_dims,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )
        self.freq_coef_layer = conv_cls(
            in_channels=self._in_spatial_features,
            out_channels=self._k_freqs * self.spatial_dims,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )
        self.phase_layer = torch.nn.Linear(self.spatial_dims, self._k_freqs, bias=False)

        # Construct the MLP according to the type of ResMLP indicated.
        # MLP takes in cos + sin of the frequency features.
        if self._res_mlp_combine_method in {None, "concat"}:
            if self._res_mlp_combine_method is None:
                mlp_kwargs = mlp_kwargs | {
                    "res_connections_out_layer2in_layer_idx": None
                }
            self.inr_mlp = _DenseResINRMLP(
                in_features=self._k_freqs * self.spatial_dims * 2,
                out_features=self._out_features,
                activate_fn=self._activate_fn_init_obj,
                **mlp_kwargs,
            )
        elif self._res_mlp_combine_method in {"add", "mul"}:
            self.inr_mlp = _ElementwiseResINRMLP(
                in_features=self._k_freqs * self.spatial_dims * 2,
                out_features=self._out_features,
                activate_fn=self._activate_fn_init_obj,
                elementwise_op=self._res_mlp_combine_method,
                end_with_lin=True,
                **mlp_kwargs,
            )
        #!TESTING
        # Zero-out final linear layer if the MLP + interp will be combined.
        if self._interpolate_skip_combine_method in {"add", "mul"}:
            self.inr_mlp.linear_blocks[-1].weight.data.zero_()
            self.inr_mlp.linear_blocks[-1].bias.data.zero_()

    # def mlp_parameters(self):
    #     params = [
    #         self.amplitude_layer.parameters(),
    #         self.freq_coef_layer.parameters(),
    #         self.phase_layer.parameters(),
    #         self.inr_mlp.parameters(),
    #     ]
    #     if getattr(self, "merge_interpolate_skip_mlp_conv", None) is not None:
    #         params.append(self.merge_interpolate_skip_mlp_conv.parameters())
    #     return itertools.chain(*params)

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
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_grid_sizes: torch.Tensor = None,
        *,
        max_q_chunks: Optional[int] = None,
        return_coord_space: bool = False,
        **kwargs,
    ) -> Union[
        mrinr.typing.Volume,
        mrinr.typing.Image,
        "mrinr.nn.spatial_coders.DenseCoordSpace",
    ]:
        if query_grid_sizes is None:
            raise ValueError("query_grid_sizes must be provided.")
        # Gather query coordinate features for chunking.
        q_spatial_dim_labels = {"x": query_coords.shape[1], "y": query_coords.shape[2]}
        if self.spatial_dims == 3:
            q_spatial_dim_labels = q_spatial_dim_labels | {"z": query_coords.shape[3]}

        # Input shape into the MLP is either:
        # - 'b x y k d cos_sin' for 2D
        # - or, 'b x y z k d cos_sin' for 3D
        # Repeat query grid sizes to fill the image/volume shape.
        if self.spatial_dims == 2:
            phase_repeat_pattern = "b k -> b k d x y"
            phase_reshape_to_spatial_pattern = "b k d x y -> b (k d) x y"
            phase_reshape_to_freq_pattern = "b (k d) x y -> b x y k d"
            local_coord_repeat_pattern = "b x y d -> b x y k d"
            amp_freq_reshape_pattern = "b (k d) x y -> b x y k d"
        elif self.spatial_dims == 3:
            phase_repeat_pattern = "b k -> b k d x y z"
            phase_reshape_to_spatial_pattern = "b k d x y z -> b (k d) x y z"
            phase_reshape_to_freq_pattern = "b (k d) x y z -> b x y z k d"
            local_coord_repeat_pattern = "b x y z d -> b x y z k d"
            amp_freq_reshape_pattern = "b (k d) x y z -> b x y z k d"

        # Pre-compute amplitude, frequency, and phase features, and sample them as if
        # sampling the original input.
        x_freq = self.freq_coef_layer(x)
        x_amp = self.amplitude_layer(x)
        # Compute phase features, and repeat + reshape to the spatial shape of the
        # query coordinates.

        # NOTE: The grid sizes ("cells") are in the
        # scale of the *coordinate space*, not the image/volume space. A.k.a., the
        # scaling according to the affine matrix. This is different from the original
        # LTE paper, where the cells were pixel scale.
        phase_q = self.phase_layer(query_grid_sizes)
        phase_q = einops.repeat(
            phase_q,
            phase_repeat_pattern,
            k=self._k_freqs,
            d=self.spatial_dims,
            **q_spatial_dim_labels,
        )
        phase_q = einops.rearrange(phase_q, phase_reshape_to_spatial_pattern)

        # Create identity affines for sampling with pixel/voxel coordinates.
        batch_size = x.shape[0]
        batch_id_affine = torch.eye(
            self.spatial_dims + 1,
            dtype=affine_x_el2coords.dtype,
            device=affine_x_el2coords.device,
        ).repeat(batch_size, 1, 1)

        ##### Iterate over chunks of query features.
        # NOTE:
        # This loop is very similar to the EnsembleSpatialINR, but the local coordinates
        # within the ensemble ('delta') are used to create the input to the MLP, so we
        # cannot just have a spatial ensemble "child module," but fully recreate the
        # ensemble loop here.
        q_chunks, _, spatial_data_cat_dim = self._get_chunked_query_features(
            1 if max_q_chunks is None else max_q_chunks,
            coords_to_chunk=[query_coords],
            spatial_data_to_chunk=[phase_q],
        )
        y = list()
        for q_coords_i, phase_q_i in q_chunks:
            ###### Loop over elements in the ensemble surrounding q.
            if self.spatial_dims == 2:
                ensemble_iter = mrinr.nn.spatial_ensemble_2d_coord_features_generator(
                    query_coords=q_coords_i, affine_pixel2coords=affine_x_el2coords
                )
            elif self.spatial_dims == 3:
                ensemble_iter = mrinr.nn.spatial_ensemble_3d_coord_features_generator(
                    query_coords=q_coords_i, affine_vox2coords=affine_x_el2coords
                )
            phase_q_i = einops.rearrange(
                phase_q_i,
                phase_reshape_to_freq_pattern,
                d=self.spatial_dims,
                k=self._k_freqs,
            )
            y_i = None
            for ensemble_feats_j in ensemble_iter:
                # Sample the ensemble surrounding the query point by using nearest-
                # neighbor sampling. The `input_el_idx` is already in (pix/vox)el
                # coordinates, so the affine transform is just identity.
                x_freq_ensemble = mrinr.grid_resample(
                    x_freq,
                    sample_coords=ensemble_feats_j.input_el_idx,
                    affine_x_el2coords=batch_id_affine,
                    interp_lib="torch",
                    mode_or_interpolation="nearest",
                    padding_mode_or_bound=self._ENSEMBLE_SAMPLE_PADDING_MODE,
                )
                # Move frequency "channels" into a normalized '[spatial_dims] x k x d'
                x_freq_ensemble = einops.rearrange(
                    x_freq_ensemble,
                    amp_freq_reshape_pattern,
                    k=self._k_freqs,
                    d=self.spatial_dims,
                )
                x_amp_ensemble = mrinr.grid_resample(
                    x_amp,
                    sample_coords=ensemble_feats_j.input_el_idx,
                    affine_x_el2coords=batch_id_affine,
                    interp_lib="torch",
                    mode_or_interpolation="nearest",
                    padding_mode_or_bound=self._ENSEMBLE_SAMPLE_PADDING_MODE,
                )
                # Move amplitude "channels" into a normalized '[spatial_dims] x k x d'
                x_amp_ensemble = einops.rearrange(
                    x_amp_ensemble,
                    amp_freq_reshape_pattern,
                    k=self._k_freqs,
                    d=self.spatial_dims,
                )
                #!
                # Compute the ensemble's coordinate in the full affine space, rather
                # in local ensemble space.
                # delta_coord_i = (
                #     mrinr.coords.transform_coords(
                #         ensemble_feats_j.input_el_idx, affine_a2b=affine_x_el2coords
                #     )
                #     - q_coords_i
                # )
                # delta_coord_i = delta_coord_i.absolute_()
                #!
                # Compute delta in ensemble coordinates, rather than full affine space.
                # The LTE paper has delta as (x_query - x_ensemble_corner), so just
                # negate the (x_ensemble_corner - x_query).
                delta_coord_i = -ensemble_feats_j.diff_ensemble_corner_to_ensemble_q
                #!
                # Repeat delta to a normalized '[spatial_dims] x k x d' shape.
                delta_coord_i = einops.repeat(
                    delta_coord_i, local_coord_repeat_pattern, k=self._k_freqs
                )

                # Compute the cosine and sine of the input signal features.
                cos_x = x_amp_ensemble * torch.cos(
                    torch.pi * x_freq_ensemble * delta_coord_i + phase_q_i
                )
                sin_x = x_amp_ensemble * torch.sin(
                    torch.pi * x_freq_ensemble * delta_coord_i + phase_q_i
                )
                # Combine cosine and sine features.
                x_features_ij = torch.stack([cos_x, sin_x], dim=-1)
                # Reshape for input to the MLP.
                if self.spatial_dims == 2:
                    x_features_ij = einops.rearrange(
                        x_features_ij, "b x y k d cos_sin -> (b x y) (k d cos_sin)"
                    )
                elif self.spatial_dims == 3:
                    x_features_ij = einops.rearrange(
                        x_features_ij, "b x y z k d cos_sin -> (b x y z) (k d cos_sin)"
                    )

                pred_ij = self.inr_mlp(x_features_ij)

                if self.spatial_dims == 2:
                    pred_ij = einops.rearrange(
                        pred_ij,
                        "(b x y) c -> b c x y",
                        x=q_coords_i.shape[1],
                        y=q_coords_i.shape[2],
                    )
                elif self.spatial_dims == 3:
                    pred_ij = einops.rearrange(
                        pred_ij,
                        "(b x y z) c -> b c x y z",
                        x=q_coords_i.shape[1],
                        y=q_coords_i.shape[2],
                        z=q_coords_i.shape[3],
                    )
                # Initialize y_i here, easier than determining the prediction shapes
                # beforehand.
                if y_i is None:
                    y_i = torch.zeros_like(pred_ij)
                # Accumulate the prediction for each ensemble element, weighted by
                # the opposite area/volume of the chosen ensemble corner.
                y_i += pred_ij * ensemble_feats_j.spatial_weight
            y.append(y_i)
        # If the query was not chunked, then there is no need to concatenate the result.
        if len(q_chunks) == 1:
            y = y[0]
        else:
            # Concatenate on the chunk dimension.
            y = torch.cat(y, dim=spatial_data_cat_dim)

        if self._interpolate_skip_connect:
            y = self._apply_interpolate_skip_connect(
                x=x,
                query_coords=query_coords,
                affine_x_el2coords=affine_x_el2coords,
                affine_query_el2coords=affine_query_el2coords,
                y_mlp_output=y,
            )

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=query_coords, affine=affine_query_el2coords
            )
        else:
            r = y
        return r


class NeighborhoodSpatialINR(_SpatialINRBase):
    _ENSEMBLE_SAMPLE_PADDING_MODE = "border"

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_spatial_features: int,
        out_features: int,
        interpolate_skip_connect: bool,
        activate_fn: str | tuple[str, dict[str, Any]],
        mlp_internal_layer_features: list[int],
        mlp_res_connect_mapping: Optional[
            dict[int, tuple[int] | int | Literal["dense"]]
        ],
        *,
        use_norm_input_coords: bool,
        use_input_grid_sizes: bool,
        use_query_grid_sizes: bool,
        coord_encoding: Optional[Literal["positional", "gaussian"]],
        use_encode_input_coords: bool,
        use_encode_query_coords: bool,
        coord_encoding_kwargs: dict[str, Any] = dict(),
        interpolate_skip_combine_method: Optional[
            Literal["concat", "add", "mul"]
        ] = "concat",
        **kwargs,
    ):
        raise NotImplementedError("Neighborhood INR is not updated for new functions.")
        # self._init_kwargs = mrinr.nn.get_module_init_kwargs(
        #     locals(), extra_kwargs_dict=kwargs, warn=True
        # )
        # torch.nn.Module.__init__(self)
        # self._set_common_properties(**self._init_kwargs)

        # self._neighborhood_size = 2**self.spatial_dims

        # # Set all feature flags before calculating feature order/size.
        # self._feature_order, self._feature_sizes = self._get_feature_order_size()
        # self.inr_mlp = _ResINRMLP(
        #     in_features=sum(self._feature_sizes),
        #     out_features=self._out_features,
        #     internal_layer_features=mlp_internal_layer_features,
        #     activate_fn=self._activate_fn_init_obj,
        #     res_connections_out_layer2in_layer_idx=mlp_res_connect_mapping,
        # )

    # def mlp_parameters(self):
    #     return self.inr_mlp.parameters()

    def _get_feature_order_size(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        # Order by type of feature: spatial features, then coordinate features.
        # Neighbor index is sorted by a binary indicator of corner position, such as
        # "000" or "101" for 3D, or "00" or "10" for 2D.
        # Then, order coordinates by input and query coordinates.
        all_features = collections.OrderedDict()
        all_feature_sizes = collections.OrderedDict()

        x_feat_names = [k for k in filter(lambda n: "x_" in n, self._inr_feature_names)]
        x_feat_names = sorted(x_feat_names, key=lambda n: n.split("_")[-1])
        for k in x_feat_names:
            all_features[k] = True
            all_feature_sizes[k] = self._in_spatial_features
        # Input coordinate features.
        # Input coordinates
        norm_in_c_names = [
            k
            for k in filter(
                lambda n: "norm_input_coords_" in n, self._inr_feature_names
            )
        ]
        norm_in_c_names = sorted(norm_in_c_names, key=lambda n: n.split("_")[-1])
        for k in norm_in_c_names:
            all_features[k] = self._use_norm_input_coords
            all_feature_sizes[k] = self.spatial_dims
        # Input coord grid sizes
        all_features["input_grid_sizes"] = self._use_input_grid_sizes
        all_feature_sizes["input_grid_sizes"] = self.spatial_dims
        # Input coord encodings
        in_c_enc_names = [
            k
            for k in filter(
                lambda n: "input_coords_encoding_" in n, self._inr_feature_names
            )
        ]
        in_c_enc_names = sorted(in_c_enc_names, key=lambda n: n.split("_")[-1])
        for k in in_c_enc_names:
            all_features[k] = self._use_encode_input_coords
            all_feature_sizes[k] = (
                self.coord_encoder.out_features
                if self.coord_encoder is not None
                else None
            )
        # Query coordinate features.
        all_features["norm_query_coords"] = self._use_norm_query_coords
        all_feature_sizes["norm_query_coords"] = self.spatial_dims
        all_features["query_coords_encoding"] = self._use_encode_query_coords
        all_feature_sizes["query_coords_encoding"] = (
            self.coord_encoder.out_features if self.coord_encoder is not None else None
        )
        all_features["query_grid_sizes"] = self._use_query_grid_sizes
        all_feature_sizes["query_grid_sizes"] = self.spatial_dims

        features = list()
        feature_sizes = list()
        # Only include features that are enabled.
        for k in all_features.keys():
            if all_features[k]:
                features.append(k)
                feature_sizes.append(all_feature_sizes[k])
        return tuple(features), tuple(feature_sizes)

    @property
    def _inr_feature_names(self):
        try:
            r = self.__inr_feature_names
        except AttributeError:
            # Feature names depend on the spatial dims of the network, so each neighbor
            # in the neighborhood is a separate feature. So, feature names must be
            # dynamically built.
            names = [
                "norm_query_coords",
                "query_grid_sizes",
                "query_coords_encoding",
            ]
            # Neighbor index is indicated by a binary indicator of corner position, such
            # as "000" or "101" for 3D, or "00" or "10" for 2D.
            neighbor_iter = itertools.product(*([(0, 1)] * self.spatial_dims))
            for n_idx in neighbor_iter:
                n_str = "".join(str(n) for n in n_idx)
                names.extend(
                    [
                        f"x_{n_str}",
                        f"norm_input_coords_{n_str}",
                        f"input_coords_encoding_{n_str}",
                    ]
                )
            names.append("input_grid_sizes")
            self.__inr_feature_names = set(names)
            r = self.__inr_feature_names

        return r

    def reorder_and_split_features(
        self, cat_dim: int = 1, **feat_names_to_tensors
    ) -> torch.Tensor | tuple[torch.Tensor]:
        # Reorder features to match the expected order.
        # Ensure all expected features are present.
        assert set(feat_names_to_tensors.keys()) == set(self._feature_order)
        feat_tensors = [feat_names_to_tensors[k] for k in self._feature_order]
        # Concatenate along the feature dimension.
        y = torch.cat(feat_tensors, dim=cat_dim)
        return y

    def _collect_neighborhood_feats(
        self,
        ensemble_iter: Iterator[_SpatialEnsembleElement],
        x: mrinr.typing.Image | mrinr.typing.Volume,
        affine_x_el2coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        x_coord_normalizer_params: Union[
            "mrinr.nn.spatial_coders.NormalizerParams", None
        ],
        use_x_features: bool,
        use_norm_coords: bool,
        use_encode_coords: bool,
        x_feature_name_prefix: str = "x_",
        coord_feature_name_prefix: str = "norm_input_coords_",
        coord_encoding_name_prefix: str = "input_coords_encoding_",
    ) -> dict[str, torch.Tensor]:
        """Helper function for gathering all neighborhood features from the ensemble
        iterator, broken out from the forward() function."""

        batch_id_affine = torch.eye(
            self.spatial_dims + 1,
            dtype=affine_x_el2coords.dtype,
            device=affine_x_el2coords.device,
        ).repeat(x.shape[0], 1, 1)

        in_neighborhood_feats = dict()
        for ensemble_element in ensemble_iter:
            # Sample the ensemble in x surrounding the query point by using nearest-
            # neighbor sampling. The `input_el_idx` is already in (pix/vox)el
            # coordinates, so the affine transform is just identity.
            corner_label = ensemble_element.ensemble_corner_label
            if use_x_features:
                x_ensemble = mrinr.grid_resample(
                    x,
                    sample_coords=ensemble_element.input_el_idx,
                    affine_x_el2coords=batch_id_affine,
                    interp_lib="torch",
                    mode_or_interpolation="nearest",
                    padding_mode_or_bound=self._ENSEMBLE_SAMPLE_PADDING_MODE,
                )
                in_neighborhood_feats[f"{x_feature_name_prefix}{corner_label}"] = (
                    x_ensemble
                )

            # Input coordinate features.
            if use_norm_coords or use_encode_coords:
                input_coords = mrinr.coords.transform_coords(
                    ensemble_element.input_el_idx, affine_a2b=affine_x_el2coords
                )
                norm_input_coords = (
                    input_coords - x_coord_normalizer_params.min_shift
                ) / x_coord_normalizer_params.size_scale
                if use_norm_coords:
                    in_neighborhood_feats[
                        f"{coord_feature_name_prefix}{corner_label}"
                    ] = self.rearrange_coords_as_channels(norm_input_coords)
                if use_encode_coords:
                    in_neighborhood_feats[
                        f"{coord_encoding_name_prefix}{corner_label}"
                    ] = self.coord_encoder(norm_input_coords)

        return in_neighborhood_feats

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
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_grid_sizes: Optional[torch.Tensor] = None,
        *,
        max_q_chunks: Optional[int] = None,
        return_coord_space: bool = False,
    ) -> Union[
        mrinr.typing.Volume,
        mrinr.typing.Image,
        "mrinr.nn.spatial_coders.DenseCoordSpace",
    ]:
        # Gather query coordinate features for chunking.
        q_spatial_dim_labels = {"x": query_coords.shape[1], "y": query_coords.shape[2]}
        if self.spatial_dims == 3:
            q_spatial_dim_labels = q_spatial_dim_labels | {"z": query_coords.shape[3]}
        norm_q_coords = (
            query_coords - query_coord_normalizer_params.min_shift
        ) / query_coord_normalizer_params.size_scale
        if self._use_encode_query_coords:
            q_coords_encoding = self.coord_encoder(norm_q_coords)
        else:
            q_coords_encoding = None
        # Repeat query grid sizes to fill the image/volume shape.
        if self._use_query_grid_sizes or self._use_input_grid_sizes:
            pattern = (
                "b coord -> b coord x y"
                if self.spatial_dims == 2
                else "b coord -> b coord x y z"
            )
            q_grid_sizes = (
                einops.repeat(query_grid_sizes, pattern, **q_spatial_dim_labels)
                if self._use_query_grid_sizes
                else None
            )
            in_grid_sizes = (
                einops.repeat(x_grid_sizes, pattern, **q_spatial_dim_labels)
                if self._use_input_grid_sizes
                else None
            )
        else:
            q_grid_sizes = None
            in_grid_sizes = None

        ##### Iterate over chunks of query features.
        q_chunks, _, spatial_data_cat_dim = self._get_chunked_query_features(
            1 if max_q_chunks is None else max_q_chunks,
            coords_to_chunk=[query_coords, norm_q_coords],
            spatial_data_to_chunk=[q_coords_encoding, q_grid_sizes, in_grid_sizes],
        )
        y = list()
        for (
            q_coords_i,
            norm_q_coords_i,
            q_coords_enc_i,
            q_grid_sizes_i,
            in_grid_sizes_i,
        ) in q_chunks:
            in_features_i = dict()
            if self._use_norm_query_coords:
                in_features_i["norm_query_coords"] = self.rearrange_coords_as_channels(
                    norm_q_coords_i
                )
            if self._use_query_grid_sizes:
                in_features_i["query_grid_sizes"] = q_grid_sizes_i
            if self._use_input_grid_sizes:
                in_features_i["input_grid_sizes"] = in_grid_sizes_i
            if self._use_encode_query_coords:
                in_features_i["query_coords_encoding"] = q_coords_enc_i

            # Gather features from elements in the ensemble surrounding q.
            if self.spatial_dims == 2:
                ensemble_iter = spatial_ensemble_2d_coord_features_generator(
                    query_coords=q_coords_i, affine_pixel2coords=affine_x_el2coords
                )
            elif self.spatial_dims == 3:
                ensemble_iter = spatial_ensemble_3d_coord_features_generator(
                    query_coords=q_coords_i, affine_vox2coords=affine_x_el2coords
                )
            neighborhood_feats_i = self._collect_neighborhood_feats(
                ensemble_iter=ensemble_iter,
                x=x,
                affine_x_el2coords=affine_x_el2coords,
                x_coord_normalizer_params=x_coord_normalizer_params,
                use_x_features=True,
                use_norm_coords=self._use_norm_input_coords,
                use_encode_coords=self._use_encode_input_coords,
                x_feature_name_prefix="x_",
                coord_feature_name_prefix="norm_input_coords_",
                coord_encoding_name_prefix="input_coords_encoding_",
            )
            in_features_i = in_features_i | neighborhood_feats_i

            # Concatenate features and pass through the MLP.
            x_features_i = self.reorder_and_split_features(**in_features_i)
            if self.spatial_dims == 2:
                x_features_i = einops.rearrange(x_features_i, "b c x y -> (b x y) c")
            elif self.spatial_dims == 3:
                x_features_i = einops.rearrange(
                    x_features_i, "b c x y z -> (b x y z) c"
                )

            y_i = self.inr_mlp(x_features_i)

            if self.spatial_dims == 2:
                y_i = einops.rearrange(
                    y_i,
                    "(b x y) c -> b c x y",
                    x=q_coords_i.shape[1],
                    y=q_coords_i.shape[2],
                )
            elif self.spatial_dims == 3:
                y_i = einops.rearrange(
                    y_i,
                    "(b x y z) c -> b c x y z",
                    x=q_coords_i.shape[1],
                    y=q_coords_i.shape[2],
                    z=q_coords_i.shape[3],
                )

            y.append(y_i)
        # If the query was not chunked, then there is no need to concatenate the result.
        if len(q_chunks) == 1:
            y = y[0]
        else:
            # Concatenate on the chunk dimension.
            y = torch.cat(y, dim=spatial_data_cat_dim)

        if self._interpolate_skip_connect:
            y = self._apply_interpolate_skip_connect(
                x=x,
                query_coords=query_coords,
                affine_x_el2coords=affine_x_el2coords,
                affine_query_el2coords=affine_query_el2coords,
                y_mlp_output=y,
            )

        if return_coord_space:
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=query_coords, affine=affine_query_el2coords
            )
        else:
            r = y
        return r
