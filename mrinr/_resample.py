# -*- coding: utf-8 -*-
# Functions for resampling 2D images and 3D volumes in pytorch.
from typing import Literal, Optional

import einops
import numpy as np
import scipy
import scipy.ndimage
import torch
import torch.nn.functional as F

import mrinr

__all__ = [
    "grid_resample_torch",
    "grid_resample_interpol",
    "grid_resample_scipy",
    "grid_resample",
    "resize",
]


def _round_up_torch_nearest_neighbor_coords(
    pre_flipped_norm_coords: torch.Tensor, spatial_fov: tuple, tol: float
) -> torch.Tensor:
    # Scale the normalized coordinates to the spatial field of view, and check for any
    # values that are very close to mantissa of 0.5. If any are found within tol, then
    # bump them up by 2 x tol to ensure nearest neighbor interpolation picks a
    # consistent point.
    # If coordinates are not floats, then return the input as is.
    if not torch.is_floating_point(pre_flipped_norm_coords) or tol <= 0:
        rounded_coords = pre_flipped_norm_coords
    else:
        assert tol < 0.25, "Tolerance must be less than 0.25."
        fov = torch.as_tensor(spatial_fov).to(pre_flipped_norm_coords)
        # Get coordinates in element (pixel or voxel) indices.
        el_coord = ((pre_flipped_norm_coords + 1) / 2) * (fov - 1)
        to_be_increased = torch.isclose(
            torch.abs(el_coord - torch.trunc(el_coord)),
            torch.Tensor([0.5]).to(el_coord),
            rtol=0.0,
            atol=tol,
        )
        # For all points that were within tol, bump their coordinates up by 2 x tol,
        # then scale back to normalized grid coordinates.
        rounded_coords = torch.where(
            to_be_increased,
            ((el_coord + 2 * tol) / (fov - 1)) * 2 - 1,
            pre_flipped_norm_coords,
        )

    return rounded_coords


def grid_resample_torch(
    x: torch.Tensor,
    affine_x_el2coords: mrinr.typing.AnyHomogeneousAffineSD,
    sample_coords: mrinr.typing.AnyCoordSD,
    mode: str,
    padding_mode: Literal["zeros", "border", "reflection"],
    clamp: bool = False,
    override_out_of_bounds_val: Optional[float] = None,
    nearest_neighbor_round_tol: float | None = 0.001,
):
    """Resample an image or volume using pytorch's grid_sample().

    Parameters
    ----------
    x : torch.Tensor
        Input image or volume to resample.
    affine_x_el2coords : mrinr.typing.AnyHomogeneousAffineSD
        Homogeneous affine that transforms x element coordinates to sample coordinates.
    sample_coords : mrinr.typing.AnyCoordSD
        Coordinates at which to sample x.
    mode : str
        Interpolation mode for grid_sample().
    padding_mode : Literal[&quot;zeros&quot;, &quot;border&quot;, &quot;reflection&quot;]
        Padding mode for grid_sample().

        Optionals are:
        'zeros'                       :  0  0  0  |  a  b  c  d  |  0  0  0
        'border'                      :  a  a  a  |  a  b  c  d  |  d  d  d
        'reflection'                  :  d  c  b  |  a  b  c  d  |  c  b  a

    override_out_of_bounds_val : Optional[float], optional
        Constant to override any out-of-bounds outputs, by default None

    nearest_neighbor_round_tol : float | None, optional
        Tolerance for rounding up coordinates to nearest neighbor, by default 0.001

    Returns
    -------
    torch.Tensor
        x resampled at sample_coords.

    Raises
    ------
    RuntimeError
        Spatial dims are not 2 or 3, or inconsistent between affines and coordinates.
    RuntimeError
        Batch sizes are incompatible between x, affine_x_el2coords, and sample_coords.
    """
    # Determine spatial dimensions based on the homogeneous affine matrix and coordinate
    # shapes.
    if (
        affine_x_el2coords.shape[-1] == 3
        and affine_x_el2coords.shape[-2] == 3
        and sample_coords.shape[-1] == 2
    ):
        spatial_dims = 2
    elif (
        affine_x_el2coords.shape[-1] == 4
        and affine_x_el2coords.shape[-2] == 4
        and sample_coords.shape[-1] == 3
    ):
        spatial_dims = 3
    else:
        raise RuntimeError(
            "Invalid or incompatible affine and coordinates shapes: "
            + f"{affine_x_el2coords.shape = }, {sample_coords.shape = }"
        )

    if spatial_dims == 2:
        d = mrinr.utils.ensure_image_channels(x, batch=True)
        coords, aff_x_el2coords = mrinr.coords._canonicalize_coords_2d_affine(
            sample_coords, affine=affine_x_el2coords
        )
    elif spatial_dims == 3:
        d = mrinr.utils.ensure_vol_channels(x, batch=True)
        coords, aff_x_el2coords = mrinr.coords._canonicalize_coords_3d_affine(
            sample_coords, affine=affine_x_el2coords
        )
    # Check for a common batch size.
    if d.shape[0] != coords.shape[0] or d.shape[0] != aff_x_el2coords.shape[0]:
        raise RuntimeError(
            "Input batch sizes are incompatible: "
            + f"x shape {tuple(x.shape)} inferred batch size {d.shape[0]}, "
            + f"affine shape {tuple(affine_x_el2coords.shape)} "
            + f"inferred batch size {aff_x_el2coords.shape[0]}, and "
            + f"sample coordinates shape {tuple(sample_coords.shape)} "
            + f"inferred batch size {coords.shape[0]}."
        )

    # Transform from (pi|vo)xel to normalized grid coordinates in range [-1, 1] as
    # required by grid_sample().
    spatial_fov = tuple(d.shape[2:])
    aff_x_el2norm_grid = mrinr.coords.affine_el2normalized_grid(
        spatial_fov, lower=-1.0, upper=1.0, to_tensor=aff_x_el2coords
    )
    aff_x_el2norm_grid.unsqueeze_(0)
    # Merge transformations to map grid coordinates to normalized grid space,
    # broadcasting the el2grid affine over the batches in the input.
    # sample coordinates -> x element indices -> normalized grid coordinates
    aff_coords2norm_grid = mrinr.coords.combine_affines(
        mrinr.coords.inv_affine(aff_x_el2coords),
        aff_x_el2norm_grid,
        transform_order_left_to_right=True,
    )
    sample_norm_grid_coords = mrinr.coords.transform_coords(
        coords, aff_coords2norm_grid
    )
    # Ensure the input is float type.
    if not torch.is_floating_point(x):
        d = d.to(torch.promote_types(torch.float32, sample_norm_grid_coords.dtype))
    # Re-sample the volume with pytorch's grid_sample().

    # Determine interpolation parameters.
    m = mode.strip().lower()
    if "linear" in m:
        mode = "bilinear"
    elif "cubic" in m:
        mode = "bicubic"
    elif ("nearest" in m) or (m == "nn"):
        mode = "nearest"
    p = padding_mode.strip().lower()
    if "zero" in p:
        padding_mode = "zeros"
    elif "reflect" in p:
        padding_mode = "reflection"

    # If nearest-neighbor interpolation is used, check for coordinates that lie in the
    # middle of a (pi|vo)xel.
    if mode == "nearest" and nearest_neighbor_round_tol is not None:
        sample_norm_grid_coords = _round_up_torch_nearest_neighbor_coords(
            sample_norm_grid_coords, spatial_fov, nearest_neighbor_round_tol
        )
    # Reverse the order of the coordinate dimension to work properly with grid_sample!
    # This is very easy to get wrong, and is not well documented in the pytorch docs!
    sample_norm_grid_coords = torch.flip(sample_norm_grid_coords, dims=(-1,))
    samples = F.grid_sample(
        d,
        grid=sample_norm_grid_coords.to(d),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    # If out-of-bounds samples should be overridden, set those sample values now.
    # Otherwise, the grid_sample() only interpolates with the padded values, which
    # allows out-of-bounds samples to appear to be in bounds.
    if override_out_of_bounds_val is not None:
        samples.masked_fill_(
            (sample_norm_grid_coords < -1).any(dim=-1)[:, None]
            | (sample_norm_grid_coords > 1).any(dim=-1)[:, None],
            override_out_of_bounds_val,
        )

    # Change samples back to the input dtype only if the input was not a floating point,
    # and the interpolation was nearest-neighbor.
    if not torch.is_floating_point(x) and mode == "nearest":
        samples = samples.round().to(x.dtype)

    # If the interpolation order was > 1, then the result may contain ringing artifacts.
    # So if indicated, clamp the samples to the input range.
    if clamp and mode == "bicubic":
        if spatial_dims == 2:
            p = "b c x y -> b c 1 1"
        elif spatial_dims == 3:
            p = "b c x y z -> b c 1 1 1"
        min_ = einops.reduce(d, p, "min")
        max_ = einops.reduce(d, p, "max")
        samples = torch.clamp(samples, min=min_, max=max_)

    if spatial_dims == 2:
        samples = mrinr.utils.undo_image_channels(samples, orig_x=x, strict=False)
    elif spatial_dims == 3:
        samples = mrinr.utils.undo_vol_channels(samples, orig_x=x, strict=False)

    return samples


def grid_resample_interpol(
    x: torch.Tensor,
    affine_x_el2coords: mrinr.typing.AnyHomogeneousAffineSD,
    sample_coords: mrinr.typing.AnyCoordSD,
    interpolation: str | int | list[str | int],
    bound: str | list[str],
    clamp: bool = False,
    prefilter: bool = True,
) -> torch.Tensor:
    """Resample an image or volume using the interpol library.

    From the interpol documentation:

    --------------------------------------------------------------------------

    Notes
    -----

    `interpolation` can be an int, a string or an InterpolationType.
    Possible values are:
        - 0 or 'nearest'
        - 1 or 'linear'
        - 2 or 'quadratic'
        - 3 or 'cubic'
        - 4 or 'fourth'
        - 5 or 'fifth'
        - etc.
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders.

    `bound` can be an int, a string or a BoundType.
    Possible values are:
        - 'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
        - 'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
        - 'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
        - 'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
        - 'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
        - 'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
        - 'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    Note that
    - `dft` corresponds to circular padding
    - `dct2` corresponds to Neumann boundary conditions (symmetric)
    - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
    See https://en.wikipedia.org/wiki/Discrete_cosine_transform
        https://en.wikipedia.org/wiki/Discrete_sine_transform
    --------------------------------------------------------------------------

    Parameters
    ----------
    x : torch.Tensor
        Input image or volume to resample.
    affine_x_el2coords : mrinr.typing.AnyHomogeneousAffineSD
        Homogeneous affine that transforms x element coordinates to sample coordinates.
    sample_coords : mrinr.typing.AnyCoordSD
        Coordinates at which to sample x.
    interpolation : str | int | list[str  |  int]
        Type of interpolation to perform; see interpol documentation snippet above.
    bound : str | list[str]
        Boundary padding method; see interpol documentation snippet above.
    prefilter : bool, optional
        Apply spline pre-filter (see interpol documentation), by default True

    Returns
    -------
    torch.Tensor
        x resampled at sample_coords.

    Raises
    ------
    RuntimeError
        Spatial dims are not 2 or 3, or inconsistent between affines and coordinates.
    RuntimeError
        Batch sizes are incompatible between x, affine_x_el2coords, and sample_coords.
    """
    # Import here to make interpol an optional dependency.
    import interpol

    # Determine spatial dimensions based on the homogeneous affine matrix and coordinate
    # shapes.
    if (
        affine_x_el2coords.shape[-1] == 3
        and affine_x_el2coords.shape[-2] == 3
        and sample_coords.shape[-1] == 2
    ):
        spatial_dims = 2
    elif (
        affine_x_el2coords.shape[-1] == 4
        and affine_x_el2coords.shape[-2] == 4
        and sample_coords.shape[-1] == 3
    ):
        spatial_dims = 3
    else:
        raise RuntimeError(
            "Invalid or incompatible affine and coordinates shapes: "
            + f"{affine_x_el2coords.shape = }, {sample_coords.shape = }"
        )

    if spatial_dims == 2:
        d = mrinr.utils.ensure_image_channels(x, batch=True)
        coords, aff_x_el2coords = mrinr.coords._canonicalize_coords_2d_affine(
            sample_coords, affine=affine_x_el2coords
        )
    elif spatial_dims == 3:
        d = mrinr.utils.ensure_vol_channels(x, batch=True)
        coords, aff_x_el2coords = mrinr.coords._canonicalize_coords_3d_affine(
            sample_coords, affine=affine_x_el2coords
        )
    # Check for a common batch size.
    if d.shape[0] != coords.shape[0] or d.shape[0] != aff_x_el2coords.shape[0]:
        raise RuntimeError(
            "Input batch sizes are incompatible: "
            + f"x shape {tuple(x.shape)} inferred batch size {d.shape[0]}, "
            + f"affine shape {tuple(affine_x_el2coords.shape)} "
            + f"inferred batch size {aff_x_el2coords.shape[0]}, and "
            + f"sample coordinates shape {tuple(sample_coords.shape)} "
            + f"inferred batch size {coords.shape[0]}."
        )

    # interpol expects the sampling grid to be in element coordinates, so bring the
    # physical coordinate grid back to (pi|vo)xel space.
    x_el_coords = mrinr.coords.transform_coords(
        coords, mrinr.coords.inv_affine(aff_x_el2coords)
    )
    if not torch.is_floating_point(d):
        d = d.to(torch.promote_types(torch.float32, x_el_coords.dtype))

    # Standardize the interpolation type string.
    if isinstance(interpolation, str):
        m = interpolation.strip().lower()
        if ("nearest" in m) or (m == "nn"):
            interpolation_ = "nearest"
        elif "linear" in m:
            interpolation_ = "linear"
        elif "cubic" in m:
            interpolation_ = "cubic"
        else:
            interpolation_ = m
    else:
        interpolation_ = interpolation
    # Standardize the boundary padding string.
    if isinstance(bound, str):
        b = bound.strip().lower()
        bound_ = "zero" if "zero" in b else b
    else:
        bound_ = bound
    samples: torch.Tensor = interpol.grid_pull(
        d,
        x_el_coords,
        interpolation=interpolation_,
        bound=bound_,
        extrapolate=True,
        prefilter=prefilter,
    )

    # Change samples back to the input dtype only if the input was not a floating point,
    # and the interpolation was nearest-neighbor.
    if not torch.is_floating_point(x) and interpolation_ in {"nearest", 0}:
        samples = samples.round().to(x.dtype)

    # If the interpolation order was > 1, then the result may contain ringing artifacts.
    # So if indicated, clamp the samples to the input range.
    if clamp:
        if (isinstance(interpolation_, int) and interpolation_ > 1) or (
            isinstance(interpolation_, str)
            and interpolation_ not in {"nearest", "linear"}
        ):
            if spatial_dims == 2:
                p = "b c x y -> b c 1 1"
            elif spatial_dims == 3:
                p = "b c x y z -> b c 1 1 1"
            min_ = einops.reduce(d, p, "min")
            max_ = einops.reduce(d, p, "max")
            samples = torch.clamp(samples, min=min_, max=max_)

    if spatial_dims == 2:
        samples = mrinr.utils.undo_image_channels(samples, orig_x=x, strict=False)
    elif spatial_dims == 3:
        samples = mrinr.utils.undo_vol_channels(samples, orig_x=x, strict=False)

    return samples


@torch.no_grad
def grid_resample_scipy(
    x: torch.Tensor,
    affine_x_el2coords: mrinr.typing.AnyHomogeneousAffineSD,
    sample_coords: mrinr.typing.AnyCoordSD,
    order: int | str,
    padding_mode: str,
    padding_cval: float = 0.0,
    clamp: bool = False,
    prefilter: bool = True,
    allow_cpu_move: bool = True,
) -> torch.Tensor:
    """Resample an image or volume using the `scipy.ndimage.map_coordinates()`.

    *Note* This function relies on scipy's `ndimage.map_coordinates()` function, which
    does not support many pytorch features. It is constrained in the following ways:

    - The input tensors *must* have a batch size of 1.
    - The input tensors will be moved to the CPU if `allow_cpu_move` is True,
        otherwise a ValueError is raised.
    - Gradients will not be backpropagated.
    - Image/volume channels will be processed over a python loop instead of a
        vectorized function, so as to avoid interpolating between channels. This
        may be slow for data with many channels.

    The `order`, `padding_mode`, and `padding_cval` parameters correspond to the
    `scipy.ndimage.map_coordinates()` function parameters described below. The `order`
    parameters also accept a subset of the `mode` and 'interpolation' parameters from
    `grid_resample_torch()` and `grid_resample_interpol()`, respectively, that will map
    to the appropriate `scipy.ndimage.map_coordinates()` parameters. Similarly, the
    `padding_mode` parameter accepts a subset of the `padding_mode` and `bound`, when
    possible.

    Parameters
    ----------
    x : torch.Tensor
        Input image or volume to resample.
    affine_x_el2coords : mrinr.typing.AnyHomogeneousAffineSD
        Homogeneous affine that transforms x element coordinates to sample coordinates.
    sample_coords : mrinr.typing.AnyCoordSD
        Coordinates at which to sample x.
    order : int | str
        The order of the spline interpolation, in the range 0-5 if an integer.

    padding_mode : str
        Padding mode for resampling, see the description below.

        From scipy documentation:
        The padding_mode parameter determines how the input array is extended
        beyond its boundaries. Behavior for each valid value is as follows (see additional
        plots and details on boundary modes):

        - `reflect` (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last pixel. This
            mode is also sometimes referred to as half-sample symmetric.

        - `grid-mirror`
            This is a synonym for `reflect`.

        - `constant` (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the same
            constant value, defined by the cval parameter. No interpolation is performed
            beyond the edges of the input.

        - `grid-constant` (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the same
            constant value, defined by the cval parameter. Interpolation occurs for
            samples outside the input`s extent as well.

        - `nearest` (a a a a | a b c d | d d d d)
            The input is extended by replicating the last pixel.

        - `mirror` (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last pixel.
            This mode is also sometimes referred to as whole-sample symmetric.

        - `grid-wrap` (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.

        - `wrap` (d b c d | a b c d | b c a b)
            The input is extended by wrapping around to the opposite edge, but in a way
            such that the last point and initial point exactly overlap. In this case it
            is not well defined which sample will be chosen at the point of overlap.

    padding_cval : float, optional
        Value to fill past edges of input if mode is `constant`. Default is 0.0.

    prefilter : bool, optional
        Prefilter with scipy.ndimage.spline_filter() before interp., by default True

        From scipy documentation:
        Determines if the input array is prefiltered with spline_filter
        before interpolation. The default is True, which will create a temporary
        float64 array of filtered values if order > 1. If setting this to False, the
        output will be slightly blurred if order > 1, unless the input is prefiltered,
        i.e. it is the result of calling spline_filter on the original input.

    allow_cpu_move : bool, optional
        Allow the input tensors to be moved to the CPU, by default True

    Returns
    -------
    torch.Tensor
        x resampled at sample_coords.

    Raises
    ------
    RuntimeError
        Spatial dims are not 2 or 3, or inconsistent between affines and coordinates.
    RuntimeError
        Batch sizes are incompatible between x, affine_x_el2coords, and sample_coords.
    ValueError
        One or more inputs have a batch size > 1.
    ValueError
        Tensors are not on cpu, and allow_cpu_move is False.
    """

    # Determine spatial dimensions based on the homogeneous affine matrix and coordinate
    # shapes.
    if (
        affine_x_el2coords.shape[-1] == 3
        and affine_x_el2coords.shape[-2] == 3
        and sample_coords.shape[-1] == 2
    ):
        spatial_dims = 2
    elif (
        affine_x_el2coords.shape[-1] == 4
        and affine_x_el2coords.shape[-2] == 4
        and sample_coords.shape[-1] == 3
    ):
        spatial_dims = 3
    else:
        raise RuntimeError(
            "Invalid or incompatible affine and coordinates shapes: "
            + f"{affine_x_el2coords.shape = }, {sample_coords.shape = }"
        )

    if spatial_dims == 2:
        d = mrinr.utils.ensure_image_channels(x, batch=True)
        coords, aff_x_el2coords = mrinr.coords._canonicalize_coords_2d_affine(
            sample_coords, affine=affine_x_el2coords
        )
    elif spatial_dims == 3:
        d = mrinr.utils.ensure_vol_channels(x, batch=True)
        coords, aff_x_el2coords = mrinr.coords._canonicalize_coords_3d_affine(
            sample_coords, affine=affine_x_el2coords
        )
    # Check for a common batch size.
    if d.shape[0] != coords.shape[0] or d.shape[0] != aff_x_el2coords.shape[0]:
        raise RuntimeError(
            "Input batch sizes are incompatible: "
            + f"x shape {tuple(x.shape)} inferred batch size {d.shape[0]}, "
            + f"affine shape {tuple(affine_x_el2coords.shape)} "
            + f"inferred batch size {aff_x_el2coords.shape[0]}, and "
            + f"sample coordinates shape {tuple(sample_coords.shape)} "
            + f"inferred batch size {coords.shape[0]}."
        )
    # Check for batch sizes > 1, which is not supported by `map_coordinates()`.
    if d.shape[0] > 1:
        raise ValueError(
            "Input tensors must have a batch size of 1, but got "
            f"{d.shape[0] = } for x, {coords.shape[0] = } for sample_coords, and "
            f"{aff_x_el2coords.shape[0] = } for affine_x_el2coords."
        )
    # Check if the input tensors are on the CPU, and move them there if allowed.
    input_device = d.device
    input_dtype = d.dtype
    if input_device.type != "cpu":
        if allow_cpu_move:
            d = d.cpu()
            coords = coords.cpu()
        else:
            raise ValueError(
                "Input tensors are not allowed to be moved to the CPU, but got input "
                f"data with device {input_device}."
            )

    # map_coordinates expects the sampling grid to be in element coordinates, so bring the
    # physical coordinate grid back to (pi|vo)xel space.
    x_el_coords = mrinr.coords.transform_coords(
        coords, mrinr.coords.inv_affine(aff_x_el2coords)
    )
    if not torch.is_floating_point(d):
        d = d.to(torch.promote_types(torch.float32, x_el_coords.dtype))

    # Standardize the interpolation order.
    if isinstance(order, str):
        o = order.strip().lower()
        if ("nearest" in o) or (o == "nn"):
            order_ = 0
        elif "linear" in o:
            order_ = 1
        elif "cubic" in o:
            order_ = 3
        elif "quadratic" in o:
            order_ = 2
        elif "fourth" == o:
            order_ = 4
        elif "fifth" == o:
            order_ = 5
        elif o.isdigit():
            order_ = int(o)
        else:
            raise ValueError(
                f"Invalid interpolation order {order = }, expected an integer or a "
                "string that matches one of the supported interpolation orders."
            )
    else:
        order_ = int(order)

    # Standardize the boundary padding string.
    mode = str(padding_mode).lower().strip()
    cval_ = padding_cval
    # Zero padding.
    if "zero" in mode:
        mode_ = "constant"
        # Override cval if it was given.
        cval_ = 0.0
    # Pytorch 'border' or interpol 'replicate'.
    elif "border" in mode or "replicat" in mode:
        mode_ = "nearest"
    # interpol 'dct1'.
    elif mode == "dct1":
        mode_ = "mirror"
    # interpol 'dct2'
    elif mode == "dct2":
        mode_ = "reflect"
    # interpol 'dft'
    elif mode == "dft":
        mode_ = "wrap"
    # Otherwise, assume it should be passed as is.
    else:
        mode_ = mode

    # Loop over channels and sample.
    samples = list()
    # Move to numpy and rearrange coordinate dimension to the order expected by
    # map_coordinates.
    x_el_coords = einops.rearrange(
        x_el_coords.numpy(), "1 x y ... coord -> coord x y ..."
    )
    d = d.squeeze(0).numpy()
    n_channels = d.shape[0]
    for i in range(n_channels):
        # d[i] should be 2D or 3D.
        s_i = scipy.ndimage.map_coordinates(
            d[i],
            coordinates=x_el_coords,
            order=order_,
            mode=mode_,
            cval=cval_,
            prefilter=prefilter,
        )
        samples.append(s_i)
    # Bring samples back to a pytorch Tensor, and reconstruct both batch and channel
    # dimensions.
    samples = (
        torch.from_numpy(np.stack(samples, axis=0))
        .to(dtype=input_dtype, device=input_device)
        .unsqueeze(0)
    )

    # # Change samples back to the input dtype only if the input was not a floating point,
    # # and the interpolation was nearest-neighbor.
    # if not torch.is_floating_point(x) and order_ == 0:
    #     samples = samples.round().to(x.dtype)

    # If the interpolation order was > 1, then the result may contain ringing artifacts.
    # So if indicated, clamp the samples to the input range.
    if clamp:
        if order_ > 1:
            if spatial_dims == 2:
                p = "b c x y -> b c 1 1"
            elif spatial_dims == 3:
                p = "b c x y z -> b c 1 1 1"
            min_ = einops.reduce(d, p, "min")
            max_ = einops.reduce(d, p, "max")
            samples = torch.clamp(samples, min=min_, max=max_)

    if spatial_dims == 2:
        samples = mrinr.utils.undo_image_channels(samples, orig_x=x, strict=False)
    elif spatial_dims == 3:
        samples = mrinr.utils.undo_vol_channels(samples, orig_x=x, strict=False)

    return samples


def grid_resample(
    x: torch.Tensor,
    affine_x_el2coords: mrinr.typing.AnyHomogeneousAffineSD,
    sample_coords: mrinr.typing.AnyCoordSD,
    mode_or_interpolation: str | int | list[str | int],
    padding_mode_or_bound: str | list[str],
    interp_lib: Literal["torch", "interpol", "scipy"] = "torch",
    clamp: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Sample points from x according to sample_coords with interpolation.

    This function wraps the pytorch grid_sample() function, interpol's grid_pull(),
    and scipy's map_coordinates() to provide a unified interface for resampling
    functions. For 'torch' and 'interpol' interpolation libraries, gradients are
    properly backpropagated, and all Tensors remain on the same device as x.

    Parameters
    ----------
    x : torch.Tensor
        Input spatial data, either a 2D image or 3D volume.
    affine_x_el2coords : mrinr.typing.AnyHomogeneousAffineSD
        Homogeneous affine matrix describing the coordinates in x.
    sample_coords : mrinr.typing.AnyCoordSD
        Coordinates in the space defined by the affine matrix at which to sample x.
    mode_or_interpolation : str | int | list[str  |  int]
        Interpolation mode or order for resampling, specific to the interpolation library.

        See `torch.nn.functional.grid_sample()` for modes available in pytorch, and
        `interpol.grid_pull()` for interpolation orders available in interpol.
    padding_mode_or_bound : str | list[str]
        Padding mode or boundary condition specific to the interpolation library.

        For torch, the available options are:
        'zeros'                       :  0  0  0  |  a  b  c  d  |  0  0  0
        'border'                      :  a  a  a  |  a  b  c  d  |  d  d  d
        'reflection'                  :  d  c  b  |  a  b  c  d  |  c  b  a

        For interpol, the available options are:
        'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
        'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
        'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
        'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
        'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
        'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
        'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0

    interp_lib : Literal[&quot;torch&quot;, &quot;interpol&quot;, &quot;scipy&quot;], optional
        Interpolation library to use, by default "torch"
    clamp : bool, optional
        Clamp the output to the intensity range of x, by default False

        This option can mitigate ringing artifacts from interpolation orders > 1
        (linear).

    Returns
    -------
    torch.Tensor
        Tensor x resampled at sample_coords.

    Raises
    ------
    ValueError
        Interpolation library is not 'torch', 'interpol', or 'scipy'.
    """
    l = str(interp_lib).strip().lower()
    if "torch" in l:
        ret = grid_resample_torch(
            x,
            affine_x_el2coords=affine_x_el2coords,
            sample_coords=sample_coords,
            mode=mode_or_interpolation,
            padding_mode=padding_mode_or_bound,
            clamp=clamp,
            **kwargs,
        )
    elif l == "interpol":
        ret = grid_resample_interpol(
            x,
            affine_x_el2coords=affine_x_el2coords,
            sample_coords=sample_coords,
            interpolation=mode_or_interpolation,
            bound=padding_mode_or_bound,
            clamp=clamp,
            **kwargs,
        )
    elif l == "scipy" or l == "numpy" or l == "np":
        ret = grid_resample_scipy(
            x,
            affine_x_el2coords=affine_x_el2coords,
            sample_coords=sample_coords,
            order=mode_or_interpolation,
            padding_mode=padding_mode_or_bound,
            clamp=clamp,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Invalid interpolation library {interp_lib = }, expected 'torch', "
            "'interpol', or 'scipy'."
        )

    return ret


def resize(
    x: mrinr.typing.AnySpatialDataSD,
    affine_x_el2coords: mrinr.typing.AnyHomogeneousAffineSD,
    target_spatial_shape: tuple[int, ...],
    centered: bool = True,
    **resample_kwargs,
) -> tuple[mrinr.typing.AnySpatialDataSD, mrinr.typing.AnyHomogeneousAffineSD]:
    spatial_dims = affine_x_el2coords.shape[-1] - 1
    in_shape = tuple(x.shape[-spatial_dims:])
    target_shape = tuple(target_spatial_shape)[-spatial_dims:]

    resized_affine, target_coord_grid = mrinr.coords.resize_affine(
        affine_x_el2coords=affine_x_el2coords,
        in_spatial_shape=in_shape,
        target_spatial_shape=target_shape,
        centered=centered,
    )
    y = grid_resample(
        x,
        affine_x_el2coords=affine_x_el2coords,
        sample_coords=target_coord_grid,
        **resample_kwargs,
    )

    return y, resized_affine
