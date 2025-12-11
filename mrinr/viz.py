# -*- coding: utf-8 -*-
import itertools
import math
from typing import List, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from mrinr._lazy_loader import LazyLoader

# Make mrinr lazy load to avoid circular imports.
mrinr = LazyLoader("mrinr", globals(), "mrinr")

VIZ_NIFTI_ORIENTATION = "IPR"
VIZ_ORNT_CODE = ("I", "P", "R")


def is_global_colorbar_safe_from_im_samples(
    *im_lists: list[np.ndarray],
    abs_lower_bound_tol: Optional[float] = None,
    abs_upper_bound_tol: Optional[float] = None,
    rel_lower_bound_tol: Optional[float] = None,
    rel_upper_bound_tol: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    exclude_0s=False,
):
    raise NotImplementedError()
    ranges = list()
    mins = list()
    maxes = list()
    for im_list in im_lists:
        s = np.concatenate([np.asarray(im).flatten() for im in im_list])
        if exclude_0s:
            s = s[s != 0]
        min_ = s.min()
        max_ = s.max()
        range_ = max_ - min_
        mins.append(min_)
        maxes.append(max_)
        ranges.append(range_)

    ranges = np.asarray(ranges)
    mins = np.asarray(mins)
    maxes = np.asarray(maxes)


def plot_fodf_3d(theta, phi, sphere_vals, fig=None, **plot_trisurf_kwargs):
    theta = torch.Tensor(theta)
    phi = torch.Tensor(phi)
    sphere_vals = torch.Tensor(sphere_vals)
    if (
        (theta.ndim > 1 and (theta.numel() / theta.shape[-1]) > 1)
        or (phi.ndim > 1 and (phi.numel() / phi.shape[-1]) > 1)
        or (sphere_vals.ndim > 1 and (sphere_vals.numel() / sphere_vals.shape[-1]) > 1)
    ):
        raise ValueError(
            "Plotting can only accept 1 sphere to plot, got",
            f"{tuple(theta.shape)}, {tuple(phi.shape)}, {tuple(sphere_vals.shape)}",
        )
    elif (theta.numel() != phi.numel()) or (phi.numel() != sphere_vals.numel()):
        raise ValueError(
            "Coordinates and function values must have the same number of elements",
            f"got {theta.numel()}, {phi.numel()}, {sphere_vals.numel()}",
        )

    vals = sphere_vals.detach().cpu().numpy().flatten()
    r = (vals - vals.min()) / (vals - vals.min()).max()
    r = vals / vals.sum()

    assert (theta.min() >= 0).all() and (theta.max() <= torch.pi).all()
    assert (phi.min() > -torch.pi).all() and (phi.max() <= torch.pi).all()
    zyx = mrinr.tract.local.unit_sphere2zyx(theta, phi)
    x = r * zyx[:, 2].detach().cpu().numpy().flatten()
    y = r * zyx[:, 1].detach().cpu().numpy().flatten()
    z = r * zyx[:, 0].detach().cpu().numpy().flatten()
    theta = theta.detach().cpu().numpy().flatten()
    phi = phi.detach().cpu().numpy().flatten()

    if fig is None:
        fig = plt.figure(dpi=120)
    ax = fig.add_subplot(projection="3d")
    tri = mpl.tri.Triangulation(phi, theta)
    ax.plot_trisurf(
        x,
        y,
        z,
        triangles=tri.triangles,
        **{**dict(cmap="gnuplot", alpha=1, linewidth=0), **plot_trisurf_kwargs},
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig


def plot_sphere_fn_vals(
    theta, phi, fn_vals, subplots_kwargs: dict = dict(), **scatter_kwargs
):
    zyx = mrinr.tract.local.unit_sphere2zyx(theta, phi)
    x = zyx[:, 2].detach().cpu().numpy().flatten()
    y = zyx[:, 1].detach().cpu().numpy().flatten()
    z = zyx[:, 0].detach().cpu().numpy().flatten()
    vals = fn_vals.detach().cpu().numpy().flatten()
    vmax = vals.max()
    vmin = -vmax
    fig, axs = plt.subplots(
        nrows=1, ncols=2, **{**{"dpi": 120, "figsize": (7, 3.5)}, **subplots_kwargs}
    )
    ax = axs[0]
    distance_from_xy_plane_vals = np.copy(vals)
    distance_from_xy_plane_vals[z < 0] = -vals[z < 0]
    ax.scatter(
        x,
        y,
        c=distance_from_xy_plane_vals,
        **{**{"vmin": vmin, "vmax": vmax, "cmap": "coolwarm"}, **scatter_kwargs},
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax = axs[1]
    distance_from_xz_plane_vals = np.copy(vals)
    distance_from_xz_plane_vals[y < 0] = -vals[y < 0]
    ax.scatter(
        x,
        z,
        c=distance_from_xz_plane_vals,
        **{**{"vmin": vmin, "vmax": vmax, "cmap": "coolwarm"}, **scatter_kwargs},
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    return fig


def plot_im_grid(
    *ims,
    nrows: int = 3,
    title: Optional[str] = None,
    row_headers: Optional[List[str]] = None,
    col_headers: Optional[List[str]] = None,
    colorbars: Optional[str] = None,
    fig=None,
    **imshow_kwargs,
):
    """Plot sequence of 2d arrays as a grid of images with optional titles & colorbars.

    Parameters
    ----------
    ims: sequence
        Sequence of numpy ndarrays or pytorch Tensors to plot into a grid.
    nrows : int, optional
        Number of rows in the grid, by default 3
    title : Optional[str], optional
        The `suptitle` of the image grid, by default None
    row_headers : Optional[List[str]], optional
        Titles for each row, by default None
    col_headers : Optional[List[str]], optional
        Titles for each column., by default None
    colorbars : Optional[str], optional
        Set the type of colorbar and intensity normalization to use, by default None

        Valid options are:
            None - no colorbar or intensity normalization.
            "global" - one colorbar is created for the entire grid, and all images are
                normalized to have color intensity ranges match.
            "each" - every image has its own colorbar with no intensity normalization.
            "col", "cols", "column", "columns" - Every column is normalized and
                given a colorbar.
            "row", "rows" - Every row is normalized and given a colorbar.

    fig : Figure, optional
        Figure to plot into, by default None
    imshow_kwargs : dict
        Kwargs to pass to the `.imshow()` function call of each image.

    Returns
    -------
    Figure

    Raises
    ------
    ValueError
        Invalid option value for `colorbars`
    """

    # AX_TITLE_SIZE_PERC = 0.05
    # SUPTITLE_SIZE_PERC = 0.1
    # AX_CBAR_SIZE_PERC = 0.1
    # EACH_CBAR_SUBPLOT_SIZE_PERC = "7%"

    if fig is None:
        fig = plt.gcf()

    ims = list(ims)
    # Canonical form of ims.
    if len(ims) == 1 and isinstance(ims[0], (list, tuple, set)):
        ims = ims[0]

    _expand_ims = list()
    for im in ims:
        assert isinstance(im, np.ndarray) or torch.is_tensor(im)
        # If batched, then collapse the batch dim(s) and count each as its own image.
        if len(im.shape) >= 3:
            _expand_ims.extend(list(im.reshape(-1, im.shape[-2], im.shape[-1])))
        else:
            _expand_ims.append(im)
    ims = _expand_ims
    # # elif len(ims) == 1 and (isinstance(ims[0], np.ndarray) or torch.is_tensor(ims[0])):
    #     # If the tensor/ndarray is batched.
    #     if len(ims[0].shape) >= 3:
    #         ims = list(ims[0])
    #     else:
    #         ims = [ims[0]]

    for i, im in enumerate(ims):
        if torch.is_tensor(im):
            ims[i] = im.detach().cpu().numpy()
        ims[i] = ims[i].astype(float)
    ncols = math.ceil(len(ims) / nrows)
    # Canonical representation of image labels.
    row_headers = (
        row_headers if row_headers is not None else list(itertools.repeat(None, nrows))
    )
    col_headers = (
        col_headers if col_headers is not None else list(itertools.repeat(None, ncols))
    )

    if len(row_headers) != nrows:
        raise RuntimeError(
            f"ERROR: Number of row headers {len(row_headers)} != number of rows {nrows}"
        )
    if len(col_headers) != ncols:
        raise RuntimeError(
            f"ERROR: Number of row headers {len(col_headers)} != number of rows {ncols}"
        )
    # Canonical colorbar setting values.
    cbars = colorbars.casefold() if colorbars is not None else colorbars
    cbars = "col" if cbars in {"column", "columns", "cols", "col"} else cbars
    cbars = "row" if cbars in {"row", "rows"} else cbars
    cbars = "global" if cbars in {"global", "one"} else cbars
    if cbars not in {"global", "each", "row", "col", None}:
        raise ValueError(f"ERROR: Colorbars value {colorbars} not valid.")

    # Pad im list with None objects.
    pad_ims = list(
        itertools.islice(itertools.chain(ims, itertools.repeat(None)), nrows * ncols)
    )
    ims_grid = list()
    for i in range(0, nrows * ncols, ncols):
        ims_grid.append(pad_ims[i : i + ncols])

    # Calculate grid shape in number of pixels/array elements in both directions.
    # row x col x 2 = (x_width, y_height)
    element_shapes = list()
    for r in ims_grid:
        row_i_shapes = list()
        for element in r:
            if element is not None:
                w = element.shape[1]
                h = element.shape[0]
            else:
                w = 1
                h = 1
            row_i_shapes.append((w, h))
        element_shapes.append(row_i_shapes)
    element_shapes = np.asarray(element_shapes)
    max_width = np.max(np.sum(element_shapes[..., 0], axis=1))
    max_height = np.max(np.sum(element_shapes[..., 1], axis=0))

    # Correct fig size according to the size of the actual arrays.
    grid_pix_dim = [max_height, max_width]
    # If a global colorbar is used, the col width needs to be extende by the colorbar
    # width, currently set to ~ 10% the total axis size.
    if cbars == "global":
        grid_pix_dim[1] = round(grid_pix_dim[1] * 1.1)
    # row and col colorbars add about 7% to their respective axes.
    elif cbars == "row":
        grid_pix_dim[1] = round(grid_pix_dim[1] * 1.07)
    elif cbars == "col":
        grid_pix_dim[0] = round(grid_pix_dim[0] * 1.07)
    fig_hw_ratio = grid_pix_dim[0] / grid_pix_dim[1]
    fig.set_figheight(fig.get_figwidth() * fig_hw_ratio)

    # Create gridspec.
    grid = mpl.gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        # width_ratios=np.asarray(col_widths) / sum(col_widths),
        # height_ratios=np.asarray(row_heights) / sum(row_heights),
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
        wspace=0.01,
        hspace=0.01,
    )

    # Keep track of each image's min and max values.
    min_max_vals = np.zeros((nrows, ncols, 2))
    # Keep track of each created axis in the grid.
    axs = list()
    # Keep track of the highest subplot position in order to avoid overlap with the
    # suptitle.
    max_subplot_height = 0
    # Step through the grid.
    for i_row, (ims_row_i, row_i_header) in enumerate(zip(ims_grid, row_headers)):
        row_axs = list()
        for j_col, (im, col_j_header) in enumerate(zip(ims_row_i, col_headers)):
            # If no im was given here, skip everything in this loop.
            if im is None:
                continue

            # Create Axes object at the grid location.
            ax = fig.add_subplot(grid[i_row, j_col])
            ax.imshow(im, **imshow_kwargs)
            # Set headers.
            if row_i_header is not None and ax.get_subplotspec().is_first_col():
                ax.set_ylabel(row_i_header)
            if col_j_header is not None and ax.get_subplotspec().is_first_row():
                ax.set_xlabel(col_j_header)
                ax.xaxis.set_label_position("top")
            # Remove pixel coordinate axis ticks.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            # Update highest subplot to put the `suptitle` later on.
            max_subplot_height = max(
                max_subplot_height, ax.get_position(original=False).get_points()[1, 1]
            )
            # Update min and max im values.
            min_max_vals[i_row, j_col] = (im.min(), im.max())
            row_axs.append(ax)
        axs.append(row_axs)

    # Handle colorbar creation.
    # If there should only be one colorbar for the entire grid.
    if cbars == "global":
        min_v = min_max_vals[:, :, 0].min()
        max_v = min_max_vals[:, :, 1].max()
        color_norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
        cmap = None
        for ax_row in axs:
            for ax in ax_row:
                disp_im = ax.get_images()[0]
                disp_im.set(norm=color_norm)
                cmap = disp_im.cmap if cmap is None else cmap

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap),
            ax=list(itertools.chain.from_iterable(axs)),
            location="right",
            fraction=0.1,
            pad=0.03,
        )
    # If colorbar setting is row, col, or each.
    elif cbars is not None:
        # Step through all subplots in the grid.
        for i_row, ax_row in enumerate(axs):
            for j_col, ax in enumerate(ax_row):
                # Determine value range depending on the cbars setting.
                if cbars == "row":
                    min_v = min_max_vals[i_row, :, 0].min()
                    max_v = min_max_vals[i_row, :, 1].max()
                elif cbars == "col":
                    min_v = min_max_vals[:, j_col, 0].min()
                    max_v = min_max_vals[:, j_col, 1].max()
                elif cbars == "each":
                    min_v = min_max_vals[i_row, j_col, 0]
                    max_v = min_max_vals[i_row, j_col, 1]

                # Get AxesImage object (the actual image plotted) from the subplot.
                disp_im = ax.get_images()[0]
                # Set up color/cmap scaling.
                color_norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
                disp_im.set(norm=color_norm)
                color_mappable = mpl.cm.ScalarMappable(
                    norm=color_norm, cmap=disp_im.cmap
                )

                # Use the (somewhat new) AxesDivider utility to divide the subplot
                # and add a colorbar with the corresponding min/max values.
                # See
                # <https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html>
                if cbars == "row":
                    if ax.get_subplotspec().is_last_col():
                        ax_div = make_axes_locatable(ax)
                        cax = ax_div.append_axes("right", size="7%", pad="4%")
                        fig.colorbar(color_mappable, cax=cax, orientation="vertical")
                elif cbars == "col":
                    if ax.get_subplotspec().is_last_row():
                        ax_div = make_axes_locatable(ax)
                        cax = ax_div.append_axes("bottom", size="7%", pad="4%")
                        cbar = fig.colorbar(
                            color_mappable,
                            cax=cax,
                            orientation="horizontal",
                            format="%.4e",
                        )
                        cax.xaxis.set_ticks_position("bottom")
                        cbar.ax.tick_params(rotation=90)
                elif cbars == "each":
                    ax_div = make_axes_locatable(ax)
                    cax = ax_div.append_axes("right", size="7%", pad="4%")
                    fig.colorbar(color_mappable, cax=cax, orientation="vertical")

    if title is not None:
        fig.suptitle(title, y=max_subplot_height + 0.05, verticalalignment="bottom")

    return fig


def _float_or_int_slice_idx(slice_idx, dim_size, reverse=False):
    # If slice idx was a fraction, interpret that as a percent of the total
    # dim size.
    if isinstance(slice_idx, float) and slice_idx >= 0.0 and slice_idx <= 1.0:
        idx = math.floor(slice_idx * (dim_size - 1))
    else:
        idx = math.floor(slice_idx)
    if reverse:
        idx = (dim_size - 1) - idx
    return idx


def plot_1_ch_multi_slice_compare_vols(
    *vols: Sequence[Union[torch.Tensor, np.ndarray]],
    axial_slice_idx: Optional[Union[tuple[float, ...], float]] = 0.5,
    saggital_slice_idx: Optional[Union[tuple[float, ...], float]] = 0.5,
    coronal_slice_idx: Optional[Union[tuple[float, ...], float]] = 0.5,
    reorient_ras_for_viz=True,
    title: Optional[str] = None,
    vol_labels: Optional[List[str]] = None,
    slice_labels: Optional[List[str]] = None,
    colorbars: Optional[str] = None,
    fig=None,
    cmap="gray",
    **imshow_kwargs,
):
    """Plot slices of multiple volumes containing 1 channel, with viz reorientation.

    All input volumes are assumed to be in RAS+ orientation.

    Parameters
    ----------
    vols : Sequence[Union[torch.Tensor, np.ndarray]]
        Sequence of volumes to plot, either pytorch Tensors or numpy ndarrays.

        Volumes must have the shapes '[batch] x [channel=1] x X x Y x Z', where 'batch'
        and 'channel' are optional, but channel must always equal 1. If batch is > 1,
        each volume in the batch is considered as another row.
    axial_slice_idx : Optional[Union[tuple[float, ...], float]], optional
        Axial slice indices as vox index or percent along axis, by default 0.5

        If a slice index is an integer, it will be used as a 0-based voxel index to
        locate the desired slice. If the index is a float, it will be considered as a
        "percentage to move along the axis." For example, if slice_idx is 0.45 and a
        volume contains 50 slices along that axis, then the voxel index would be
        "floor(22.05) = 22".

        Slice indices can also be None (don't plot any slice on this axis), or a tuple
        of indices (plot multiple slices along this axis).
    saggital_slice_idx : Optional[Union[tuple[float, ...], float]], optional
        Saggital slice indices as vox index or percent along axis, by default 0.5

        Similar to `axial_slice_idx`, but in the saggital plane.
    coronal_slice_idx : Optional[Union[tuple[float, ...], float]], optional
        Coronal slice indices as vox index or percent along axis, by default 0.5

        Similar to `axial_slice_idx`, but in the coronal plane.
    reorient_ras_for_viz : bool, optional
        Reorient volumes from RAS to a more intuitive viz orientation, by default True

        If set, all input volumes are assumed to be in RAS, and will be reoriented to
        IPR (superior->inferior, anterior->posterior, left->right).
    title : Optional[str], optional
        Title given to the figure, by default None
    vol_labels : Optional[List[str]], optional
        Row labels given for each volume in `vols`, by default None

        If vol_labels is not None, then it expects a list of labels with length equal
        to the number of volumes in `vols`, including any batch dimensions.
    slice_labels : Optional[List[str]], optional
        Column labels given for each slice index, by default None

        If slice_labels is not None, it expects a list of labels with length equal
        to the number of given slice indices given by `axial_slice_idx`,
        `saggital_slice_idx`, and `coronal_slice_idx`.
    colorbars : Optional[str], optional
        Colorbar(s) to plot, by default None

        See `plot_im_grid()` for details.
    fig : Figure, optional
        Matplotlib Figure to plot onto, by default None
    cmap : str, optional
        Matplotlib cmap passed to `imshow()`, by default "gray"

    Returns
    -------
    Figure
    """

    viz_vols = list()
    vol_titles = list()
    for i, vol in enumerate(vols):
        if torch.is_tensor(vol):
            v = vol.detach().cpu().numpy()
        else:
            v = np.asarray(vol)
        if len(v.shape) == 3:
            v = v[np.newaxis]
        if len(v.shape) == 4:
            v = v[np.newaxis]
        assert (
            v.shape[1] == 1
        ), f"All volumes must have only 1 channel, got shape {v.shape}"
        # Bool arrays are not supported by nibabel, so convert to a uint8.
        if v.dtype == bool:
            v = v.astype(np.uint8)

        # Undo channel to just batch x spatial.
        v = v[:, 0]
        for j_b in range(v.shape[0]):
            v_j = v[j_b]
            # Assume all volumes are in RAS orientation, then transform them for better
            # visualization.
            if reorient_ras_for_viz:
                # We don't care about spacing, rotation, etc., just flip/90 deg rotate
                # to get a better view.
                v_im = nib.Nifti1Image(v_j, np.eye(4))
                v_j = mrinr.data.io.reorient_nib_im(
                    v_im, VIZ_NIFTI_ORIENTATION
                ).get_fdata()
            viz_vols.append(v_j)
            if vol_labels is not None and not isinstance(vol_labels, str):
                vol_titles.append(vol_labels[i])
            elif isinstance(vol_labels, str):
                vol_titles.append(vol_labels)
            elif vol_labels is None:
                vol_titles.append(None)

    ax_slices = (
        (axial_slice_idx,)
        if np.isscalar(axial_slice_idx) or (axial_slice_idx is None)
        else tuple(axial_slice_idx)
    )
    sag_slices = (
        (saggital_slice_idx,)
        if np.isscalar(saggital_slice_idx) or (saggital_slice_idx is None)
        else tuple(saggital_slice_idx)
    )
    cor_slices = (
        (coronal_slice_idx,)
        if (np.isscalar(coronal_slice_idx) or coronal_slice_idx is None)
        else tuple(coronal_slice_idx)
    )
    slice_idx_axis_tuple = tuple(
        filter(
            lambda s_axis: s_axis[0] is not None,
            zip(
                itertools.chain(ax_slices, sag_slices, cor_slices),
                itertools.chain(
                    itertools.repeat("ax", times=len(ax_slices)),
                    itertools.repeat("sag", times=len(sag_slices)),
                    itertools.repeat("cor", times=len(cor_slices)),
                ),
            ),
        )
    )
    if (slice_labels is not None) and (len(slice_labels) != len(slice_idx_axis_tuple)):
        raise ValueError(
            "ERROR: slice_labels must have 1 label for every requested slice_idx; "
            f"got {len(slice_labels)} slice_labels "
            f"and {len(slice_idx_axis_tuple)} slices."
        )

    flat_slices = list()
    for i_v, v in enumerate(viz_vols):
        for j_s, slice_idx_axis in enumerate(slice_idx_axis_tuple):
            # Find the location of this slice, whether it is an int index or a float
            # fraction index, and whether or not reorientation was done.
            slice_idx, slice_axis = slice_idx_axis
            if slice_axis == "ax":
                slice_dim_idx = 2
            elif slice_axis == "cor":
                slice_dim_idx = 1
            elif slice_axis == "sag":
                slice_dim_idx = 0

            # Handle possible flips/rotations for viz orientation.
            # RAS -> IPR
            if reorient_ras_for_viz:
                if slice_axis == "ax":
                    slice_dim_idx = 0
                    array_slice = _float_or_int_slice_idx(
                        slice_idx, v.shape[slice_dim_idx], reverse=True
                    )
                    array_slicer = (array_slice, slice(None), slice(None))
                elif slice_axis == "sag":
                    slice_dim_idx = 2
                    array_slice = _float_or_int_slice_idx(
                        slice_idx, v.shape[slice_dim_idx], reverse=True
                    )
                    array_slicer = (slice(None), slice(None), array_slice)
                elif slice_axis == "cor":
                    array_slice = _float_or_int_slice_idx(
                        slice_idx, v.shape[slice_dim_idx], reverse=True
                    )
                    array_slicer = (slice(None), array_slice, slice(None))
            else:
                array_slice = _float_or_int_slice_idx(
                    slice_idx, v.shape[slice_dim_idx], reverse=False
                )
                if slice_axis == "ax":
                    array_slicer = (slice(None), slice(None), array_slice)
                elif slice_axis == "cor":
                    array_slicer = (slice(None), array_slice, slice(None))
                elif slice_axis == "sag":
                    array_slicer = (array_slice, slice(None), slice(None))
            flat_slices.append(v[array_slicer])

    row_headers = vol_titles if not all([t is None for t in vol_titles]) else None

    return plot_im_grid(
        *flat_slices,
        nrows=len(viz_vols),
        title=title,
        row_headers=row_headers,
        col_headers=slice_labels,
        colorbars=colorbars,
        fig=fig,
        cmap=cmap,
        **imshow_kwargs,
    )


def plot_vol_slices(
    *vols: Sequence[Union[torch.Tensor, np.ndarray]],
    slice_idx=(0.5, 0.5, 0.5),
    title: Optional[str] = None,
    vol_labels: Optional[List[str]] = None,
    slice_labels: Optional[List[str]] = None,
    channel_labels: Optional[List[str]] = None,
    colorbars: Optional[str] = None,
    fig=None,
    **imshow_kwargs,
):
    """Plot 2D slices of a full 3D volume, supports multi-channel and multi-batch vols.

    Parameters
    ----------
    slice_idx : tuple, optional
        Tuple of ints, floats, or Nones to select vol slices, by default (0.5, 0.5, 0.5)

        Indices that correspond to each spatial dimension in the input volumes. Given
            the slice value `s = slice_idx[i]`:
                * If `s` is a float, then the spatial dimension `i` will be sliced at
                  (approximately) `s`% of the size of that dimension. For example, if
                  `s` is 0.5, then the slice that will be visualized will be 50% of the
                  way through dimension `i`.
                * If `s` is an integer, then the spatial dimension `i` will be an index.
                * If `s` is None, then the spatial dimension `i` will not be sliced
                  or visualized.

        `slice_idx` should always be a 3-tuple.

    title : Optional[str], optional
        The `suptitle` of the image grid, by default None
    vol_labels : Optional[List[str]], optional
        Labels for each volume in the batch of volumes, by default None
    slice_labels : Optional[List[str]], optional
        Labels for each spatial slice according to `slice_idx`, by default None
    channel_labels : Optional[List[str]], optional
        Labels for each channel in `vols`, by default None
    colorbars : Optional[str], optional
        Set the type of colorbar and intensity normalization to use, by default None

        Valid options are:
            None - no colorbar or intensity normalization.
            "global" - one colorbar is created for the entire grid, and all images are
                normalized to have color intensity ranges match.
            "each" - every image has its own colorbar with no intensity normalization.
            "col", "cols", "column", "columns" - Every column is normalized and
                given a colorbar.
            "row", "rows" - Every row is normalized and given a colorbar.
    fig : Figure, optional
        Figure to plot into, by default None
    imshow_kwargs :
        Kwargs to pass to the `.imshow()` function call of each image.

    Returns
    -------
    Figure

    Raises
    ------
    ValueError
        Invalid option value for `colorbars`
    """

    # Canonical format of vols.
    # Enforce a B x C x D x H x W shape.
    bcdwh_vols = list()
    for vol in vols:
        if len(vol.shape) == 3:
            vol = vol.reshape(1, *vol.shape)
        if len(vol.shape) == 4:
            vol = vol.reshape(1, *vol.shape)
        bcdwh_vols.append(vol)
    # Flatten into a list of C x ... arrays.
    bcdwh_vols = list(itertools.chain.from_iterable(bcdwh_vols))

    row_slice_by_vol_labels = list()
    flat_slices = list()
    for i_b, chan_v in enumerate(bcdwh_vols):
        for k_s, s in enumerate(slice_idx):
            # Use None as a sentinal to only create a new row label for every
            # (vol x slice) pairing, disregarding the channel index.
            row_label = None
            # Need channel index to be the inner-most loop for plotting.
            for v in chan_v:
                # If slice idx was None, skip this slice.
                if s is None:
                    continue
                # If slice idx was a fraction, interpret that as a percent of the total
                # dim size.
                if isinstance(s, float) and s >= 0.0 and s <= 1.0:
                    idx = math.floor(s * v.shape[k_s])
                else:
                    idx = math.floor(s)

                # Generate the slice(None) objects that follow the integer index.
                slice_after = tuple(
                    itertools.repeat(slice(None), len(slice_idx) - (k_s + 1))
                )
                slicer = (
                    ...,
                    idx,
                ) + slice_after
                vol_slice = v[slicer]

                if torch.is_tensor(vol_slice):
                    vol_slice = vol_slice.detach().cpu().numpy()
                flat_slices.append(vol_slice)

                # Handle labelling of the rows, only one label per row.
                if row_label is None:
                    row_label = ""
                    if vol_labels is not None:
                        row_label = row_label + vol_labels[i_b] + " "
                    if slice_labels is not None:
                        row_label = row_label + slice_labels[k_s]

                    row_slice_by_vol_labels.append(row_label.strip())

    maybe_empty_row_vol_labels = (
        None
        if all(map(lambda s: s == "", row_slice_by_vol_labels))
        else row_slice_by_vol_labels
    )

    return plot_im_grid(
        *flat_slices,
        nrows=len(row_slice_by_vol_labels),
        title=title,
        row_headers=maybe_empty_row_vol_labels,
        col_headers=channel_labels,
        colorbars=colorbars,
        fig=fig,
        **imshow_kwargs,
    )


def plot_fodf_coeff_slices(
    *fodf_vols: Sequence[Union[torch.Tensor, np.ndarray]],
    fodf_coeff_idx=(0, 3, 10, 21, 36),
    slice_idx=(0.5, 0.5, 0.5),
    reorient_ras_for_viz=True,
    **plot_im_grid_kwargs,
):
    vols = list()
    for v in fodf_vols:
        v = torch.as_tensor(v).detach().cpu()
        if v.ndim == 5 and int(v.shape[0]) == 1:
            v = v[0]
        v = v.numpy()
        if reorient_ras_for_viz:
            # Assume all volumes are in RAS orientation, then transform them for better
            # visualization.
            # We don't care about spacing, rotation, etc., just flip/90 deg rotate
            # to get a better view.
            v_im = nib.Nifti1Image(np.moveaxis(v, 0, -1), np.eye(4))
            v = np.moveaxis(
                mrinr.data.io.reorient_nib_im(v_im, VIZ_NIFTI_ORIENTATION).get_fdata(),
                -1,
                0,
            )
        vols.append(v)

    n_fod_coeffs_to_plot = len(fodf_coeff_idx)
    nrows = n_fod_coeffs_to_plot

    slices_to_plot = list()
    for i_c, c_idx in enumerate(fodf_coeff_idx):
        for coeff_vs in vols:
            coeff_v = coeff_vs[c_idx]
            for j_dim, s in enumerate(slice_idx):
                if s is None:
                    continue
                # Handle possible flips/rotations for viz orientation.
                # RAS -> IPR
                if reorient_ras_for_viz:
                    # if slice_axis == "ax":
                    if j_dim == 2:
                        slice_dim_idx = 0
                        array_slice = _float_or_int_slice_idx(
                            slice_idx[slice_dim_idx],
                            coeff_v.shape[slice_dim_idx],
                            reverse=True,
                        )
                        slicer = (array_slice, slice(None), slice(None))
                    # elif slice_axis == "sag":
                    elif j_dim == 0:
                        slice_dim_idx = 2
                        array_slice = _float_or_int_slice_idx(
                            slice_idx[slice_dim_idx],
                            coeff_v.shape[slice_dim_idx],
                            reverse=True,
                        )
                        slicer = (slice(None), slice(None), array_slice)
                    # elif slice_axis == "cor":
                    elif j_dim == 1:
                        array_slice = _float_or_int_slice_idx(
                            slice_idx[slice_dim_idx],
                            coeff_v.shape[slice_dim_idx],
                            reverse=True,
                        )
                        slicer = (slice(None), array_slice, slice(None))
                    else:
                        raise ValueError("Invalid slice axis.")
                else:
                    # If slice idx was a fraction, interpret that as a percent of the total
                    # dim size.
                    if isinstance(s, float) and s >= 0.0 and s <= 1.0:
                        idx = math.floor(s * coeff_v.shape[j_dim])
                    else:
                        idx = math.floor(s)
                    slice_template = [slice(None), slice(None), slice(None)]
                    slice_template[j_dim] = idx
                    slicer = tuple(slice_template)

                coeff_v_slice = coeff_v[slicer]
                slices_to_plot.append(coeff_v_slice)

    return plot_im_grid(*slices_to_plot, nrows=nrows, **plot_im_grid_kwargs)


def _plot_fodf_coeff_slices(
    *fodf_vols,
    fig,
    rect=111,
    vol_slice_idx_as_proportions=(0.5, 0.5, 0.5),
    fodf_coeff_idx=(0, 3, 10, 21, 36),
    fodf_vol_labels=None,
    imshow_kwargs: dict = dict(),
    image_grid_kwargs: dict = dict(),
):
    vols = list()
    for v in fodf_vols:
        v = v.detach().cpu()
        if v.ndim == 5 and int(v.shape[0]) == 1:
            v = v[0]
        v = v.numpy()
        vols.append(v)
    n_vols = len(vols)
    if fodf_vol_labels is None:
        vol_labels = list(itertools.repeat(None, n_vols))
    else:
        vol_labels = list(fodf_vol_labels)
    # n_fod_coeffs = int(vols[0].shape[0])
    n_slices = len(list(filter(lambda x: x is not None, vol_slice_idx_as_proportions)))
    n_fod_coeffs_to_plot = len(fodf_coeff_idx)
    image_grid_kwargs = {
        "fig": fig,
        "rect": rect,
        "nrows_ncols": (n_fod_coeffs_to_plot, n_vols * n_slices),
        "label_mode": "1",
        "cbar_mode": "edge",
        "cbar_location": "right",
        "cbar_size": "9%",
    } | image_grid_kwargs

    imshow_kwargs = {"cmap": "gray", "interpolation": "antialiased"} | imshow_kwargs

    grid = ImageGrid(**image_grid_kwargs)

    row_order_grid_idx = 0
    for i_fodf_coeff_idx, fodf_coeff_idx in enumerate(fodf_coeff_idx):
        for j_vol, vol in enumerate(vols):
            vol_label = vol_labels[j_vol]
            for k_slice, slice_idx_prop in enumerate(vol_slice_idx_as_proportions):
                if slice_idx_prop is None:
                    continue
                shape = tuple(vol.shape[1:])
                vol_slice_idx = round(shape[k_slice] * slice_idx_prop)
                slicer = [slice(None), slice(None), slice(None)]
                slicer[k_slice] = vol_slice_idx
                slicer = [fodf_coeff_idx] + slicer
                slicer = tuple(slicer)
                im = vol[slicer]

                ax = grid[row_order_grid_idx]
                ax.imshow(im, **imshow_kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                if vol_label is not None and i_fodf_coeff_idx == 0:
                    ax.set_title(vol_label)

                row_order_grid_idx += 1

    return fig, grid
