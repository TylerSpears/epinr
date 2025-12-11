# -*- coding: utf-8 -*-
# Functions related to sampling images/volumes for training and evaluation.
import collections
import itertools
import math
import typing
from typing import Callable, Literal, Optional

import einops
import monai
import monai.data
import monai.transforms
import numpy as np
import scipy
import scipy.ndimage
import skimage
import skimage.morphology
import torch

import mrinr

__all__ = [
    "BaseAlignedPatchDataset",
    "PairedSpatialSample",
    "AutoencoderSpatialSample",
    "deterministic_sample_batched_paired_spatial_data",
    "deterministic_sample_batched_autoencoder_spatial_data",
    "gaussian_kernel_weight_at_coord_orig",
    "gaussian_kernel_weight_at_center",
    "DataCoordAffSet",
    "LatentCoordAffSet",
    "cumul_sampling_weight_mask",
    "cumul_sampling_mask_from_weight_mask",
    "erode_patch_sample_mask_by_chebyshev_dist",
    "erode_patch_sample_mask",
    "sample_batched_el_bbs",
    "sample_patch_coords_from_weight_map",
    "_extend_el_bbs_to_max_shape_in_batch",
    "_el_bb_from_weight_map",
]


class BaseAlignedPatchDataset(
    monai.transforms.Randomizable, monai.data.IterableDataset
):
    def __init__(
        self,
        data: list[dict],
        samples_per_data_entry: int,
        batch_size: int,
        batches_per_iter_exhaust: int,
        patch_sample_fn: Optional[Callable] = None,
        seed: int = 0,
        starting_epoch: int = 1,
        skip_every_n_batches: Optional[int] = None,
        stop_batch_skip_after_epochs: Optional[int] = None,
    ):
        super().__init__(data=data)
        self._data = data
        self._patch_sample_fn = patch_sample_fn
        self._samples_per_data_entry = int(samples_per_data_entry)
        self._batch_size = int(batch_size)
        # Total batch size must be divisible by the number of samples per subject.
        assert self._batch_size % self._samples_per_data_entry == 0
        self._unique_data_entries_per_batch = (
            self._batch_size // self._samples_per_data_entry
        )
        self.seed = seed

        self._n_batches_per_iter_exhaust = int(math.floor(batches_per_iter_exhaust))

        self._curr_epoch = starting_epoch
        self._skip_every_n_batches = skip_every_n_batches
        self._stop_skip_after_epochs = stop_batch_skip_after_epochs
        self._all_data_idx = np.arange(len(self._data))
        # 'self.randomize()' initializes/populates 'self._data_idx_queue' and
        # instantiates a pytorch Generator for passing to sampling functions, but is
        # entirely contained within this Dataset object.
        self._data_idx_queue = None
        self._torch_generator = None
        super().set_random_state(seed=self.seed)

    @property
    def n_batches_per_iter_exhaust(self):
        return self._n_batches_per_iter_exhaust

    @staticmethod
    def _batched(iterable, n):
        "Batch data into tuples of length n. The last batch may be shorter."
        # batched('ABCDEFG', 3) --> ABC DEF G
        # From python docs
        # <https://docs.python.org/3.11/library/itertools.html#itertools-recipes>.
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

    def _deterministic_fill_data_idx_queue(self):
        q_len = self._n_batches_per_iter_exhaust * self._unique_data_entries_per_batch

        if len(self._all_data_idx) > q_len:
            q = self._all_data_idx[:q_len]
        elif len(self._all_data_idx) < q_len:
            # If the number of data entries is less than the number of batches, then
            # just repeat the data entries until we have enough.
            q = np.tile(
                self._all_data_idx, int(math.ceil(q_len / len(self._all_data_idx)))
            )
            q = q[:q_len]
        else:
            q = self._all_data_idx

        self._data_idx_queue = einops.rearrange(
            q, "(n s) -> n s", s=self._unique_data_entries_per_batch
        )
        # Data index table should have shape (n_batches, m_elements_from_data).
        assert self._data_idx_queue.shape[0] == self._n_batches_per_iter_exhaust
        assert self._data_idx_queue.shape[1] == self._unique_data_entries_per_batch

    def randomize(self) -> None:
        shuffle_q = self.R.choice(
            self._all_data_idx,
            size=(
                self._n_batches_per_iter_exhaust * self._unique_data_entries_per_batch
            ),
            replace=True,
        )
        self._data_idx_queue = einops.rearrange(
            shuffle_q, "(n s) -> n s", s=self._unique_data_entries_per_batch
        )
        # Data index table should have shape (n_batches, m_elements_from_data).
        assert self._data_idx_queue.shape[0] == self._n_batches_per_iter_exhaust
        assert self._data_idx_queue.shape[1] == self._unique_data_entries_per_batch

        # If batches are to be skipped, then set the skipped batches to -1.
        if self._skip_every_n_batches is not None:
            if (
                self._stop_skip_after_epochs is not None
                and self._curr_epoch <= self._stop_skip_after_epochs
            ):
                # Split the data indices into chunks of size N.
                batched_row_idx = list(
                    self._batched(
                        np.arange(len(self._data_idx_queue)).tolist(),
                        n=self._skip_every_n_batches,
                    )
                )
                # Now set every other chunk to -1, starting with the second chunk.
                for b in batched_row_idx[1::2]:
                    self._data_idx_queue[b, :] = -1

        # assert len(self._data_idx_queue) == self._n_batches_per_iter_exhaust

        # Instantiate the pytorch rng state from the numpy rng.
        if self._torch_generator is None:
            self._torch_generator = mrinr.utils.fork_rng(torch.default_generator)
        # Randomize the torch generator state.
        # Bounds of the pytorch generator seed are
        # [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff], inclusive. See
        # <https://pytorch.org/docs/stable/random.html#torch.random.manual_seed>.
        # However, that requires (at least) 128-bit int support, while numpy usually
        # only supports up to 64-bit ints.
        new_torch_seed = self.R.randint(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
        )
        self._torch_generator = self._torch_generator.manual_seed(new_torch_seed)

    @staticmethod
    def _step_torch_generator(
        generator: torch.Generator, steps: int = 1
    ) -> torch.Generator:
        rng_2 = mrinr.utils.fork_rng(generator)
        for _ in range(steps):
            torch.randint(0, 2**32, (10,), generator=rng_2)
        return rng_2

    def __iter__(self):
        raise NotImplementedError(
            "Subclasses must implement the '__iter__' method for custom sampling logic."
        )

    # Example __iter__() method:
    # def __iter__(self):
    #     self.seed += 1
    #     self._curr_epoch += 1
    #     super().set_random_state(seed=self.seed)  # make all workers in sync
    #     self.randomize()

    #     # Enable multiprocessing for yielding samples.
    #     info = torch.utils.data.get_worker_info()
    #     num_workers = info.num_workers if info is not None else 1
    #     id_ = info.id if info is not None else 0
    #     # print(f"========= Worker {id_} Epoch {self._curr_epoch}")

    #     for i, data_idx_row in enumerate(self._data_idx_queue):
    #         if i % num_workers == id_:
    #             if (np.asarray(data_idx_row) < 0).all():
    #                 sample = [-torch.ones(1)] * (
    #                     len(data_idx_row) * self._samples_per_data_entry
    #                 )
    #             else:
    #                 # Pass n_subjs_per_batch to the sampling function.
    #                 sample = self._patch_sample_fn(
    #                     [self._data[j] for j in data_idx_row]
    #                 )
    #                 # 'sample' should be of length 'n_subjs_per_batch * n_samples_per_subj'
    #             for s in sample:
    #                 yield s


class PairedSpatialSample(typing.TypedDict):
    spatial_data_in: mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
    spatial_data_out: mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
    mask_in: Optional[mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage]
    mask_out: Optional[mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage]
    affine_el2coord_in: (
        mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
    )
    affine_el2coord_out: (
        mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
    )
    coord_grid_in: mrinr.typing.SingleCoordGrid3D | mrinr.typing.SingleCoordGrid2D
    coord_grid_out: mrinr.typing.SingleCoordGrid3D | mrinr.typing.SingleCoordGrid2D
    spacing_in: torch.Tensor
    spacing_out: torch.Tensor
    props: Optional[dict[str, typing.Any]]


class AutoencoderSpatialSample(typing.TypedDict):
    spatial_data_in: mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
    spatial_data_out: mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
    mask_in: Optional[mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage]
    mask_out: Optional[mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage]
    affine_el2coord_in: (
        mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
    )
    affine_el2coord_latent: (
        mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
    )
    affine_el2coord_out: (
        mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
    )
    coord_grid_in: mrinr.typing.SingleCoordGrid3D | mrinr.typing.SingleCoordGrid2D
    coord_grid_latent: mrinr.typing.SingleCoordGrid3D | mrinr.typing.SingleCoordGrid2D
    coord_grid_out: mrinr.typing.SingleCoordGrid3D | mrinr.typing.SingleCoordGrid2D
    spacing_in: torch.Tensor
    spacing_latent: torch.Tensor
    spacing_out: torch.Tensor
    props: Optional[dict[str, typing.Any]]


class _SampleBBs(typing.NamedTuple):
    bb: mrinr.typing.BoundingBox2D | mrinr.typing.BoundingBox3D
    shape: tuple[int, ...]


def _extend_el_bbs_to_max_shape_in_batch(
    el_bb: mrinr.typing.BoundingBox2D | mrinr.typing.BoundingBox3D,
    pad_size_factor: int,
) -> _SampleBBs:
    el_sizes = el_bb[:, 1] - el_bb[:, 0] + 1
    # Find total padding amount to have all bounding boxes have the same number of
    # (pix|vox)els.
    max_el_size = el_sizes.max(0, keepdim=True).values
    el_to_pad = max_el_size - el_sizes
    # Ensure sizes are factors of some given integer.
    el_to_pad += (
        pad_size_factor - (max_el_size.round().int() % pad_size_factor)
    ) % pad_size_factor
    pad_low = (el_to_pad / 2).ceil()
    pad_high = el_to_pad - pad_low
    padded_el_bb = torch.stack(
        [
            el_bb[:, 0] - pad_low,
            el_bb[:, 1] + pad_high,
        ],
        1,
    )

    # All patch bbs should be the same size now, so just select the first one.
    patch_size = (padded_el_bb[:, 1] - padded_el_bb[:, 0] + 1)[0]

    return _SampleBBs(
        padded_el_bb, tuple(patch_size.flatten().round().int().detach().cpu().tolist())
    )


@torch.no_grad()
def cumul_sampling_weight_mask(
    mask: mrinr.typing.SingleMaskImage | mrinr.typing.SingleMaskVolume,
    mask_erode_by: Optional[int | Literal["patch"]] = None,
    mask_dilate_by: Optional[int] = None,
    mask_erode_patch_kwargs: Optional[dict] = None,
) -> mrinr.typing.SingleScalarImage | mrinr.typing.SingleScalarVolume:
    # Create the volume sampling weights from locations in mask.
    weight_mask = mask.squeeze(0).clone().cpu().bool().numpy()
    if mask_erode_by is not None and mask_dilate_by is not None:
        raise ValueError(
            "Cannot specify both 'mask_erode_by' and 'mask_dilate_by' at the same time."
        )

    if mask_erode_by is not None:
        if isinstance(mask_erode_by, int):
            weight_mask = scipy.ndimage.binary_erosion(
                weight_mask, iterations=mask_erode_by
            )
        elif (
            isinstance(mask_erode_by, str)
            and "patch" in mask_erode_by.lower()
            and mask_erode_patch_kwargs is not None
        ):
            weight_mask = erode_patch_sample_mask(
                torch.from_numpy(weight_mask),
                **({"ensure_nonzero": True} | mask_erode_patch_kwargs),
            ).numpy()
        else:
            raise ValueError(
                "Invalid value for 'mask_erode_by'. Must be an integer or 'patch'."
            )
    elif mask_dilate_by is not None:
        weight_mask = scipy.ndimage.binary_dilation(
            weight_mask, iterations=mask_dilate_by
        )

    weight_mask = weight_mask.astype(float)
    norm_sampling_w = weight_mask.astype(float) / weight_mask.astype(float).sum()
    assert not np.isnan(norm_sampling_w).any()
    cumul_sampling_w = np.cumsum(norm_sampling_w.reshape(-1), axis=0).reshape(
        norm_sampling_w.shape
    )
    cumul_sampling_w = torch.from_numpy(cumul_sampling_w).to(
        dtype=torch.float32, device=mask.device
    )
    cumul_sampling_w.reshape_as(mask).clamp(0.0, 1.0).to(mask.device)
    return cumul_sampling_w


@torch.no_grad()
def cumul_sampling_mask_from_weight_mask(
    weight_mask: mrinr.typing.SingleScalarImage | mrinr.typing.SingleScalarVolume,
) -> mrinr.typing.SingleScalarImage | mrinr.typing.SingleScalarVolume:
    # Create the volume sampling weights from locations in mask.
    w = weight_mask.squeeze(0).clone().cpu()
    w = w.to(torch.promote_types(w.dtype, torch.float32)).numpy()
    norm_sampling_w = w.astype(float) / w.astype(float).sum()
    assert not np.isnan(norm_sampling_w).any()
    cumul_sampling_w = np.cumsum(norm_sampling_w.reshape(-1), axis=0).reshape(
        norm_sampling_w.shape
    )
    cumul_sampling_w = torch.from_numpy(cumul_sampling_w).to(
        dtype=torch.float32, device=weight_mask.device
    )
    cumul_sampling_w.reshape_as(weight_mask).clamp(0.0, 1.0).to(weight_mask.device)
    return cumul_sampling_w


def sample_patch_coords_from_weight_map(
    sample_r_vs: torch.Tensor,
    cumulative_weight_map: mrinr.typing.SingleScalarImage
    | mrinr.typing.SingleScalarVolume,
    affine_weight_map_el2coord: mrinr.typing.SingleHomogeneousAffine2D
    | mrinr.typing.SingleHomogeneousAffine3D,
    patch_size: tuple[int, ...],
    affine_x_el2coord: mrinr.typing.SingleHomogeneousAffine2D
    | mrinr.typing.SingleHomogeneousAffine3D,
) -> mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D:
    # Sample the (pix|vox)el bounding boxes from the weight map.
    sample_el_bbs = _el_bb_from_weight_map(
        sample_r_vs=sample_r_vs,
        cumulative_weight_map=cumulative_weight_map,
        affine_weight_map_el2coord=affine_weight_map_el2coord,
        patch_size=patch_size,
        affine_x_el2coord=affine_x_el2coord,
        clamp_sample_r_vs=True,
    )
    # Transform the el bbs to full coordinate grids in the affine space.
    sample_coord_grids = mrinr.coords.bb_to_affine_grid(
        sample_el_bbs, spatial_shape=patch_size, affine=affine_x_el2coord
    )
    return sample_coord_grids


def _el_bb_from_weight_map(
    sample_r_vs: torch.Tensor,
    cumulative_weight_map: mrinr.typing.SingleScalarImage
    | mrinr.typing.SingleScalarVolume,
    affine_weight_map_el2coord: mrinr.typing.SingleHomogeneousAffine2D
    | mrinr.typing.SingleHomogeneousAffine3D,
    patch_size: tuple[int, ...],
    affine_x_el2coord: mrinr.typing.SingleHomogeneousAffine2D
    | mrinr.typing.SingleHomogeneousAffine3D,
    allow_fractional_el: bool = False,
    clamp_sample_r_vs: bool = True,
) -> mrinr.typing.BoundingBox2D | mrinr.typing.BoundingBox3D:
    """Patch bounding box coordinates based on the weight map and given random values.

    Parameters
    ----------
    sample_r_vs : torch.Tensor
        Values in range (0, 1] that indicate the patch center in the weight mask.
    cumulative_weight_map : SingleScalarImage | SingleScalarVolume
        Weight mask used for sampling patches with the given sample_r_vs.

        This should be a single-channel image/volume where values are sampling weights,
        cumulatively summed along the spatial dimensions; i.e. the max value should be
        1.0.
    affine_weight_map_el2coord : SingleHomogeneousAffine2D | SingleHomogeneousAffine3D
        Affine matrix that defines coordinates in the weight map.
    patch_size : tuple[int, ...]
        Target patch size in x element coordinates.
    affine_x_el2coord : SingleHomogeneousAffine2D | SingleHomogeneousAffine3D
        Affine matrix that defines the el->coord for the target space.
    clamp_sample_r_vs : bool, optional
        Clamp random values to $[0.0 + \epsilon, 1.0 - \epsilon]$, by default True

    Returns
    -------
    mrinr.typing.BoundingBox2D | mrinr.typing.BoundingBox3D
        Bounding box coordinates of the target patch.
    """
    if clamp_sample_r_vs:
        # Ensure that random values do not fall on the edge of the cumulative weight map.
        r_vs = torch.clamp(
            sample_r_vs,
            2 * torch.finfo(sample_r_vs.dtype).resolution,
            1 - 2 * torch.finfo(sample_r_vs.dtype).resolution,
        )
    else:
        r_vs = sample_r_vs

    # Remove channel dimension.
    w_cumul = cumulative_weight_map.squeeze(0)
    # Get patch center (pix|vox)el and template coordinates based on the sample random
    # values.
    patch_center_flat_indices = torch.searchsorted(w_cumul.view(-1), r_vs, side="right")
    patch_center_indices = np.unravel_index(
        patch_center_flat_indices.cpu().numpy(), shape=tuple(w_cumul.shape)
    )
    patch_center_el_coords = torch.from_numpy(np.asarray(patch_center_indices)).T.to(
        cumulative_weight_map.device
    )

    # Transform from weight map el coords to x el coords.
    aff_dtype = torch.promote_types(
        affine_weight_map_el2coord.dtype, affine_x_el2coord.dtype
    )
    patch_center_x_el_coords = mrinr.coords.transform_coords(
        patch_center_el_coords,
        mrinr.coords.combine_affines(
            affine_weight_map_el2coord.to(aff_dtype),
            mrinr.coords.inv_affine(affine_x_el2coord.to(aff_dtype)),
            transform_order_left_to_right=True,
        ),
        broadcast_batch=False,
    )

    size = torch.as_tensor(patch_size)
    size_lower = (size / 2).ceil()
    size_upper = size - size_lower
    patch_x_bb_el = torch.stack(
        [
            patch_center_x_el_coords - size_lower.unsqueeze(0),
            patch_center_x_el_coords + size_upper.unsqueeze(0) - 1,
        ],
        1,
    ).to(torch.float32)

    # # patch x el bb -> space template bb
    # patch_coord_bb = mrinr.coords.transform_coords(patch_x_bb_el, affine_x_el2coord)
    return patch_x_bb_el


def sample_batched_el_bbs(
    xs: list[mrinr.typing.Image | mrinr.typing.Volume]
    | mrinr.typing.Image
    | mrinr.typing.Volume,
    affine_x_el2coord: mrinr.typing.HomogeneousAffine2D
    | mrinr.typing.HomogeneousAffine3D,
    patch_size: tuple[int, ...] | Literal["whole"],
    sample_r_vs: torch.Tensor | None,
    cumulative_weight_map: list[mrinr.typing.ScalarImage | mrinr.typing.ScalarVolume]
    | None,
    affine_weight_map_el2coord: mrinr.typing.HomogeneousAffine2D
    | mrinr.typing.HomogeneousAffine3D
    | None,
    patch_size_factor: int = 8,
    clamp_sample_r_vs: bool | None = True,
) -> _SampleBBs:
    spatial_dims = affine_x_el2coord.shape[-1] - 1
    if torch.is_tensor(xs):
        squeeze_batch = True
        xs = [xs]
        if affine_x_el2coord.ndim == 2:
            affine_x_el2coord = affine_x_el2coord.unsqueeze(0)
        if sample_r_vs is not None and sample_r_vs.ndim == 1:
            sample_r_vs = sample_r_vs.unsqueeze(0)
        if torch.is_tensor(cumulative_weight_map):
            cumulative_weight_map = [cumulative_weight_map]
        if (
            affine_weight_map_el2coord is not None
            and affine_weight_map_el2coord.ndim == 2
        ):
            affine_weight_map_el2coord = affine_weight_map_el2coord.unsqueeze(0)
    else:
        squeeze_batch = False

    el_bbs = list()
    for i, x in enumerate(xs):
        if patch_size == "whole":
            s = tuple(x.shape[-spatial_dims:])
            # Just get the bounding box in element space, so an identity affine matrix.
            el_bb = mrinr.coords.el_bb_from_shape(
                affine=torch.eye(spatial_dims + 1), spatial_shape=s
            )
            # Unsqueeze to have a batch dim of 1, later concat.
            el_bb = el_bb.unsqueeze(0)
            el_bbs.append(el_bb)
        else:
            el_bb = _el_bb_from_weight_map(
                sample_r_vs=sample_r_vs[i].view(-1),
                cumulative_weight_map=cumulative_weight_map[i],
                affine_weight_map_el2coord=affine_weight_map_el2coord[i],
                patch_size=patch_size,
                affine_x_el2coord=affine_x_el2coord[i],
                clamp_sample_r_vs=clamp_sample_r_vs,
            )
            el_bbs.append(el_bb)
    # Concat the bounding boxes to form the batch.
    el_bbs = torch.concat(el_bbs, dim=0).to(affine_x_el2coord)

    sample_bbs = _extend_el_bbs_to_max_shape_in_batch(
        el_bbs, pad_size_factor=patch_size_factor
    )
    if squeeze_batch:
        sample_bbs = _SampleBBs(bb=sample_bbs.bb.squeeze(0), shape=sample_bbs.shape)
    return sample_bbs


_DEFAULT_PATCH_GRID_RESAMPLE_KWARGS = dict(
    mode_or_interpolation="nearest",
    padding_mode_or_bound="border",
    interp_lib="torch",
)
_DEFAULT_PAD_SAMPLE_MODE = "replicate"
DataCoordAffSet = collections.namedtuple(
    "_DataCoordAffSet", ["spatial_data", "mask", "affine_el2coord"]
)
LatentCoordAffSet = collections.namedtuple(
    "_LatentCoordAffSet", ["coord_grid", "affine_el2coord"]
)


def deterministic_sample_batched_paired_spatial_data(
    input_to_sample: list[DataCoordAffSet],
    output_to_sample: list[DataCoordAffSet],
    out_patch_size: tuple[int, ...] | Literal["whole"],
    sample_r_vs: torch.Tensor | None,
    cumulative_weight_map: list[mrinr.typing.ScalarImage | mrinr.typing.ScalarVolume]
    | None,
    affine_weight_map_el2coord: mrinr.typing.HomogeneousAffine2D
    | mrinr.typing.HomogeneousAffine3D
    | None,
    patch_size_factor: int = 8,
    exscribe_expansion_buffer: float = 1.1,
    clamp_sample_r_vs: bool | None = True,
    batched_props: Optional[list[dict[str, typing.Any]]] = None,
    output_patch_size_factor: Optional[int] = None,
    input_patch_size_factor: Optional[int] = None,
    patch_grid_resample_kwargs: dict[
        str, typing.Any
    ] = _DEFAULT_PATCH_GRID_RESAMPLE_KWARGS,
) -> list[PairedSpatialSample]:
    # Assert all batch sizes are the same.
    assert len(input_to_sample) == len(output_to_sample)
    if sample_r_vs is not None:
        assert len(sample_r_vs) == len(output_to_sample)
    if cumulative_weight_map is not None:
        assert len(cumulative_weight_map) == len(output_to_sample)
    if affine_weight_map_el2coord is not None:
        assert affine_weight_map_el2coord.shape[0] == len(output_to_sample)
    if batched_props is not None:
        assert len(batched_props) == len(output_to_sample)

    if output_patch_size_factor is None:
        output_patch_size_factor = patch_size_factor
    if input_patch_size_factor is None:
        input_patch_size_factor = patch_size_factor

    spatial_dims = input_to_sample[0].affine_el2coord.shape[-1] - 1

    # Get pixel/voxel bounding boxes for the ouptut samples.
    output_affs = torch.stack([d.affine_el2coord for d in output_to_sample], dim=0)
    output_el_bbs, output_spatial_shape = sample_batched_el_bbs(
        xs=[d.spatial_data for d in output_to_sample],
        affine_x_el2coord=output_affs,
        patch_size=out_patch_size,
        sample_r_vs=sample_r_vs,
        cumulative_weight_map=cumulative_weight_map,
        affine_weight_map_el2coord=affine_weight_map_el2coord,
        patch_size_factor=output_patch_size_factor,
        clamp_sample_r_vs=clamp_sample_r_vs,
    )
    # Corners and BBs have 3 dimensions for coordinates, but that number can be
    # ambiguous for `transform_coords()`. So, we need to expand and squeeze the
    # coordinates to ensure everything is clear and consistent.
    if spatial_dims == 2:
        expand_pattern = "b n_points dim -> b 1 n_points dim"
        squeeze_pattern = "b 1 n_points dim -> b n_points dim"
    elif spatial_dims == 3:
        expand_pattern = "b n_points dim -> b 1 1 n_points dim"
        squeeze_pattern = "b 1 1 n_points dim -> b n_points dim"
    else:
        raise ValueError("Invalid spatial dimensions.")

    output_el_corners = mrinr.coords.enumerate_bb_to_corners(output_el_bbs)
    # Expand the corner coordinates to clarify which dimension is the batch dimension.
    batched_output_el_corners = einops.rearrange(output_el_corners, expand_pattern)
    batched_output_coord_corners = mrinr.coords.transform_coords(
        batched_output_el_corners,
        affine_a2b=output_affs,
        broadcast_batch=False,
    )

    # Get corner coordinates of output samples, bring them into the input space.
    # output coord corners
    # == input coord corners
    # -> input el corners
    # -(exscribed)> input el bounding box
    input_affs = torch.stack([d.affine_el2coord for d in input_to_sample], dim=0)
    batched_input_el_corners = mrinr.coords.transform_coords(
        batched_output_coord_corners,
        mrinr.coords.inv_affine(input_affs),
        broadcast_batch=False,
    )
    input_el_corners = einops.rearrange(batched_input_el_corners, squeeze_pattern)
    input_el_bbs = mrinr.coords.exscribe_corners_to_bb(
        input_el_corners,
        round_bb=True,
        expansion_buffer=exscribe_expansion_buffer,
    )
    # Extend input bbs across the batch.
    input_el_bbs, input_spatial_shape = _extend_el_bbs_to_max_shape_in_batch(
        input_el_bbs, pad_size_factor=input_patch_size_factor
    )

    ## Transform (pix|vox)el bounding boxes to full grids of template coordinates. Then
    ## nearest-neighbor sample the images/volumes to create the input and output
    ## patches.
    input_coord_grids = mrinr.coords.bb_to_affine_grid(
        input_el_bbs, spatial_shape=input_spatial_shape, affine=input_affs
    ).to(torch.float32)
    output_coord_grids = mrinr.coords.bb_to_affine_grid(
        output_el_bbs, spatial_shape=output_spatial_shape, affine=output_affs
    ).to(torch.float32)

    # Construct list of paired samples.
    sample_dicts = list()
    for (
        i_sample,
        input_to_sample_i,
        in_el_bb_i,
        in_coord_grid_i,
        output_to_sample_i,
        out_el_bb_i,
        out_coord_grid_i,
    ) in zip(
        range(len(input_to_sample)),
        input_to_sample,
        input_el_bbs.unbind(0),
        input_coord_grids.unbind(0),
        output_to_sample,
        output_el_bbs.unbind(0),
        output_coord_grids.unbind(0),
        strict=True,
    ):
        d_i: PairedSpatialSample = dict()

        # Input sample
        d_i["coord_grid_in"] = in_coord_grid_i.to(torch.float32)
        d_i["spatial_data_in"] = mrinr.grid_resample(
            input_to_sample_i.spatial_data,
            affine_x_el2coords=input_to_sample_i.affine_el2coord,
            sample_coords=in_coord_grid_i,
            **patch_grid_resample_kwargs,
        ).to(torch.float32)
        if input_to_sample_i.mask is not None:
            d_i["mask_in"] = mrinr.grid_resample(
                # input_to_sample_i.spatial_data,
                input_to_sample_i.mask,
                affine_x_el2coords=input_to_sample_i.affine_el2coord,
                sample_coords=in_coord_grid_i,
                **patch_grid_resample_kwargs,
            ).bool()
        else:
            d_i["mask_in"] = None
        # Construct new affine for the patch now that cropping/padding has been applied
        # (i.e., the new space is translated).
        # TODO: Check if this translation amount is correct. Does the "bottom" of the
        # element BB correspond to a translation of the input patch in pix/vox?
        in_tr_el_aff = torch.eye(spatial_dims + 1).to(input_to_sample_i.affine_el2coord)
        in_tr_el_aff[:-1, -1] = torch.Tensor(in_el_bb_i[0]).to(in_tr_el_aff)
        in_aff = mrinr.coords.combine_affines(
            in_tr_el_aff,
            input_to_sample_i.affine_el2coord,
            transform_order_left_to_right=True,
        ).to(torch.float32)
        d_i["affine_el2coord_in"] = in_aff
        d_i["spacing_in"] = mrinr.coords.spacing(in_aff).flatten()

        # Output sample
        d_i["coord_grid_out"] = out_coord_grid_i.to(torch.float32)
        d_i["spatial_data_out"] = mrinr.grid_resample(
            output_to_sample_i.spatial_data,
            affine_x_el2coords=output_to_sample_i.affine_el2coord,
            sample_coords=out_coord_grid_i,
            **patch_grid_resample_kwargs,
        ).to(torch.float32)
        if output_to_sample_i.mask is not None:
            d_i["mask_out"] = mrinr.grid_resample(
                # output_to_sample_i.spatial_data,
                output_to_sample_i.mask,
                affine_x_el2coords=output_to_sample_i.affine_el2coord,
                sample_coords=out_coord_grid_i,
                **patch_grid_resample_kwargs,
            ).bool()
        else:
            d_i["mask_out"] = None
        # Construct new affine for the patch now that cropping/padding has been applied
        # (i.e., the new space is translated).
        out_tr_el_aff = torch.eye(spatial_dims + 1).to(
            output_to_sample_i.affine_el2coord
        )
        out_tr_el_aff[:-1, -1] = torch.Tensor(out_el_bb_i[0]).to(out_tr_el_aff)
        out_aff = mrinr.coords.combine_affines(
            out_tr_el_aff,
            output_to_sample_i.affine_el2coord,
            transform_order_left_to_right=True,
        ).to(torch.float32)
        d_i["affine_el2coord_out"] = out_aff
        d_i["spacing_out"] = mrinr.coords.spacing(out_aff).flatten()

        # Add misc. properties if provided, like subj id, dataset name, etc.
        if batched_props is not None:
            d_i["props"] = batched_props[i_sample]
        else:
            d_i["props"] = None

        sample_dicts.append(d_i)

    # #!-----------------------------------------------------------
    # # Debug code to visualize patches
    # from pathlib import Path

    # Path("tmp").mkdir(exist_ok=True)
    # if spatial_dims == 2:
    #     for i, d in enumerate(sample_dicts):
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             input_to_sample[i].spatial_data,
    #             input_to_sample[i].affine_el2coord,
    #             f"tmp/example_{i}_input.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             output_to_sample[i].spatial_data,
    #             output_to_sample[i].affine_el2coord,
    #             f"tmp/example_{i}_output.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             d["spatial_data_in"],
    #             d["affine_el2coord_in"],
    #             f"tmp/patch_example_{i}_input.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             d["spatial_data_out"],
    #             d["affine_el2coord_out"],
    #             f"tmp/patch_example_{i}_output.nii.gz",
    #         )
    # #!-----------------------------------------------------------

    return sample_dicts


def __crop_autoencoder_sample_to_max_input_shape(
    homogeneous_spatial_samples: list[AutoencoderSpatialSample],
    patch_size_factor: int,
    max_in_spatial_size: Optional[int | tuple[int, ...]] = None,
) -> list[AutoencoderSpatialSample]:
    # !All values in 'homogeneous_spatial_samples' are assumed to have a consistent
    # !spatial shape between matching keys!
    if max_in_spatial_size is None:
        r = homogeneous_spatial_samples
    else:
        d = homogeneous_spatial_samples
        spatial_dims = d[0]["affine_el2coord_in"].shape[-1] - 1
        if isinstance(max_in_spatial_size, int):
            max_in_spatial_size = (max_in_spatial_size,) * spatial_dims

        in_spatial_size = np.asarray(
            tuple(d[0]["spatial_data_in"].shape[-spatial_dims:])
        )
        max_in_spatial_size = np.asarray(max_in_spatial_size)

        if (in_spatial_size > max_in_spatial_size).any():
            # Crop the input spatial size to meet the maximum allowed size.
            total_crop_size = np.clip(
                in_spatial_size - max_in_spatial_size, a_min=0, a_max=None
            )
            # Account for the size factor.
            total_crop_size += (in_spatial_size - total_crop_size) % patch_size_factor
            crop_low = np.floor(total_crop_size / 2).astype(int)
            crop_high = np.ceil(total_crop_size / 2).astype(int)
            # in_crop_ratio = total_crop_size / in_spatial_size
            in_pads_low_high = list(zip(-1 * crop_low, -1 * crop_high))

            # Crop the latent samples by a proportional amount. This is not precise, and
            # spatial padding may be necessary later in the pipeline. This is meant as
            # a quick, hopefully rare, fix to avoid memory issues.
            latent_spatial_size = np.asarray(
                tuple(d[0]["coord_grid_latent"].shape[-(spatial_dims + 1) : -1])
            )
            # Scale the crop size in the input space by the ratio of input-to-latent
            # space as a whole.
            in_to_latent_size_ratio = latent_spatial_size / in_spatial_size
            latent_crop_size = np.ceil(
                in_to_latent_size_ratio * total_crop_size
            ).astype(int)
            latent_crop_size += (
                latent_spatial_size - latent_crop_size
            ) % patch_size_factor
            # Clip the crop size if it meets or exceeds the latent spatial size.
            latent_crop_size = np.where(
                latent_crop_size >= latent_spatial_size, 0, latent_crop_size
            )
            latent_crop_low = np.floor(latent_crop_size / 2).astype(int)
            latent_crop_high = np.ceil(latent_crop_size / 2).astype(int)
            latent_pads_low_high = list(
                zip(-1 * latent_crop_low, -1 * latent_crop_high)
            )

            # Crop the output samples by an amount proportional to the latent space
            # cropping. Again, not precise.
            out_spatial_size = np.asarray(
                tuple(d[0]["spatial_data_out"].shape[-spatial_dims:])
            )
            latent_to_out_size_ratio = out_spatial_size / latent_spatial_size
            out_crop_size = np.ceil(latent_to_out_size_ratio * latent_crop_size).astype(
                int
            )
            out_crop_size += (out_spatial_size - out_crop_size) % patch_size_factor
            out_crop_low = np.floor(out_crop_size / 2).astype(int)
            out_crop_high = np.ceil(out_crop_size / 2).astype(int)
            out_pads_low_high = list(zip(-1 * out_crop_low, -1 * out_crop_high))

            # Crop all samples and move into new dictionaries.
            r = list()
            for d_i in d:
                r_i = AutoencoderSpatialSample()

                # Input sample.
                (
                    (r_i["spatial_data_in"], r_i["mask_in"], r_i["coord_grid_in"]),
                    r_i["affine_el2coord_in"],
                ) = mrinr.vols.crop_pad(
                    (
                        d_i["spatial_data_in"],
                        d_i["mask_in"],
                        mrinr.nn.coords_as_channels(
                            d_i["coord_grid_in"], has_batch_dim=False
                        ),
                    ),
                    d_i["affine_el2coord_in"],
                    *in_pads_low_high,
                )
                # Undo the coordinate to channel conversion needed for cropping.
                r_i["coord_grid_in"] = mrinr.nn.channels_as_coords(
                    r_i["coord_grid_in"], has_batch_dim=False
                )
                # Spacing is the same.
                r_i["spacing_in"] = d_i["spacing_in"]

                # Latent space sample coordinates.
                (
                    r_i["coord_grid_latent"],
                    r_i["affine_el2coord_latent"],
                ) = mrinr.vols.crop_pad(
                    mrinr.nn.coords_as_channels(
                        d_i["coord_grid_latent"], has_batch_dim=False
                    ),
                    d_i["affine_el2coord_latent"],
                    *latent_pads_low_high,
                )
                r_i["coord_grid_latent"] = mrinr.nn.channels_as_coords(
                    r_i["coord_grid_latent"], has_batch_dim=False
                )
                r_i["spacing_latent"] = d_i["spacing_latent"]

                # Output space sample.
                (
                    (r_i["spatial_data_out"], r_i["mask_out"], r_i["coord_grid_out"]),
                    r_i["affine_el2coord_out"],
                ) = mrinr.vols.crop_pad(
                    (
                        d_i["spatial_data_out"],
                        d_i["mask_out"],
                        mrinr.nn.coords_as_channels(
                            d_i["coord_grid_out"], has_batch_dim=False
                        ),
                    ),
                    d_i["affine_el2coord_out"],
                    *out_pads_low_high,
                )
                r_i["coord_grid_out"] = mrinr.nn.channels_as_coords(
                    r_i["coord_grid_out"], has_batch_dim=False
                )
                r_i["spacing_out"] = d_i["spacing_out"]

                r_i["props"] = d_i["props"]
                r.append(r_i)
        # Input size is acceptable, return the input.
        else:
            r = d

    return r


def deterministic_sample_batched_autoencoder_spatial_data(
    input_to_sample: list[DataCoordAffSet],
    latent_to_sample: list[LatentCoordAffSet],
    output_to_sample: list[DataCoordAffSet],
    out_patch_size: tuple[int, ...] | Literal["whole"] | Literal["whole_all"],
    sample_r_vs: torch.Tensor | None,
    cumulative_weight_map: list[mrinr.typing.ScalarImage | mrinr.typing.ScalarVolume]
    | None,
    affine_weight_map_el2coord: mrinr.typing.HomogeneousAffine2D
    | mrinr.typing.HomogeneousAffine3D
    | None,
    patch_size_factor: int = 8,
    latent_patch_size_factor: int = 1,
    exscribe_expansion_buffer: float = 0.51,
    max_in_spatial_size: Optional[int | tuple[int, ...]] = None,
    clamp_sample_r_vs: bool | None = True,
    pad_sample_mode: str = _DEFAULT_PAD_SAMPLE_MODE,
    batched_props: Optional[list[dict[str, typing.Any]]] = None,
) -> list[AutoencoderSpatialSample]:
    # Assert all batch sizes are the same.
    assert len(input_to_sample) == len(latent_to_sample) == len(output_to_sample)
    if sample_r_vs is not None:
        assert len(sample_r_vs) == len(output_to_sample)
    if cumulative_weight_map is not None:
        assert len(cumulative_weight_map) == len(output_to_sample)
    if affine_weight_map_el2coord is not None:
        assert affine_weight_map_el2coord.shape[0] == len(output_to_sample)
    if batched_props is not None:
        assert len(batched_props) == len(output_to_sample)

    # If 'whole_all' is specified, then no subsampling is done. Input, latent, and
    # output are all taken in their entirety.
    if out_patch_size == "whole_all":
        assert (
            max_in_spatial_size is None
        ), 'Cannot specify input size limit when "whole_all" is specified.'

        if len(output_to_sample) > 1:
            # Pad outputs, latent sizes, and inputs to the same size in each respective
            # space, if necessary.
            # Output samples.
            output_shapes = np.asarray(
                [tuple(d.spatial_data.shape[1:]) for d in output_to_sample]
            )
            output_max_shape = output_shapes.max(0)
            # Ensure shape is divisible by given patch factor.
            output_max_shape = output_max_shape + (
                patch_size_factor - (output_max_shape % patch_size_factor)
            )
            if not (output_shapes == output_max_shape).all():
                for i in range(len(output_to_sample)):
                    (d, m), a = mrinr.vols.center_crop_pad_to_shape(
                        (output_to_sample[i].spatial_data, output_to_sample[i].mask),
                        output_to_sample[i].affine_el2coord,
                        target_shape=output_max_shape,
                        return_spatial_pads=False,
                        mode=pad_sample_mode,
                    )
                    output_to_sample[i] = DataCoordAffSet(
                        spatial_data=d, mask=m, affine_el2coord=a
                    )

            # Latent samples.
            latent_shapes = np.asarray(
                [tuple(d.coord_grid.shape[:-1]) for d in latent_to_sample]
            )
            latent_max_shape = latent_shapes.max(0)
            # Ensure shape is divisible by given patch factor.
            latent_max_shape = latent_max_shape + (
                patch_size_factor - (latent_max_shape % patch_size_factor)
            )
            if not (latent_shapes == latent_max_shape).all():
                # new_output_to_sample = list()
                dummy_latent_to_sample = [
                    DataCoordAffSet(
                        spatial_data=einops.rearrange(l.coord_grid, "... d -> d ..."),
                        mask=None,
                        affine_el2coord=l.affine_el2coord,
                    )
                    for l in latent_to_sample
                ]
                for i in range(len(latent_to_sample)):
                    _, a = mrinr.vols.center_crop_pad_to_shape(
                        dummy_latent_to_sample[i].spatial_data,
                        latent_to_sample[i].affine_el2coord,
                        target_shape=latent_max_shape,
                        return_spatial_pads=False,
                        mode=pad_sample_mode,
                    )
                    latent_to_sample[i] = LatentCoordAffSet(
                        coord_grid=mrinr.coords.affine_coord_grid(a, latent_max_shape),
                        affine_el2coord=a,
                    )
                del dummy_latent_to_sample
            # Input samples.
            input_shapes = np.asarray(
                [tuple(d.spatial_data.shape[1:]) for d in input_to_sample]
            )
            input_max_shape = input_shapes.max(0)
            # Ensure shape is divisible by given patch factor.
            input_max_shape = input_max_shape + (
                patch_size_factor - (input_max_shape % patch_size_factor)
            )
            if not (input_shapes == input_max_shape).all():
                for i in range(len(input_to_sample)):
                    (d, m), a = mrinr.vols.center_crop_pad_to_shape(
                        (input_to_sample[i].spatial_data, input_to_sample[i].mask),
                        input_to_sample[i].affine_el2coord,
                        target_shape=input_max_shape,
                        return_spatial_pads=False,
                        mode=pad_sample_mode,
                    )
                    input_to_sample[i] = DataCoordAffSet(
                        spatial_data=d, mask=m, affine_el2coord=a
                    )
            try:
                del d
                del m
            except NameError:
                pass

        # Latent->output pairings
        output_coord_grids = [
            mrinr.coords.affine_coord_grid(d.affine_el2coord, d.spatial_data.shape[1:])
            for d in output_to_sample
        ]
        output_spacings = [
            mrinr.coords.spacing(d.affine_el2coord).flatten() for d in output_to_sample
        ]
        if batched_props is None:
            batched_props = [None] * len(output_to_sample)
        latent_spacings = [
            mrinr.coords.spacing(d.affine_el2coord).flatten() for d in latent_to_sample
        ]
        latent_output_paired_samples = [
            PairedSpatialSample(
                # Output sample
                spatial_data_out=d.spatial_data,
                mask_out=d.mask,
                affine_el2coord_out=d.affine_el2coord,
                coord_grid_out=g,
                spacing_out=o_s,
                # Latent sample
                affine_el2coord_in=l.affine_el2coord,
                coord_grid_in=l.coord_grid,
                spacing_in=l_s,
                mask_in=None,
                spatial_data_in=None,
                props=p,
            )
            for d, g, l, o_s, l_s, p in zip(
                output_to_sample,
                output_coord_grids,
                latent_to_sample,
                output_spacings,
                latent_spacings,
                batched_props,
                strict=True,
            )
        ]
        # Input -> latent pairings
        input_coord_grids = [
            mrinr.coords.affine_coord_grid(d.affine_el2coord, d.spatial_data.shape[1:])
            for d in input_to_sample
        ]
        input_spacings = [
            mrinr.coords.spacing(d.affine_el2coord).flatten() for d in input_to_sample
        ]
        input_latent_paired_samples = [
            PairedSpatialSample(
                # Latent sample
                spatial_data_out=None,
                mask_out=None,
                affine_el2coord_out=l.affine_el2coord,
                coord_grid_out=l.coord_grid,
                spacing_out=l_s,
                # Input sample
                spatial_data_in=d.spatial_data,
                mask_in=d.mask,
                affine_el2coord_in=d.affine_el2coord,
                coord_grid_in=g,
                spacing_in=i_s,
                props=None,
            )
            for d, g, l, i_s, l_s in zip(
                input_to_sample,
                input_coord_grids,
                latent_to_sample,
                input_spacings,
                latent_spacings,
                strict=True,
            )
        ]

    else:
        # Create a dummy spatial data for the latent space, and sample the latent and output
        # spaces as a "paired" batch.
        dummy_latent_to_sample = [
            DataCoordAffSet(
                spatial_data=einops.rearrange(l.coord_grid, "... d -> d ..."),
                mask=None,
                affine_el2coord=l.affine_el2coord,
            )
            for l in latent_to_sample
        ]
        latent_output_paired_samples = deterministic_sample_batched_paired_spatial_data(
            input_to_sample=dummy_latent_to_sample,
            output_to_sample=output_to_sample,
            out_patch_size=out_patch_size,
            sample_r_vs=sample_r_vs,
            cumulative_weight_map=cumulative_weight_map,
            affine_weight_map_el2coord=affine_weight_map_el2coord,
            patch_size_factor=patch_size_factor,
            exscribe_expansion_buffer=exscribe_expansion_buffer,
            clamp_sample_r_vs=clamp_sample_r_vs,
            batched_props=batched_props,
            input_patch_size_factor=latent_patch_size_factor,
        )
        dummy_latent_to_sample = [
            DataCoordAffSet(
                spatial_data=d["spatial_data_in"],
                mask=None,
                affine_el2coord=d["affine_el2coord_in"],
            )
            for d in latent_output_paired_samples
        ]
        # Now pair the input and latent spaces, but make sure to take the "whole" spatial
        # extent of the latent space data.
        input_latent_paired_samples = deterministic_sample_batched_paired_spatial_data(
            input_to_sample=input_to_sample,
            output_to_sample=dummy_latent_to_sample,
            out_patch_size="whole",
            patch_size_factor=patch_size_factor,
            exscribe_expansion_buffer=exscribe_expansion_buffer,
            sample_r_vs=None,
            cumulative_weight_map=None,
            affine_weight_map_el2coord=None,
            clamp_sample_r_vs=None,
            batched_props=None,
            output_patch_size_factor=latent_patch_size_factor,
        )
        del dummy_latent_to_sample

    # Now combine the two pairings into an autoncoder sample list.
    sample_dicts = list()
    for i, in_latent_pair_i, latent_out_pair_i in zip(
        range(len(input_latent_paired_samples)),
        input_latent_paired_samples,
        latent_output_paired_samples,
        strict=True,
    ):
        d_i: AutoencoderSpatialSample = dict()

        # Input sample.
        d_i["spatial_data_in"] = in_latent_pair_i["spatial_data_in"]
        d_i["mask_in"] = in_latent_pair_i["mask_in"]
        d_i["affine_el2coord_in"] = in_latent_pair_i["affine_el2coord_in"]
        d_i["coord_grid_in"] = in_latent_pair_i["coord_grid_in"]
        d_i["spacing_in"] = in_latent_pair_i["spacing_in"]

        # Latent space sample coordinates.
        d_i["affine_el2coord_latent"] = latent_out_pair_i["affine_el2coord_in"]
        d_i["coord_grid_latent"] = latent_out_pair_i["coord_grid_in"]
        d_i["spacing_latent"] = latent_out_pair_i["spacing_in"]

        # Output space sample.
        d_i["spatial_data_out"] = latent_out_pair_i["spatial_data_out"]
        d_i["mask_out"] = latent_out_pair_i["mask_out"]
        d_i["affine_el2coord_out"] = latent_out_pair_i["affine_el2coord_out"]
        d_i["coord_grid_out"] = latent_out_pair_i["coord_grid_out"]
        d_i["spacing_out"] = latent_out_pair_i["spacing_out"]

        d_i["props"] = latent_out_pair_i["props"]

        sample_dicts.append(d_i)

    # Check for input patches that are too large.
    if max_in_spatial_size is not None:
        sample_dicts = __crop_autoencoder_sample_to_max_input_shape(
            sample_dicts,
            patch_size_factor=patch_size_factor,
            max_in_spatial_size=max_in_spatial_size,
        )
    # #!-----------------------------------------------------------
    # # Debug code to visualize patches
    # from pathlib import Path

    # Path("tmp").mkdir(exist_ok=True)
    # spatial_dims = sample_dicts[0]["affine_el2coord_in"].shape[-1] - 1
    # if spatial_dims == 2:
    #     for i, d in enumerate(sample_dicts):
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             input_to_sample[i].spatial_data,
    #             input_to_sample[i].affine_el2coord,
    #             f"tmp/example_{i}_input.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             latent_to_sample[i].coord_grid.movedim(-1, 0),
    #             latent_to_sample[i].affine_el2coord,
    #             f"tmp/example_{i}_latent.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             output_to_sample[i].spatial_data,
    #             output_to_sample[i].affine_el2coord,
    #             f"tmp/example_{i}_output.nii.gz",
    #         )

    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             d["spatial_data_in"],
    #             d["affine_el2coord_in"],
    #             f"tmp/patch_example_{i}_input.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             d["coord_grid_latent"].movedim(-1, 0),
    #             d["affine_el2coord_latent"],
    #             f"tmp/patch_example_{i}_latent.nii.gz",
    #         )
    #         mrinr.data.io.save_im_as_pseudo_vol(
    #             d["spatial_data_out"],
    #             d["affine_el2coord_out"],
    #             f"tmp/patch_example_{i}_output.nii.gz",
    #         )
    #!-----------------------------------------------------------

    return sample_dicts


def erode_patch_sample_mask_by_chebyshev_dist(
    mask: mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage,
    dist_thresh: int,
    fallback_dist_quantile: float = 0.9,
) -> mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage:
    m = mask.squeeze(0).detach().cpu().numpy().astype(bool)
    ch_dist = scipy.ndimage.distance_transform_cdt(m, metric="chessboard")
    # Threshold the distance map by either the half the patch size, or some quantile
    # of distances; whichever is smaller.
    center_sample_dist_thresh = min(
        dist_thresh, np.quantile(ch_dist, fallback_dist_quantile)
    )
    m2 = ch_dist >= center_sample_dist_thresh
    m2 = (
        torch.from_numpy(m2)
        .to(dtype=mask.dtype, device=mask.device)
        .reshape(mask.shape)
    )

    return m2


def _ellipsoid_st_elem(kernel_radii: tuple[int, ...]):
    coords = np.stack(
        np.meshgrid(*[np.arange(-r, r + 1) for r in kernel_radii], indexing="xy"),
        axis=-1,
    )
    k = np.sum(coords**2 / np.asarray(kernel_radii) ** 2, -1) <= 1.0
    return k.astype(bool)


@torch.no_grad()
def erode_patch_sample_mask(
    mask: mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage,
    patch_shape: tuple[int, ...],
    metric: Literal["euclidean", "approx_euclidean", "chessboard"] = "chessboard",
    border_buffer: int | float = 0.2,
    ensure_nonzero: bool = False,
) -> mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage:
    m = mask.squeeze(0).detach().cpu().numpy().astype(bool)

    if ensure_nonzero and (m == 0).all():
        raise ValueError("Cannot erode a mask that is all zeros.")

    # Patch diameter.
    d = np.asarray(patch_shape)

    if (d <= 1).all():
        eroded_m = m
    else:
        if isinstance(border_buffer, int):
            b = border_buffer
        elif isinstance(border_buffer, float):
            b = d * border_buffer
        else:
            b = 0
        b = np.maximum(b, 0)
        d = d - b
        # Because sampling occurs at the center index, we only want to erode half of the
        # patch size.
        r1 = np.floor(d / 2)
        # Reduce by the border buffer.
        r1 = r1.round().astype(int)

        metric = str(metric).lower().replace("_", "")
        if "euclid" in metric:
            # Create an ellipsoid structuring element.
            # The ellipsoid radius should be half the target erosion radius.
            r_ellipse = r1 // 2
            # Reduce the structure by the shape GCD, and erode for GCD iterations.
            if "approx" in metric:
                gcd = np.gcd.reduce(r_ellipse)
                st_elem = _ellipsoid_st_elem(tuple(r_ellipse // gcd))
                iterations = gcd
            else:
                st_elem = _ellipsoid_st_elem(tuple(r_ellipse))
                iterations = 1

            eroded_m = scipy.ndimage.binary_erosion(
                m, structure=st_elem, mask=m, border_value=0, iterations=iterations
            )
        elif "chess" in metric:
            # A rectangular structuring element can be decomposed into n_dim arrays of the
            # size that is to be eroded.
            st_elem_shapes = np.clip(np.diagflat(r1), a_min=1, a_max=None)
            # Decompose into 1D arrays
            footprint = [(np.ones(d_i), 1) for d_i in st_elem_shapes]
            eroded_m = skimage.morphology.binary_erosion(
                m,
                footprint=footprint,
                mode="min",
            )
        else:
            raise ValueError(f"Unrecognized metric '{metric}'.")

    eroded_m = (
        torch.from_numpy(eroded_m)
        .to(dtype=mask.dtype, device=mask.device)
        .reshape(mask.shape)
    )

    if ensure_nonzero and (eroded_m == 0).all():
        # If the mask was fully eroded, reduce the patch shape by half and recursively
        # erode again.
        del eroded_m
        eroded_m = erode_patch_sample_mask(
            mask=mask,
            patch_shape=tuple(np.array(patch_shape) // 2),
            metric=metric,
            border_buffer=border_buffer,
            ensure_nonzero=ensure_nonzero,
        )

    return eroded_m


def gaussian_kernel_weight_at_center(
    spatial_shape: tuple[int, ...],
    sigma_el: float | tuple[float, ...],
    truncate: float = 4.0,
    ensure_nonzero: bool = True,
) -> mrinr.typing.SingleScalarImage | mrinr.typing.SingleScalarVolume:
    spatial_dims = len(spatial_shape)
    if isinstance(sigma_el, float):
        sigma_el = (sigma_el,) * spatial_dims
    sigma_el = np.asarray(sigma_el)

    #### Create a delta function at the center of the data.
    # Find the origin of the template space in the pixel/voxel space.
    p_orig = tuple((np.asarray(spatial_shape) - 1) // 2)
    dirac_delta = np.zeros(spatial_shape, dtype=np.float32)
    dirac_delta[p_orig] = 1.0
    k = scipy.ndimage.gaussian_filter(
        dirac_delta, sigma=sigma_el, truncate=truncate, mode="reflect", order=0
    )
    k = torch.from_numpy(k).unsqueeze(0).to(dtype=torch.float32)

    # If the kernel is all zeros, ensure that the center is non-zero.
    if ensure_nonzero and (k.sum() == 0):
        k[p_orig] = 1.0

    return k


def gaussian_kernel_weight_at_coord_orig(
    spatial_shape: tuple[int, ...],
    affine_el2coord: mrinr.typing.SingleHomogeneousAffine2D
    | mrinr.typing.SingleHomogeneousAffine3D,
    sigma_coord_units: float | tuple[float, ...],
    truncate: float = 4.0,
    ensure_nonzero: bool = True,
    # mask: Optional[torch.Tensor] = None,
) -> mrinr.typing.SingleScalarImage | mrinr.typing.SingleScalarVolume:
    # Import transforms3d here to avoid a dependency.
    import transforms3d
    import transforms3d.affines

    spatial_dims = affine_el2coord.shape[-1] - 1
    if isinstance(sigma_coord_units, float):
        sigma_coord_units = (sigma_coord_units,) * spatial_dims
    sigma_coord_units = np.asarray(sigma_coord_units)

    #### Create a delta function at the origin of the coordinate space.
    # Find the origin of the template space in the pixel/voxel space.
    #!
    # p_orig = scipy.ndimage.center_of_mass(
    #     mask.squeeze(0).cpu().numpy().astype(np.uint8)
    # )
    # p_orig = tuple(np.asarray(p_orig).flatten().round().astype(int).tolist())
    #!
    # p_orig = tuple((np.asarray(spatial_shape) - 1) // 2)
    #!
    p_orig = tuple(
        mrinr.coords.transform_coords(
            torch.zeros(spatial_dims),
            mrinr.coords.inv_affine(affine_el2coord),
        )
        .round()
        .int()
        .flatten()
        .tolist()
    )
    #!
    dirac_delta = np.zeros(spatial_shape, dtype=np.float32)
    assert (np.asarray(p_orig) >= 0).all()
    assert (np.asarray(p_orig) < np.asarray(spatial_shape)).all()
    dirac_delta[p_orig] = 1.0

    # Decompose the affine
    # "The order of transformations is therefore shears, followed by zooms, followed by
    # rotations, followed by translations."
    t, rot, zoom, shear = transforms3d.affines.decompose(affine_el2coord.cpu().numpy())
    #### Gaussian filter the delta function with sigmas scaled by the spacing of the
    #### coordinate space.
    sigma_el_units = sigma_coord_units / zoom
    k = scipy.ndimage.gaussian_filter(
        dirac_delta, sigma=sigma_el_units, truncate=truncate, mode="reflect", order=0
    )
    k = torch.from_numpy(k).unsqueeze(0).to(dtype=torch.float32)
    #### Consider the gaussian kernel in an "un-rotated" orientation, then re-rotate
    #### the kernel to match the target space.
    r_0 = np.eye(spatial_dims)
    s_0 = np.zeros_like(shear)
    axis_aligned_space_affine = transforms3d.affines.compose(T=t, R=r_0, Z=zoom, S=s_0)
    axis_aligned_space_affine = torch.from_numpy(axis_aligned_space_affine).to(
        affine_el2coord
    )

    # Resample into the original coordinate grid.
    target_coord_grid = mrinr.coords.affine_coord_grid(
        affine_el2coord, spatial_shape=spatial_shape
    )

    k_coord_aligned = mrinr.grid_resample(
        k,
        affine_x_el2coords=axis_aligned_space_affine,
        sample_coords=target_coord_grid,
        mode_or_interpolation="linear",
        padding_mode_or_bound="border",
        interp_lib="torch",
    )
    k_coord_aligned = k_coord_aligned.to(
        dtype=torch.float32, device=affine_el2coord.device
    )
    # If the kernel is all zeros, ensure that the center is non-zero.
    if ensure_nonzero and (k_coord_aligned.sum() == 0):
        k_coord_aligned[p_orig] = 1.0

    return k_coord_aligned
