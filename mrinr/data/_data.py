# -*- coding: utf-8 -*-
import collections
import math
import multiprocessing.managers
import warnings
from typing import Callable, Optional

import monai
import monai.data
import monai.transforms
import numpy as np
import scipy
import skimage
import torch

import mrinr

__all__ = [
    "LazyLIFOCachedList",
    "DictDataset",
    "WithinSpatialDataPatchSampleDataset",
    "_CallablePromisesList",
    "LazyDict",
    "pad_list_data_collate_tensor",
    "pair_named_images_by_norm_mutual_information",
]


class LazyLIFOCachedList(collections.UserList):
    """Utility class that calls a callable when using indexing/using __getitem__.
    Used for lazily accessing items in a list of objects/containers.
    """

    def __init__(
        self,
        *args,
        cache_maxsize: int | float,
        share_cache_mp_manager: Optional[multiprocessing.managers.SyncManager] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if share_cache_mp_manager is not None:
            self._lifo_cache = share_cache_mp_manager.dict()
        else:
            self._lifo_cache = dict()
        if cache_maxsize in {math.inf, "inf"}:
            self._cache_maxsize = math.inf
        else:
            self._cache_maxsize = abs(int(round(cache_maxsize)))

    def _try_push_to_cache(self, idx, value):
        if idx not in self._lifo_cache and self._cache_maxsize > 0:
            while len(self._lifo_cache) >= self._cache_maxsize:
                self._lifo_cache.popitem()
            self._lifo_cache[idx] = value

    def __getitem__(self, idx):
        if callable(self.data[idx]):
            ret = self._lifo_cache.get(idx, self.data[idx]())
            self._try_push_to_cache(idx, ret)
        else:
            # There is no need to cache, as the object is already in the data list.
            ret = self.data[idx]
        return ret


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dicts):
        super().__init__()
        if not isinstance(dataset_dicts, (list, tuple, set)):
            dataset_dicts = (dataset_dicts,)
        self._data = tuple(dataset_dicts)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class WithinSpatialDataPatchSampleDataset(
    monai.transforms.Randomizable, monai.data.IterableDataset
):
    def __init__(
        self,
        data,
        patch_sample_fn: Callable[
            ["mrinr.data.PreprocedTemplateAffAlignedSpatialData", int],
            list["mrinr.data.EncodeDecodeTemplateAffAlignedSample"],
        ],
        batch_size: int,
        batches_per_iter_exhaust: int,
        exhaust_data_before_repeat: bool = True,
        seed: int = 0,
    ):
        """Iterable dataset for sampling patches from spatially aligned data.

        Parameters
        ----------
        data : Sequential
            Sequence of spatially aligned data objects that may be integer indexed.
        patch_sample_fn : Callable

            Callable that samples patches from a spatially aligned data object.

            The callable should take two arguments:
                - A spatial data object (a 2D image or 3D volume) or container
                  of such objects
                - The number of patches to sample from this object.
            It should return a list of samples, where each sample contains patches
            from the original input.

            Full type annotation:
                Callable[
                    [mrinr.data.PreprocedTemplateAffAlignedSpatialData, int],
                    list[mrinr.data.EncodeDecodeTemplateAffAlignedSample],
                ]
        batch_size : int
            Number of patches to sample per iteration.
        batches_per_iter_exhaust : int
            Number of batches to provide before exhausting __iter__, by default None

            This is analogous to the number of batches provided in a single epoch.
        exhaust_data_before_repeat : bool, optional
            Sample indices from data without replacement, by default True

            This option determines whether the random sampling of indices from 'data'
            will be sampled with or without replacement. If the number of requested
            batches is 's x len(data)' and this argument is True, then each item in
            'data' will be sampled at least 'floor(s)' times; this is valid even if
            0.0 < s < 1.0. If False, then sampling will occur with replacement.
        seed : int, optional
            Initial seed for index sampling randomization, by default 0
        """
        super().__init__(data=data)
        self._data = data
        self._patch_sample_fn = patch_sample_fn
        self._batch_size = int(batch_size)
        self.seed = seed

        self._n_batches_per_iter_exhaust = int(math.floor(batches_per_iter_exhaust))
        self._exhaust_data_before_repeat = exhaust_data_before_repeat

        self._all_data_idx = np.arange(len(self._data))
        # 'self.randomize()' initializes/populates 'self._data_idx_queue'
        self._data_idx_queue = None
        super().set_random_state(seed=self.seed)

    @property
    def n_batches_per_iter_exhaust(self):
        return self._n_batches_per_iter_exhaust

    def randomize(self) -> None:
        # If all data indices must be exhausted before repeating, then copy all indices
        # 'floor(n_batches / len(data))' times, sub-sample for the remaining
        # 'n_batches % len(data)' batches, and shuffle the entire list.
        if self._exhaust_data_before_repeat:
            n_full_repeats = math.floor(
                self._n_batches_per_iter_exhaust / len(self._data)
            )
            q = np.repeat(self._all_data_idx, repeats=n_full_repeats)
            n_remaining_batches = self._n_batches_per_iter_exhaust % len(self._data)
            q_remaining = self.R.choice(
                self._all_data_idx,
                size=n_remaining_batches,
                replace=False,
            )
            q = np.concatenate((q, q_remaining))
            self._data_idx_queue = self.R.permutation(q)
        else:
            self._data_idx_queue = self.R.choice(
                self._all_data_idx, size=self._n_batches_per_iter_exhaust, replace=True
            )
        assert len(self._data_idx_queue) == self._n_batches_per_iter_exhaust

    def __iter__(self):
        self.seed += 1
        super().set_random_state(seed=self.seed)  # make all workers in sync
        self.randomize()

        # Enable multiprocessing for yielding samples.
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0

        for i, data_idx in enumerate(self._data_idx_queue):
            if i % num_workers == id:
                sample = self._patch_sample_fn(self._data[data_idx], self._batch_size)
                for s in sample:
                    yield s


class _CallablePromisesList(collections.UserList):
    """Utility class that calls a callable when using indexing/using __getitem__.

    Used for lazily accessing items in a (potentially large) list of (potentially
    large) objects/containers.
    """

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ret = list()
            for i in idx:
                if callable(self.data[i]):
                    ret_i = self.data[i]()
                    # Cache any items already accessed.
                    self.data[i] = ret_i
                    ret.append(ret_i)
                else:
                    ret.append(self.data[i])
        else:
            if callable(self.data[idx]):
                ret = self.data[idx]()
                # Cache any items already accessed.
                self.data[idx] = ret
            else:
                ret = self.data[idx]
        return ret


class LazyDict(collections.UserDict):
    """Utility class that calls a callable when using indexing/using __getitem__.

    Used for lazily accessing items in a dictionary of objects/containers.
    """

    def __init__(self, cache: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache

    def __getitem__(self, key):
        if callable(self.data[key]):
            ret = self.data[key]()
            if self._cache:
                self.data[key] = ret
        else:
            ret = self.data[key]
        return ret


def pad_list_data_collate_tensor(batch, *args, **kwargs):
    """Wrapper around monai's `pad_list_data_collate` that maintains Tensor objects."""
    ret = monai.data.utils.pad_list_data_collate(batch, *args, **kwargs)

    keys = tuple(ret.keys())
    for k in keys:
        v = ret[k]
        if isinstance(v, monai.data.MetaObj):
            ret[k] = monai.utils.convert_to_tensor(v, track_meta=False)

    return ret


def pair_named_images_by_norm_mutual_information(
    v1s: dict[str, torch.Tensor],
    v2s: dict[str, torch.Tensor],
    min_norm_mutual_information: float,
    bins: int = 100,
) -> tuple[
    dict[str, str],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Match two sets of named volumes pairwise according to their norm mutual info.

    Parameters
    ----------
    v1s : dict[str, torch.Tensor]
        Volumes in group 1 with names mapped to the volume values themselves.
    v2s : dict[str, torch.Tensor]
        Volumes in group 2 with names mapped to the volume values themselves.
    min_norm_mutual_information : float
        Minimum normal mutual information value to consider as matches, in [1.0, 2.0)
    bins : int, optional
        Bins passed to `skimage.metrics.normalized_mutual_information, by default 100

    Returns
    -------
    tuple[ dict[str, str], tuple[str, ...], tuple[str, ...], ]
        Matched names group1->group2, unmatched group 1 names, unmatched group2 names
    """
    v1_numpy = dict()
    for k, v in v1s.items():
        if torch.is_tensor(v):
            v_ = v.detach().cpu().numpy()
        else:
            v_ = np.asarray(v)
        v1_numpy[k] = v_
    v2_numpy = dict()
    for k, v in v2s.items():
        if torch.is_tensor(v):
            v_ = v.detach().cpu().numpy()
        else:
            v_ = np.asarray(v)
        v2_numpy[k] = v_

    nmi_pairwise_mat = list()
    for k_i, v1_i in v1_numpy.items():
        pairwise_row_i = list()
        for k_j, v2_j in v2_numpy.items():
            nmi = skimage.metrics.normalized_mutual_information(v1_i, v2_j, bins=bins)
            if nmi < min_norm_mutual_information:
                nmi = -1
            pairwise_row_i.append(nmi)
        nmi_pairwise_mat.append(pairwise_row_i)

    nmi_mat = np.asarray(nmi_pairwise_mat)
    if (nmi_mat == -1.0).all():
        warnings.warn("WARNING No matches were found for volumes!")
        row_ind = np.asarray([])
        col_ind = np.asarray([])
    else:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(nmi_mat, maximize=True)

    v1_names = list(v1s.keys())
    v2_names = list(v2s.keys())
    unmatched_v1_names = set(v1_names)
    unmatched_v2_names = set(v2_names)
    matches = dict()
    for r, c in zip(row_ind, col_ind):
        nmi_rc = nmi_mat[r, c].sum()
        if nmi_rc == -1.0:
            continue
        else:
            v1_name = v1_names[r]
            v2_name = v2_names[c]
            unmatched_v1_names = unmatched_v1_names - {v1_name}
            unmatched_v2_names = unmatched_v2_names - {v2_name}
            matches[v1_name] = v2_name

    return matches, tuple(unmatched_v1_names), tuple(unmatched_v2_names)
