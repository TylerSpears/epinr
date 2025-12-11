# -*- coding: utf-8 -*-
import numpy as np
import scipy
import scipy.spatial.distance
import torch

import mrinr


def join_streamlines(
    streamlines: list[np.ndarray],
    min_match_arc_len: float,
    antipodal_arc_len_thresh: float,
) -> list[np.ndarray]:
    n_streamlines = len(streamlines)
    start_coords = np.asarray([s[0] for s in streamlines])
    start_dist_mat = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(start_coords, metric="euclidean")
    )
    # Set the diagonal to inf to avoid matching a streamline to itself.
    start_dist_mat[np.diag_indices(start_dist_mat.shape[0], ndim=2)] = np.inf
    START_COORD_TOL = 1e-6
    start_dist_mat = start_dist_mat <= START_COORD_TOL
    all_row_idx = np.arange(start_dist_mat.shape[1])

    init_streamline_labels = np.arange(1, n_streamlines + 1).astype(int)
    current_streamline_labels = init_streamline_labels.copy()
    for i_stream in range(n_streamlines):
        # If this streamline has already been matched to another, then skip.
        if init_streamline_labels[i_stream] != current_streamline_labels[i_stream]:
            continue
        matching_starts = start_dist_mat[i_stream]
        if matching_starts.sum() == 0:
            continue
        tangent_xyz_start = streamlines[i_stream][1] - streamlines[i_stream][0]
        tangent_xyz_matches = np.asarray(
            [streamlines[j][1] for j in np.where(matching_starts)[0]]
        ) - np.asarray([streamlines[j][0] for j in np.where(matching_starts)[0]])
        row_idx_matches = all_row_idx[matching_starts]
        theta_start, phi_start = mrinr.coords.xyz2unit_sphere_theta_phi(
            torch.from_numpy(tangent_xyz_start).unsqueeze(0)
        )
        theta_start = theta_start.flatten()
        phi_start = phi_start.flatten()
        theta_matches, phi_matches = mrinr.coords.xyz2unit_sphere_theta_phi(
            torch.from_numpy(tangent_xyz_matches)
        )
        # Make sure the starting directions are on opposite hemispheres, to some given
        # arc length minimum.
        hemisphere_check_arc_lens = mrinr.coords.unit_sphere_arc_len(
            theta_start.unsqueeze(1),
            phi_start.unsqueeze(1),
            theta_matches.unsqueeze(0),
            phi_matches.unsqueeze(0),
        ).flatten()
        if (hemisphere_check_arc_lens < min_match_arc_len).all():
            continue

        theta_matches = theta_matches[hemisphere_check_arc_lens >= min_match_arc_len]
        phi_matches = phi_matches[hemisphere_check_arc_lens >= min_match_arc_len]
        row_idx_matches = row_idx_matches[
            (hemisphere_check_arc_lens >= min_match_arc_len).numpy()
        ]

        antipodal_dist = mrinr.coords.antipodal_unit_sphere_arc_len(
            theta_start.unsqueeze(1),
            phi_start.unsqueeze(1),
            theta_matches.unsqueeze(0),
            phi_matches.unsqueeze(0),
        )
        if (antipodal_dist > antipodal_arc_len_thresh).all():
            continue

        match_idx = row_idx_matches[np.argmin(antipodal_dist.numpy()).item()]
        current_streamline_labels[match_idx] = init_streamline_labels[i_stream]

    return_streamlines = list()
    labels, counts = np.unique(current_streamline_labels, return_counts=True)
    for l, count in zip(labels.tolist(), counts.tolist()):
        if count == 1:
            idx = np.where(current_streamline_labels == l)[0].item()
            return_streamlines.append(streamlines[idx])
        else:
            matching_streamline_indices = np.where(current_streamline_labels == l)[0]
            assert len(matching_streamline_indices) == 2
            start_streamline = streamlines[matching_streamline_indices[0]]
            end_streamline = streamlines[matching_streamline_indices[1]]
            combined = np.concatenate(
                (np.flip(start_streamline, axis=0), end_streamline[1:]), axis=0
            )
            return_streamlines.append(combined)

    return return_streamlines
