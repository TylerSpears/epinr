# -*- coding: utf-8 -*-
# isort: skip_file

from ._affine import *  # noqa: F403

# from ._affine import (
#     _BatchVoxRealAffineSpace,
#     _canonicalize_coords_3d_affine,
#     _fov_coord_grid,
#     _reorient_affine_space,
#     _transform_affine_space,
#     _VoxRealAffineSpace,
#     RealAffineSpaceVol,
#     affine_coordinate_grid,
#     affine_vox2normalized_grid,
#     fov_bb_coords_from_vox_shape,
#     inv_affine,
#     scale_fov_spacing,
#     transform_coords,
#     vox_shape_from_fov,
# )
from ._sphere import (
    AT_POLE_EPS,
    MAX_COS_SIM,
    MAX_PHI,
    MAX_THETA,
    MAX_UNIT_ARC_LEN,
    MIN_COS_SIM,
    MIN_PHI,
    MIN_THETA,
    MIN_UNIT_ARC_LEN,
    _adjacent_sphere_points_idx,
    _antipodal_sym_arc_len,
    _antipodal_sym_pairwise_arc_len,
    antipodal_sphere_coords,
    antipodal_unit_sphere_arc_len,
    antipodal_xyz_coords,
    unit_sphere2xyz,
    unit_sphere_arc_len,
    xyz2unit_sphere_theta_phi,
)
