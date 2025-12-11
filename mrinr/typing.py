# -*- coding: utf-8 -*-
from typing import Any

import jaxtyping
import torch

# Type alias for objects that can be parsed into standard pytorch modules, primarily
# normalization layers and activation functions.
NNModuleConstructT = str | tuple[str, dict[str, Any]] | None

# 3D
Volume = jaxtyping.Shaped[torch.Tensor, "batch channel x y z"]
MaskVolume = jaxtyping.Bool[torch.Tensor, "batch channel=1 x y z"]
ScalarVolume = jaxtyping.Float[torch.Tensor, "batch channel x y z"]
Coord3D = jaxtyping.Float[torch.Tensor, "batch dims=3"]
CoordGrid3D = jaxtyping.Float[torch.Tensor, "batch x y z coord=3"]
CoordGrid3DMask = jaxtyping.Bool[torch.Tensor, "batch x y z 1"]
WithinVolCoord3D = jaxtyping.Float[torch.Tensor, "batch space coord=3"]
HomogeneousAffine3D = jaxtyping.Float[torch.Tensor, "batch 4 4"]
FOV3D = jaxtyping.Float[torch.Tensor, "batch lower_upper=2 coord=3"]
BoundingBox3D = jaxtyping.Float[torch.Tensor, "batch lower_upper=2 coord=3"]
Corners3D = jaxtyping.Float[torch.Tensor, "batch lower_upper_xyz=8 coord=3"]
# 2D
Image = jaxtyping.Shaped[torch.Tensor, "batch channel x y"]
MaskImage = jaxtyping.Bool[torch.Tensor, "batch channel=1 x y"]
ScalarImage = jaxtyping.Float[torch.Tensor, "batch channel x y"]
Coord2D = jaxtyping.Float[torch.Tensor, "batch dims=2"]
CoordGrid2D = jaxtyping.Float[torch.Tensor, "batch x y coord=2"]
CoordGrid2DMask = jaxtyping.Bool[torch.Tensor, "batch x y 1"]
WithinImageCoord2D = jaxtyping.Float[torch.Tensor, "batch space coord=2"]
HomogeneousAffine2D = jaxtyping.Float[torch.Tensor, "batch 3 3"]
FOV2D = jaxtyping.Float[torch.Tensor, "batch lower_upper=2 coord=2"]
BoundingBox2D = jaxtyping.Float[torch.Tensor, "batch lower_upper=2 coord=2"]
Corners2D = jaxtyping.Float[torch.Tensor, "batch lower_upper_xy=4 coord=2"]

# Unbatched types, used more in preprocessing pipelines and viz.
# 3D
SingleVolume = jaxtyping.Shaped[torch.Tensor, "channel x y z"]
SingleMaskVolume = jaxtyping.Bool[torch.Tensor, "channel=1 x y z"]
SingleScalarVolume = jaxtyping.Float[torch.Tensor, "channel x y z"]
SingleCoord3D = jaxtyping.Float[torch.Tensor, "dims=3"]
SingleCoordGrid3D = jaxtyping.Float[torch.Tensor, "x y z coord=3"]
SingleCoordGrid3DMask = jaxtyping.Bool[torch.Tensor, "x y z 1"]
SingleWithinVolCoord3D = jaxtyping.Float[torch.Tensor, "space coord=3"]
SingleHomogeneousAffine3D = jaxtyping.Float[torch.Tensor, "4 4"]
SingleFOV3D = jaxtyping.Float[torch.Tensor, "lower_upper=2 coord=3"]
SingleBoundingBox3D = jaxtyping.Float[torch.Tensor, "lower_upper=2 coord=3"]
SingleCorners3D = jaxtyping.Float[torch.Tensor, "lower_upper_xyz=8 coord=3"]
# 2D
SingleImage = jaxtyping.Shaped[torch.Tensor, "channel x y"]
SingleMaskImage = jaxtyping.Bool[torch.Tensor, "channel=1 x y"]
SingleScalarImage = jaxtyping.Float[torch.Tensor, "channel x y"]
SingleCoord2D = jaxtyping.Float[torch.Tensor, "dims=2"]
SingleCoordGrid2D = jaxtyping.Float[torch.Tensor, "x y coord=2"]
SingleCoordGrid2DMask = jaxtyping.Bool[torch.Tensor, "x y 1"]
SingleWithinImageCoord2D = jaxtyping.Float[torch.Tensor, "space coord=2"]
SingleHomogeneousAffine2D = jaxtyping.Float[torch.Tensor, "3 3"]
SingleFOV2D = jaxtyping.Float[torch.Tensor, "lower_upper=2 coord=2"]
SingleBoundingBox2D = jaxtyping.Float[torch.Tensor, "lower_upper=2 coord=2"]
SingleCorners2D = jaxtyping.Float[torch.Tensor, "lower_upper_xyz=4 coord=2"]

# Combined types
# Use the notation "SD" instead of "ND" because only 2D and 3D are included here, not
# 1D, 4D, etc. "SD" stands for "S-Dimensional" or "Space-Dimensional".
AnyHomogeneousAffineSD = (
    SingleHomogeneousAffine2D
    | HomogeneousAffine2D
    | SingleHomogeneousAffine3D
    | HomogeneousAffine3D
)
AnySpatialDataSD = SingleImage | Image | SingleVolume | Volume
AnyCoordSD = (
    SingleCoord2D
    | SingleWithinImageCoord2D
    | SingleCoordGrid2D
    | Coord2D
    | WithinImageCoord2D
    | CoordGrid2D
    | SingleCoord3D
    | SingleWithinVolCoord3D
    | SingleCoordGrid3D
    | Coord3D
    | WithinVolCoord3D
    | CoordGrid3D
)
