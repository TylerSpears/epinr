# -*- coding: utf-8 -*-
#### Template-space aligned volume utilities.

import torch

import mrinr

__all__ = [
    # "RobustScaleParams",
    # "EMPTY_SESSION",
    # "EMPTY_RUN",
    # "LoadedTemplateAffAlignedSpatialData",
    # "PreprocedTemplateAffAlignedSpatialData",
    # "EncodeDecodeTemplateAffAlignedSample",
    # "EncodeDecodeTemplateAffAlignedBatch",
    # "LoadedSubjSpatialData",
    # "PreprocedSubjSpatialData",
]


###### 2D/3D template-aligned data dicts.


######### Containers for general coordinate-oriented spatial data.
# class LoadedSubjSpatialData(TypedDict):
#     subj_id: str
#     spatial_data: mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
#     mask: mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage
#     affine_el2coords: (
#         mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
#     )
#     header: Optional[dict]
#     spatial_data_path: Path
#     dataset_name: str
#     session: str
#     run: str


# class PreprocedSubjSpatialData(TypedDict):
#     subj_id: str
#     spatial_data: mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
#     mask: mrinr.typing.SingleMaskVolume | mrinr.typing.SingleMaskImage
#     affine_el2coords: (
#         mrinr.typing.SingleHomogeneousAffine3D | mrinr.typing.SingleHomogeneousAffine2D
#     )
#     spacing: torch.Tensor
#     dataset_name: str
#     session: str
#     run: str
#     spatial_sampling_cumulative_weights: (
#         mrinr.typing.SingleScalarVolume | mrinr.typing.SingleScalarImage
#     )
#     spatial_data_scale_params: Optional[RobustScaleParams]
