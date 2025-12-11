# -*- coding: utf-8 -*-
# Layers, networks, and functions for continuous-space autoencoders.

# import gc
import itertools
import typing
from typing import Any, Literal, Optional

import monai
import monai.networks.blocks
import monai.networks.nets
import numpy as np
import scipy
import scipy.spatial.transform
import torch

import mrinr

__all__ = [
    "DenseCoordSpace",
    "NormalizerParams",
    "ContAutoencoder",
    "ContEncoder",
    "ContDecoder",
    "ContFullyConvEncoder",
    "ContFullyConvDecoder",
]


class NormalizerParams(typing.NamedTuple):
    min_shift: torch.Tensor
    size_scale: torch.Tensor


class DenseCoordSpace(typing.NamedTuple):
    values: mrinr.typing.Image | mrinr.typing.Volume
    coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D
    affine: mrinr.typing.HomogeneousAffine2D | mrinr.typing.HomogeneousAffine3D


##### Builder functions and objects for sub-modules of *coder networks.

_FCONV_MODELS_T = Literal["fconv"]
_CONV_MODELS_T = (
    Literal[
        "carn",
        "plaincnn",
        "rescnn",
        "unet",
        "attention_unet",
    ]
    | _FCONV_MODELS_T
)


def _init_conv_coord_aware(
    conv_model: _CONV_MODELS_T,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    default_activate_fn_init_obj: Optional[str | tuple[str, dict[str, Any]]],
    encoder_or_decoder: Literal["encoder", "decoder", None] = None,
    **kwargs,
) -> tuple[torch.nn.Module, bool]:
    conv_model = str(conv_model).strip().lower().replace("-", "").replace("_", "")

    # Monai models.
    if conv_model in {"unet", "attentionunet", "unetattention"}:
        # monai uses the 'act' keyword instead of 'activate_fn'.
        if default_activate_fn_init_obj is not None:
            kwargs = {"act": default_activate_fn_init_obj} | kwargs
        # Attention UNet
        if ("attention" in conv_model) and ("unet" in conv_model):
            c = monai.networks.nets.AttentionUnet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        # Plain UNet
        elif conv_model == "unet":
            c = monai.networks.nets.UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
    # CARN/Dense CNN
    elif "carn" in conv_model:
        if default_activate_fn_init_obj is not None:
            kwargs = {"activate_fn": default_activate_fn_init_obj} | kwargs
        c = mrinr.nn.CARN(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    # Vanilla feed-forward CNN
    elif conv_model in {"cnn", "plaincnn"}:
        if default_activate_fn_init_obj is not None:
            kwargs = {"activate_fn": default_activate_fn_init_obj} | kwargs
        c = mrinr.nn.PlainCNN(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    # Residual CNN
    elif conv_model in {"rescnn", "residualcnn", "resnet"}:
        if default_activate_fn_init_obj is not None:
            kwargs = {"activate_fn": default_activate_fn_init_obj} | kwargs
        c = mrinr.nn.ResCNN(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    # Fully-convolutional networks.
    elif "fconv" in conv_model:
        if default_activate_fn_init_obj is not None:
            kwargs = {"activate_fn": default_activate_fn_init_obj} | kwargs
        if str(encoder_or_decoder).lower() == "encoder":
            c = mrinr.nn.spatial_coders.FConvEncoder(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        elif str(encoder_or_decoder).lower() == "decoder":
            c = mrinr.nn.spatial_coders.FConvDecoder(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        else:
            raise ValueError(
                "Fully-convolutional networks must specify 'encoder' or 'decoder'!"
            )
    else:
        raise ValueError(f"Invalid conv_model {conv_model}")
    # Check if conv module has an "_is_coord_aware" attribute. If not, assume False.
    try:
        conv_is_coord_aware = c._is_coord_aware
    except AttributeError:
        conv_is_coord_aware = False
    try:
        c._is_coord_aware = conv_is_coord_aware
    except AttributeError:
        pass

    return c, conv_is_coord_aware


def _init_norm(
    norm: Optional[str | tuple[str, dict[str, Any]]],
    num_channels: int,
    **kwargs,
) -> torch.nn.Module | None:
    if norm is None:
        n = None
    else:
        n = mrinr.nn.make_norm_module(
            norm, num_channels_or_features=num_channels, **kwargs
        )
    return n


# None is mapped to an "identity" resampler.
_RESAMPLER_MODELS_T = Literal[
    None,
    "identity",
    "translation",
    "interpolate",
    "inr_ensemble",
    "lte",
    "liif_conv",
    "lte_conv",
]


def _init_resampler(
    resampler_model: _RESAMPLER_MODELS_T,
    spatial_dims: int,
    in_features: int,
    out_features: int,
    encoder_or_decoder: Literal["encoder", "decoder", None] = None,
    allow_approx_infer: bool = False,
    **kwargs,
) -> torch.nn.Module:
    res_model = str(resampler_model).strip().lower().replace("-", "").replace("_", "")
    if "interp" in res_model:
        resampler = mrinr.nn.spatial_coders.InterpolationResampler(
            spatial_dims=spatial_dims,
            in_features=in_features,
            out_features=out_features,
            allow_approx_infer=allow_approx_infer,
            **kwargs,
        )
    elif "inrensemble" in res_model:
        resampler = mrinr.nn.EnsembleSpatialINR(
            spatial_dims=spatial_dims,
            in_spatial_features=in_features,
            out_features=out_features,
            allow_approx_infer=allow_approx_infer,
            **kwargs,
        )
    elif res_model == "lte":
        resampler = mrinr.nn.LTE(
            spatial_dims=spatial_dims,
            in_spatial_features=in_features,
            out_features=out_features,
            allow_approx_infer=allow_approx_infer,
            **kwargs,
        )
    elif "translat" in res_model:
        resampler = mrinr.nn.spatial_coders.TranslationOnlyResampler(
            spatial_dims=spatial_dims,
            in_features=in_features,
            out_features=out_features,
            allow_approx_infer=allow_approx_infer,
            **kwargs,
        )
    elif "lte" in res_model and "conv" in res_model:
        resampler = mrinr.nn.LTEConv(
            spatial_dims=spatial_dims,
            in_spatial_features=in_features,
            out_features=out_features,
            allow_approx_infer=allow_approx_infer,
            **kwargs,
        )
    # "None" is interpreted to be an identity resampler, such as in a fully-
    # convolutional autoencoder.
    elif res_model in {"identity", "none"}:
        resampler = mrinr.nn.spatial_coders.IdentityResampler()
    else:
        raise ValueError(f"Invalid resampler_model {resampler_model}")

    return resampler


###### Autoencoder, encoder, and decoder network classes.
class ContAutoencoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        latent_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        resampler_model: _RESAMPLER_MODELS_T,
        conv_model: _CONV_MODELS_T,
        conv_kwargs: dict[str, Any] | None,
        resampler_kwargs: dict[str, Any] | None,
        latent_activate_fn: Optional[str | tuple[str, dict[str, Any]]] = None,
        is_fully_conv: bool = False,
        is_vae: bool = False,
        is_branched: bool = False,
        latent_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        encoder_kwargs: Optional[dict[str, Any]] = None,
        decoder_kwargs: Optional[dict[str, Any]] = None,
        encoder_conv_kwargs: Optional[dict[str, Any]] = None,
        encoder_resampler_kwargs: Optional[dict[str, Any]] = None,
        decoder_conv_kwargs: Optional[dict[str, Any]] = None,
        decoder_resampler_kwargs: Optional[dict[str, Any]] = None,
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=kwargs, warn=True
        )
        super().__init__()
        self._set_common_properties(**self._init_kwargs)
        self._allow_approx_infer = allow_approx_infer

        # Init Modules in order.
        self.encoder = self._init_encoder(
            is_fully_conv=self.is_fully_conv,
            is_vae=self.is_vae,
            is_branched=self.is_branched,
            allow_approx_infer=self._allow_approx_infer,
        )
        self.latent_norm = mrinr.nn.make_norm_module(
            self._latent_norm_init_obj,
            num_channels_or_features=self.latent_channels,
        )
        self.latent_activate_fn = (
            mrinr.nn.make_activate_fn_module(self._latent_activate_fn_init_obj)
            if self._latent_activate_fn_init_obj is not None
            else None
        )
        self.decoder = self._init_decoder(
            is_fully_conv=self.is_fully_conv,
            is_branched=self.is_branched,
            allow_approx_infer=self._allow_approx_infer,
        )

    def _set_common_properties(
        self,
        spatial_dims: int,
        in_channels: int,
        latent_channels: int,
        out_channels: int,
        activate_fn,
        conv_model: Optional[str] = None,
        resampler_model: Optional[str] = None,
        conv_kwargs=None,
        resampler_kwargs=None,
        latent_activate_fn=None,
        latent_norm=None,
        is_fully_conv: bool = False,
        is_vae: bool = False,
        is_branched: bool = False,
        encoder_conv_kwargs=None,
        encoder_resampler_kwargs=None,
        decoder_conv_kwargs=None,
        decoder_resampler_kwargs=None,
        encoder_kwargs=None,
        decoder_kwargs=None,
        **ignored_kwargs,
    ) -> None:
        self._spatial_dims = spatial_dims
        self._in_channels = in_channels
        self._latent_channels = latent_channels
        self._out_channels = out_channels
        self.is_fully_conv = is_fully_conv
        self.is_vae = is_vae
        self.is_branched = is_branched
        if self.is_fully_conv:
            assert (
                not self.is_branched
            ), "Fully-convolutional networks cannot be branched."
        elif self.is_branched:
            assert not self.is_fully_conv, "Branched networks cannot be fully-conv."

        self._resampler_model = (
            str(resampler_model).lower().strip().replace("-", "")
            if resampler_model is not None
            else None
        )
        self._conv_model = (
            str(conv_model).lower().strip().replace("-", "")
            if conv_model is not None
            else None
        )

        self._activate_fn_init_obj = activate_fn
        self._latent_activate_fn_init_obj = latent_activate_fn
        self._latent_norm_init_obj = latent_norm

        self._conv_kwargs = conv_kwargs if conv_kwargs is not None else dict()
        self._resampler_kwargs = (
            resampler_kwargs if resampler_kwargs is not None else dict()
        )
        self._encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else dict()
        self._decoder_kwargs = decoder_kwargs if decoder_kwargs is not None else dict()
        self._encoder_conv_kwargs = (
            encoder_conv_kwargs if encoder_conv_kwargs is not None else dict()
        )
        self._encoder_resampler_kwargs = (
            encoder_resampler_kwargs if encoder_resampler_kwargs is not None else dict()
        )
        self._decoder_conv_kwargs = (
            decoder_conv_kwargs if decoder_conv_kwargs is not None else dict()
        )
        self._decoder_resampler_kwargs = (
            decoder_resampler_kwargs if decoder_resampler_kwargs is not None else dict()
        )

    def _init_encoder(
        self,
        is_fully_conv: bool,
        is_vae: bool,
        is_branched: bool,
        allow_approx_infer: bool,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        encoder_conv_kwargs = self._conv_kwargs | self._encoder_conv_kwargs
        encoder_resampler_kwargs = (
            self._resampler_kwargs | self._encoder_resampler_kwargs
        )
        if is_fully_conv:
            m = ContFullyConvEncoder(
                *args,
                **(
                    dict(
                        spatial_dims=self.spatial_dims,
                        in_channels=self.in_channels,
                        latent_channels=self.latent_channels,
                        activate_fn=self._activate_fn_init_obj,
                        # conv_model=self._conv_model,
                        conv_kwargs=encoder_conv_kwargs,
                        is_variational=is_vae,
                        allow_approx_infer=allow_approx_infer,
                    )
                    | self._encoder_kwargs
                    | kwargs
                ),
            )
        else:
            m = ContEncoder(
                *args,
                **(
                    dict(
                        spatial_dims=self.spatial_dims,
                        in_channels=self.in_channels,
                        latent_channels=self.latent_channels,
                        activate_fn=self._activate_fn_init_obj,
                        resampler_model=self._resampler_model,
                        conv_model=self._conv_model,
                        conv_kwargs=encoder_conv_kwargs,
                        resampler_kwargs=encoder_resampler_kwargs,
                        is_variational=is_vae,
                        allow_approx_infer=allow_approx_infer,
                    )
                    | self._encoder_kwargs
                    | kwargs
                ),
            )
        return m

    def _init_decoder(
        self,
        is_fully_conv: bool,
        is_branched: bool,
        allow_approx_infer: bool,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        decoder_conv_kwargs = self._conv_kwargs | self._decoder_conv_kwargs
        decoder_resampler_kwargs = (
            self._resampler_kwargs | self._decoder_resampler_kwargs
        )
        if is_fully_conv:
            m = ContFullyConvDecoder(
                *args,
                **(
                    dict(
                        spatial_dims=self.spatial_dims,
                        latent_channels=self.latent_channels,
                        out_channels=self.out_channels,
                        activate_fn=self._activate_fn_init_obj,
                        conv_kwargs=decoder_conv_kwargs,
                        allow_approx_infer=allow_approx_infer,
                    )
                    | self._decoder_kwargs
                    | kwargs
                ),
            )
        else:
            m = ContDecoder(
                *args,
                **(
                    dict(
                        spatial_dims=self.spatial_dims,
                        latent_channels=self.latent_channels,
                        out_channels=self.out_channels,
                        activate_fn=self._activate_fn_init_obj,
                        resampler_model=self._resampler_model,
                        conv_model=self._conv_model,
                        conv_kwargs=decoder_conv_kwargs,
                        resampler_kwargs=decoder_resampler_kwargs,
                        allow_approx_infer=allow_approx_infer,
                    )
                    | self._decoder_kwargs
                    | kwargs
                ),
            )
        return m

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def latent_channels(self):
        return self._latent_channels

    @property
    def out_channels(self):
        return self._out_channels

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def get_base_refine_shared_params(self):
        # If both of the resamplers are not spatial INRs, then all parameters are shared.
        if (
            not (
                isinstance(
                    getattr(self.encoder, "resampler", None),
                    mrinr.nn._SpatialINRBase,
                )
            )
        ) and (
            not (
                isinstance(
                    getattr(self.decoder, "resampler", None),
                    mrinr.nn._SpatialINRBase,
                )
            )
        ):
            r = (None, None, tuple(self.parameters()))
        else:
            # Collect base model parameters, usually the conv subnetworks.
            base_params = list()
            if getattr(self.encoder, "conv", None) is not None:
                base_params.extend(self.encoder.conv.parameters())
            if getattr(self.decoder, "conv", None) is not None:
                base_params.extend(self.decoder.conv.parameters())

            # Collect refine-stage parameters, adding hardcoded shared params if they are
            # defined.
            refine_params = list()
            shared_params = list()
            for coder in (self.encoder, self.decoder):
                # Check for 'resampler' and 'natresampler' attributes.
                for resampler_attr in ("resampler", "natresampler"):
                    # If the resampler is a spatial INR, then add its parameters to the
                    # resampler pool, except for possible interp+mlp merge layers.
                    if isinstance(
                        getattr(coder, resampler_attr, None), mrinr.nn._SpatialINRBase
                    ):
                        resampler_module: mrinr.nn._SpatialINRBase = getattr(
                            coder, resampler_attr
                        )
                        # Get all resampler parameters, then
                        refine_code_res_params = set(resampler_module.parameters())
                        # Put the interpolation skip parameters in the base model params
                        # pool.
                        if (
                            getattr(resampler_module, "interpolate_skip", None)
                            is not None
                        ):
                            refine_code_res_params -= set(
                                resampler_module.interpolate_skip.parameters()
                            )
                            base_params.extend(
                                resampler_module.interpolate_skip.parameters()
                            )
                            # If the interp+mlp merge strategy is a convolutional layer,
                            # then its parameters should always be updating, so shared.
                            if (
                                getattr(
                                    resampler_module,
                                    "merge_interpolate_skip_mlp_conv",
                                    None,
                                )
                                is not None
                            ):
                                refine_code_res_params -= set(
                                    resampler_module.merge_interpolate_skip_mlp_conv.parameters()
                                )
                                shared_params.extend(
                                    resampler_module.merge_interpolate_skip_mlp_conv.parameters()
                                )
                        refine_params.extend(list(refine_code_res_params))
                    # If the resampler is not a spatial inr, but is still a module, then add
                    # its params to the shared pool.
                    elif getattr(coder, resampler_attr, None) is not None:
                        shared_params.extend(
                            getattr(coder, resampler_attr).parameters()
                        )

            # If the autoencoder is a vae, then add the mu and logvar layers to the
            # shared pool.
            if self.encoder.is_variational:
                shared_params.extend(list(self.encoder.latent2mu.parameters()))
                shared_params.extend(list(self.encoder.latent2logvar.parameters()))
            # Add the remaining params such as between-subnetwork normalizations to the
            # shared pool.
            shared_params.extend(
                list(
                    set(self.parameters())
                    - set(itertools.chain(base_params, refine_params))
                )
            )
            # Remove duplicates.
            shared_params = list(set(shared_params))

            # Sort Parameters by their order in the self.parameters() iterator. Tensors
            # are hashed by their id() result, so putting them into an unordered set
            # will not guarantee the same order between runs.
            ordered_params = list(self.parameters())
            base_params = [p for p in ordered_params if p in set(base_params)]
            refine_params = [p for p in ordered_params if p in set(refine_params)]
            shared_params = [p for p in ordered_params if p in set(shared_params)]

            r = (
                tuple(base_params) if len(base_params) > 0 else None,
                tuple(refine_params) if len(refine_params) > 0 else None,
                tuple(shared_params) if len(shared_params) > 0 else None,
            )

        return r

    # Make resampler skipping an additional option, outside of the Module's init args.
    def set_skip_resampler(self, skip_resampler: bool):
        self.encoder.set_skip_resampler(skip_resampler)
        self.decoder.set_skip_resampler(skip_resampler)

    def forward(
        self,
        mode: Literal["reconstruct", "encode", "decode"] = "reconstruct",
        *args,
        return_coord_space: bool = False,
        **kwargs,
    ):
        mode = mode.strip().lower()
        if "recon" in mode:
            ret = self.reconstruct(
                *args, return_coord_space=return_coord_space, **kwargs
            )
        elif mode == "encode":
            ret = self.encode(*args, return_coord_space=return_coord_space, **kwargs)
        elif mode == "decode":
            ret = self.decode(*args, return_coord_space=return_coord_space, **kwargs)
        else:
            raise ValueError(f"ERROR: Invalid mode value {mode}")

        return ret

    def encode(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        x_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_z_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_x_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_z_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        x_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_z_grid_sizes: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
        return_posterior: bool = False,
        query_z_chunks: Optional[int] = None,
        allow_infer_chunking: bool = False,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> (
        torch.Tensor
        | DenseCoordSpace
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[DenseCoordSpace, torch.Tensor, torch.Tensor]
    ):
        z = self.encoder(
            x,
            x_coords=x_coords,
            query_z_coords=query_z_coords,
            affine_x_el2z_coords=affine_x_el2z_coords,
            affine_z_el2z_coords=affine_z_el2z_coords,
            x_coord_normalizer_params=NormalizerParams(
                min_shift=x_coord_normalizer_params[0],
                size_scale=x_coord_normalizer_params[1],
            ),
            z_coord_normalizer_params=NormalizerParams(
                min_shift=z_coord_normalizer_params[0],
                size_scale=z_coord_normalizer_params[1],
            ),
            x_grid_sizes=x_grid_sizes,
            query_z_grid_sizes=query_z_grid_sizes,
            return_coord_space=return_coord_space,
            query_z_chunks=query_z_chunks,
            allow_infer_chunking=allow_infer_chunking,
            _conv_inference_sliding_window_size=_conv_inference_sliding_window_size,
            **(dict(return_posterior=return_posterior) if self.is_vae else dict()),
        )
        # Split output objects.
        if return_posterior:
            z, mu_z, logvar_z = z
        if return_coord_space:
            z, z_coords, z_affine = z
        # Do any final transformations to get to the latent space.
        if self.latent_norm is not None:
            z = self.latent_norm(z)
        if self.latent_activate_fn is not None:
            z = self.latent_activate_fn(z)
        # Organize output tensors to the requested output format.
        if return_posterior and not return_coord_space:
            r = z, mu_z, logvar_z
        elif return_coord_space and not return_posterior:
            r = DenseCoordSpace(values=z, coords=z_coords, affine=z_affine)
        elif return_coord_space and return_posterior:
            r = (
                DenseCoordSpace(values=z, coords=z_coords, affine=z_affine),
                mu_z,
                logvar_z,
            )
        else:
            r = z

        return r

    def decode(
        self,
        z: mrinr.typing.Image | mrinr.typing.Volume,
        z_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_xp_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_z_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_xp_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        xp_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | NormalizerParams,
        z_grid_sizes: Optional[torch.Tensor] = None,
        query_xp_grid_sizes: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
        query_xp_chunks: Optional[int] = None,
        allow_infer_chunking: bool = False,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> torch.Tensor | DenseCoordSpace:
        return self.decoder(
            z,
            z_coords=z_coords,
            query_xp_coords=query_xp_coords,
            affine_z_el2z_coords=affine_z_el2z_coords,
            affine_xp_el2z_coords=affine_xp_el2z_coords,
            z_coord_normalizer_params=NormalizerParams(
                min_shift=z_coord_normalizer_params[0],
                size_scale=z_coord_normalizer_params[1],
            ),
            xp_coord_normalizer_params=NormalizerParams(
                min_shift=xp_coord_normalizer_params[0],
                size_scale=xp_coord_normalizer_params[1],
            ),
            z_grid_sizes=z_grid_sizes,
            query_xp_grid_sizes=query_xp_grid_sizes,
            return_coord_space=return_coord_space,
            query_xp_chunks=query_xp_chunks,
            allow_infer_chunking=allow_infer_chunking,
            _conv_inference_sliding_window_size=_conv_inference_sliding_window_size,
        )

    def reconstruct(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        x_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_z_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_xp_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_x_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_z_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_xp_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        x_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        xp_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | NormalizerParams,
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_z_grid_sizes: Optional[torch.Tensor] = None,
        query_xp_grid_sizes: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
        query_chunks: Optional[int] = None,
        allow_infer_chunking: bool = False,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> torch.Tensor | DenseCoordSpace:
        z, z_coords, z_affine = self.encode(
            x=x,
            x_coords=x_coords,
            query_z_coords=query_z_coords,
            affine_x_el2z_coords=affine_x_el2z_coords,
            affine_z_el2z_coords=affine_z_el2z_coords,
            x_coord_normalizer_params=x_coord_normalizer_params,
            z_coord_normalizer_params=z_coord_normalizer_params,
            x_grid_sizes=x_grid_sizes,
            query_z_grid_sizes=query_z_grid_sizes,
            query_z_chunks=query_chunks,
            _conv_inference_sliding_window_size=_conv_inference_sliding_window_size,
            return_coord_space=True,
            allow_infer_chunking=allow_infer_chunking,
        )
        # Get the actual grid sizes for z, in case the encoder altered the coordinate
        # grid of z from the given query coordinates.
        z_grid_sizes = mrinr.coords.spacing(z_affine)
        xp, xp_coords, xp_affine = self.decode(
            z=z,
            z_coords=z_coords,
            query_xp_coords=query_xp_coords,
            affine_z_el2z_coords=z_affine,
            affine_xp_el2z_coords=affine_xp_el2z_coords,
            z_coord_normalizer_params=z_coord_normalizer_params,
            xp_coord_normalizer_params=xp_coord_normalizer_params,
            z_grid_sizes=z_grid_sizes,
            query_xp_grid_sizes=query_xp_grid_sizes,
            query_xp_chunks=query_chunks,
            _conv_inference_sliding_window_size=_conv_inference_sliding_window_size,
            return_coord_space=True,
            allow_infer_chunking=allow_infer_chunking,
        )

        if return_coord_space:
            r = DenseCoordSpace(values=xp, coords=xp_coords, affine=xp_affine)
        else:
            r = xp
        return r


class ContEncoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        conv2resampler_channels: int,
        latent_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        resampler_model: _RESAMPLER_MODELS_T,
        conv_model: _CONV_MODELS_T,
        use_input_coords_as_channels: bool = False,
        resampler_kwargs: Optional[dict[str, Any]] = None,
        conv_kwargs: Optional[dict[str, Any]] = None,
        input2conv_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        conv2resampler_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        post_conv_activate_fn: bool = True,
        is_variational: bool = False,
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=kwargs, warn=True
        )
        super().__init__()
        self._set_common_properties(**self._init_kwargs)

        # Init submodules in order.
        self.input2conv_norm = self._init_input2conv_norm(
            self._input2conv_norm_init_obj, num_channels=self._in_channels
        )
        self.conv, self._conv_is_coord_aware = _init_conv_coord_aware(
            conv_model=self._conv_model,
            spatial_dims=self.spatial_dims,
            in_channels=self._in_channels,
            out_channels=self._conv2resampler_channels,
            default_activate_fn_init_obj=self._activate_fn_init_obj,
            encoder_or_decoder="encoder",
            **self._conv_kwargs,
        )
        self.conv2resampler_norm = self._init_conv2resampler_norm(
            self._conv2resampler_norm_init_obj,
            num_channels=self._conv2resampler_channels,
        )
        self.resampler = _init_resampler(
            resampler_model=self._resampler_model,
            spatial_dims=self.spatial_dims,
            in_features=self._conv2resampler_channels,
            out_features=self._latent_channels
            if not self.is_variational
            else 2 * self._latent_channels,
            encoder_or_decoder="encoder",
            allow_approx_infer=self._allow_approx_infer,
            **self._resampler_kwargs,
        )
        # Init layers for estimating mu and logvar of the latent space.
        if self.is_variational:
            if self.spatial_dims == 2:
                self.latent2mu = torch.nn.Conv2d(
                    self.latent_channels, self.latent_channels, kernel_size=1
                )
                self.latent2logvar = torch.nn.Conv2d(
                    self.latent_channels, self.latent_channels, kernel_size=1
                )
            elif self.spatial_dims == 3:
                self.latent2mu = torch.nn.Conv3d(
                    self.latent_channels, self.latent_channels, kernel_size=1
                )
                self.latent2logvar = torch.nn.Conv3d(
                    self.latent_channels, self.latent_channels, kernel_size=1
                )
        else:
            self.latent2mu = None
            self.latent2logvar = None

        self._skip_resampler = False

    def _set_common_properties(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        latent_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        use_input_coords_as_channels: bool,
        resampler_model: Optional[str],
        conv_model: str,
        conv2resampler_channels: Optional[int] = None,
        input2conv_norm=None,
        conv2resampler_norm=None,
        conv_kwargs: Optional[dict[str, Any]] = None,
        resampler_kwargs: Optional[dict[str, Any]] = None,
        post_conv_activate_fn: bool = True,
        is_variational: bool = False,
        allow_approx_infer: bool = False,
        **kwargs,
    ) -> None:
        self._spatial_dims = spatial_dims
        self._in_channels = in_channels
        self._use_input_coords_as_channels = use_input_coords_as_channels
        self.is_variational = is_variational
        if self._use_input_coords_as_channels and self._in_channels < self.spatial_dims:
            raise ValueError("in_channels must include input coordinates as channels!")
        if conv2resampler_channels is not None:
            self._conv2resampler_channels = conv2resampler_channels
        self._latent_channels = latent_channels
        if conv_model is not None:
            self._conv_model = str(conv_model)
        else:
            self._conv_model = None
        if resampler_model is not None:
            self._resampler_model = str(resampler_model)
        else:
            self._resampler_model = None
        self._allow_approx_infer = allow_approx_infer
        self._activate_fn_init_obj = activate_fn
        self._input2conv_norm_init_obj = input2conv_norm
        self._conv2resampler_norm_init_obj = conv2resampler_norm
        self._conv_kwargs = conv_kwargs if conv_kwargs is not None else dict()
        self._resampler_kwargs = (
            resampler_kwargs if resampler_kwargs is not None else dict()
        )

        # Helper layer to reshape coordinate volumes as channels.
        self.rearrange_coords_as_channels = mrinr.nn.spatial_coders.CoordsAsChannels(
            self.spatial_dims
        )
        # Create activation function module.
        self.activate_fn = mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj)
        self._post_conv_activate_fn = post_conv_activate_fn

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self.latent_channels

    @property
    def conv2resampler_channels(self):
        return self._conv2resampler_channels

    @property
    def latent_channels(self):
        return self._latent_channels

    def _init_input2conv_norm(
        self,
        norm: Optional[str | tuple[str, dict[str, Any]]],
        num_channels: int,
        **kwargs,
    ) -> torch.nn.Module | None:
        return _init_norm(norm, num_channels, **kwargs)

    def _init_conv2resampler_norm(
        self,
        norm: Optional[str | tuple[str, dict[str, Any]]],
        num_channels: int,
        **kwargs,
    ) -> torch.nn.Module | None:
        return _init_norm(norm, num_channels, **kwargs)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    # Make resampler skipping an additional option, outside of the Module's init args.
    def set_skip_resampler(self, skip_resampler: bool):
        self._skip_resampler = skip_resampler

    def forward(
        self,
        x: mrinr.typing.Image | mrinr.typing.Volume,
        x_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_z_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_x_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_z_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        x_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_z_grid_sizes: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
        return_posterior: bool = False,
        query_z_chunks: Optional[int] = None,
        allow_infer_chunking: bool = False,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> (
        torch.Tensor
        | DenseCoordSpace
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[DenseCoordSpace, torch.Tensor, torch.Tensor]
    ):
        if self._use_input_coords_as_channels:
            x_feats = torch.cat([x, self.rearrange_coords_as_channels(x_coords)], dim=1)
        else:
            x_feats = x

        if self.input2conv_norm is not None:
            z = self.input2conv_norm(x_feats)
        else:
            z = x_feats

        # If the conv object is aware of the coordinate space, pass the coordinates
        # in case it performs spatial operations that are not an identity transform.
        if self._conv_is_coord_aware:
            z, t_coords, t_affine = self.conv(
                z,
                x_coords=x_coords,
                affine_x_el2coords=affine_x_el2z_coords,
                return_coord_space=True,
            )
        else:
            z = self.conv(z)
            t_coords = x_coords
            t_affine = affine_x_el2z_coords

        if self.conv2resampler_norm is not None:
            z = self.conv2resampler_norm(z)

        if self._post_conv_activate_fn:
            z = self.activate_fn(z)

        if not self._skip_resampler:
            z, resampled_z_coords, resampled_z_affine = self.resampler(
                z,
                x_coords=t_coords,
                query_coords=query_z_coords,
                affine_x_el2coords=t_affine,
                affine_query_el2coords=affine_z_el2z_coords,
                x_grid_sizes=x_grid_sizes,
                query_grid_sizes=query_z_grid_sizes,
                x_coord_normalizer_params=NormalizerParams(
                    min_shift=x_coord_normalizer_params[0],
                    size_scale=x_coord_normalizer_params[1],
                ),
                query_coord_normalizer_params=NormalizerParams(
                    min_shift=z_coord_normalizer_params[0],
                    size_scale=z_coord_normalizer_params[1],
                ),
                max_q_chunks=query_z_chunks,
                return_coord_space=True,
                allow_infer_chunking=allow_infer_chunking,
            )

            if self.is_variational:
                # Split the latent space into mu and logvar.
                mu_z, logvar_z = torch.chunk(z, 2, dim=1)
                mu_z = mu_z.contiguous()
                logvar_z = logvar_z.contiguous()
                mu_z = self.latent2mu(mu_z)
                # Clamp to avoid numerical explosion in the KL loss.
                logvar_z = torch.clamp_max(self.latent2logvar(logvar_z), 10.0)
                # If training, randomly sample from the latent space.
                if self.training:
                    z = mu_z + torch.exp(0.5 * logvar_z) * torch.randn_like(mu_z)
                # If in eval, make the latent space deterministic.
                else:
                    z = mu_z
            else:
                mu_z = None
                logvar_z = None
        # If the resampler should be skipped, then the conv output serves as the latent
        # space. No variational layers are applied here.
        else:
            resampled_z_coords = t_coords
            resampled_z_affine = t_affine
            mu_z = None
            logvar_z = None

        will_return_posterior = self.is_variational and return_posterior
        if will_return_posterior and not return_coord_space:
            r = z, mu_z, logvar_z
        elif return_coord_space and not will_return_posterior:
            r = DenseCoordSpace(
                values=z, coords=resampled_z_coords, affine=resampled_z_affine
            )
        elif return_coord_space and will_return_posterior:
            r = (
                DenseCoordSpace(
                    values=z, coords=resampled_z_coords, affine=resampled_z_affine
                ),
                mu_z,
                logvar_z,
            )
        else:
            r = z

        return r


class ContDecoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        latent_channels: int,
        resampler2conv_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        resampler_model: _RESAMPLER_MODELS_T,
        conv_model: _CONV_MODELS_T,
        resampler_kwargs: Optional[dict[str, Any]] = None,
        conv_kwargs: Optional[dict[str, Any]] = None,
        latent2resampler_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        resampler2conv_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        pre_conv_activate_fn: bool = True,
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=kwargs, warn=True
        )
        super().__init__()
        self._set_common_properties(**self._init_kwargs)

        # Init submodules in order.
        self.latent2resampler_norm = self._init_latent2resampler_norm(
            self._latent2resampler_norm_init_obj, num_channels=self.latent_channels
        )
        self.resampler = _init_resampler(
            resampler_model=self._resampler_model,
            spatial_dims=self.spatial_dims,
            in_features=self.latent_channels,
            out_features=self.resampler2conv_channels,
            encoder_or_decoder="decoder",
            allow_approx_infer=self._allow_approx_infer,
            **self._resampler_kwargs,
        )
        self.resampler2conv_norm = self._init_resampler2conv_norm(
            self._resampler2conv_norm_init_obj,
            num_channels=self.resampler2conv_channels,
        )
        self.conv, self._conv_is_coord_aware = _init_conv_coord_aware(
            conv_model=self._conv_model,
            spatial_dims=self.spatial_dims,
            in_channels=self.resampler2conv_channels,
            out_channels=self.out_channels,
            default_activate_fn_init_obj=self._activate_fn_init_obj,
            encoder_or_decoder="decoder",
            **self._conv_kwargs,
        )

        self._skip_resampler = False

    def _set_common_properties(
        self,
        *,
        spatial_dims: int,
        latent_channels: int,
        resampler2conv_channels: Optional[int],
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        resampler_model: str,
        conv_model: str,
        latent2resampler_norm,
        resampler2conv_norm,
        conv_kwargs: Optional[dict[str, Any]],
        resampler_kwargs: Optional[dict[str, Any]],
        pre_conv_activate_fn: bool,
        allow_approx_infer: bool = False,
        **kwargs,
    ) -> None:
        self._spatial_dims = spatial_dims
        self._latent_channels = latent_channels
        if resampler2conv_channels is not None:
            self._resampler2conv_channels = resampler2conv_channels
        self._out_channels = out_channels
        self._conv_model = str(conv_model) if conv_model is not None else conv_model
        self._resampler_model = (
            str(resampler_model) if resampler_model is not None else resampler_model
        )
        self._allow_approx_infer = allow_approx_infer
        self._activate_fn_init_obj = activate_fn
        self._latent2resampler_norm_init_obj = latent2resampler_norm
        self._resampler2conv_norm_init_obj = resampler2conv_norm
        self._conv_kwargs = conv_kwargs if conv_kwargs is not None else dict()
        self._resampler_kwargs = (
            resampler_kwargs if resampler_kwargs is not None else dict()
        )

        # Helper layer to reshape coordinate volumes as channels.
        self.rearrange_coords_as_channels = mrinr.nn.spatial_coders.CoordsAsChannels(
            self.spatial_dims
        )
        # Create activation function module.
        self.activate_fn = mrinr.nn.make_activate_fn_module(self._activate_fn_init_obj)
        self._pre_conv_activate_fn = pre_conv_activate_fn

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def latent_channels(self):
        return self._latent_channels

    @property
    def resampler2conv_channels(self):
        return self._resampler2conv_channels

    @property
    def out_channels(self):
        return self._out_channels

    def _init_latent2resampler_norm(
        self,
        norm: Optional[str | tuple[str, dict[str, Any]]],
        num_channels: int,
        **kwargs,
    ) -> torch.nn.Module | None:
        return _init_norm(norm, num_channels, **kwargs)

    def _init_resampler2conv_norm(
        self,
        norm: Optional[str | tuple[str, dict[str, Any]]],
        num_channels: int,
        **kwargs,
    ) -> torch.nn.Module | None:
        return _init_norm(norm, num_channels, **kwargs)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    # Make resampler skipping an additional option, outside of the Module's init args.
    def set_skip_resampler(self, skip_resampler: bool):
        self._skip_resampler = skip_resampler

    def forward(
        self,
        z: mrinr.typing.Image | mrinr.typing.Volume,
        z_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_xp_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_z_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_xp_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        xp_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | NormalizerParams,
        z_grid_sizes: Optional[torch.Tensor] = None,
        query_xp_grid_sizes: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
        query_xp_chunks: Optional[int] = None,
        allow_infer_chunking: bool = False,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> torch.Tensor | DenseCoordSpace:
        if self.latent2resampler_norm is not None and not self._skip_resampler:
            xp = self.latent2resampler_norm(z)
        else:
            xp = z

        if not self._skip_resampler:
            # Check if the conv module needs different coordinates than the query coordinate
            # space. If so, the output coordinates for the resampler must be transformed to
            # be compatible with the conv module.
            if getattr(self.conv, "needs_coord_alignment", False):
                affine_q, q_coords = self.conv.align_target_coords_to_input(
                    target_affine_el2coords=affine_xp_el2z_coords,
                    target_coords=query_xp_coords,
                )
                q_grid_sizes = mrinr.coords.spacing(affine_q)
            else:
                affine_q = affine_xp_el2z_coords
                q_coords = query_xp_coords
                q_grid_sizes = query_xp_grid_sizes

            xp, t_coords, t_affine = self.resampler(
                xp,
                x_coords=z_coords,
                query_coords=q_coords,
                affine_x_el2coords=affine_z_el2z_coords,
                affine_query_el2coords=affine_q,
                x_grid_sizes=z_grid_sizes,
                query_grid_sizes=q_grid_sizes,
                x_coord_normalizer_params=NormalizerParams(
                    min_shift=z_coord_normalizer_params[0],
                    size_scale=z_coord_normalizer_params[1],
                ),
                query_coord_normalizer_params=NormalizerParams(
                    min_shift=xp_coord_normalizer_params[0],
                    size_scale=xp_coord_normalizer_params[1],
                ),
                max_q_chunks=query_xp_chunks,
                return_coord_space=True,
                allow_infer_chunking=allow_infer_chunking,
            )
            if self.resampler2conv_norm is not None:
                xp = self.resampler2conv_norm(xp)
            if self._pre_conv_activate_fn:
                xp = self.activate_fn(xp)
        else:
            t_coords = z_coords
            t_affine = affine_z_el2z_coords
        # If the conv object is aware of the coordinate space, pass the coordinates
        # in case it performs spatial operations that are not an identity transform.
        if self._conv_is_coord_aware:
            xp, decoded_coords, decoded_affine = self.conv(
                xp,
                x_coords=t_coords,
                affine_x_el2coords=t_affine,
                return_coord_space=True,
            )
            # If the conv output has coordinates not aligned with the query coordinates,
            # use the conv module's alignment function to transform the conv output to
            # the correct query coordinates.
            if hasattr(self.conv, "align_output_to_target"):
                xp, decoded_coords, decoded_affine = self.conv.align_output_to_target(
                    y=xp,
                    y_coords=decoded_coords,
                    affine_y=decoded_affine,
                    affine_target=affine_xp_el2z_coords,
                    target_coords=query_xp_coords,
                    return_coord_space=True,
                )
        else:
            xp = self.conv(xp)
            decoded_coords = t_coords
            decoded_affine = t_affine

        if return_coord_space:
            r = DenseCoordSpace(values=xp, coords=decoded_coords, affine=decoded_affine)
        else:
            r = xp
        return r


class ContFullyConvEncoder(ContEncoder):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        latent_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        conv_kwargs: Optional[dict[str, Any]] = None,
        input2conv_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        conv2latent_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        is_variational: bool = False,
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        locals_copy = locals().copy()
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            latent_channels=latent_channels,
            activate_fn=activate_fn,
            conv_model="fconv",
            conv_kwargs=conv_kwargs,
            input2conv_norm=input2conv_norm,
            conv2resampler_norm=conv2latent_norm,
            resampler_model="translation",
            conv2resampler_channels=latent_channels
            if not is_variational
            else 2 * latent_channels,
            resampler_kwargs=None,
            use_input_coords_as_channels=False,
            post_conv_activate_fn=False,
            is_variational=is_variational,
            allow_approx_infer=allow_approx_infer,
            # Post conv activation is not necessary, as the resampler does not change
            # the spatial data intensities, only the field of view.
        )
        # Override init kwargs from the super class.
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals_copy, extra_kwargs_dict=kwargs, warn=True
        )
        # The conv module must be able to handle coordinate transformations.
        assert (
            self._conv_is_coord_aware
        ), "The Conv module must be able to handle coordinate transformations."


class ContFullyConvDecoder(ContDecoder):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        latent_channels: int,
        out_channels: int,
        activate_fn: str | tuple[str, dict[str, Any]],
        conv_kwargs: Optional[dict[str, Any]] = None,
        latent2conv_norm: Optional[str | tuple[str, dict[str, Any]]] = None,
        allow_approx_infer: bool = False,
        **kwargs,
    ):
        locals_copy = locals().copy()
        super().__init__(
            spatial_dims=spatial_dims,
            latent_channels=latent_channels,
            out_channels=out_channels,
            activate_fn=activate_fn,
            conv_model="fconv",
            conv_kwargs=conv_kwargs,
            resampler2conv_norm=latent2conv_norm,
            resampler_model="identity",
            resampler2conv_channels=latent_channels,
            resampler_kwargs=None,
            pre_conv_activate_fn=False,
            allow_approx_infer=allow_approx_infer,
        )
        # Override init kwargs from the super class.
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals_copy, extra_kwargs_dict=kwargs, warn=True
        )
        # The conv module must be able to handle coordinate transformations.
        assert (
            self._conv_is_coord_aware
        ), "The Conv module must be able to handle coordinate transformations."

        # The latent -> conv transform should be identity, but a translation resampler
        # after the conv module will align the conv output to the query coordinates.
        self.output_resampler = _init_resampler(
            "translation",
            spatial_dims=self.spatial_dims,
            in_features=self.out_channels,
            out_features=self.out_channels,
            encoder_or_decoder="decoder",
            allow_approx_infer=self._allow_approx_infer,
            **self._resampler_kwargs,
        )

    def forward(
        self,
        z: mrinr.typing.Image | mrinr.typing.Volume,
        z_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        query_xp_coords: mrinr.typing.CoordGrid2D | mrinr.typing.CoordGrid3D,
        affine_z_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        affine_xp_el2z_coords: mrinr.typing.HomogeneousAffine2D
        | mrinr.typing.HomogeneousAffine3D,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor] | NormalizerParams,
        xp_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | NormalizerParams,
        z_grid_sizes: Optional[torch.Tensor] = None,
        query_xp_grid_sizes: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
        query_xp_chunks: Optional[int] = None,
        allow_infer_chunking: bool = False,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> torch.Tensor | DenseCoordSpace:
        # Call the parent class forward method with the identity resampler, before
        # translating with the output resampler.
        xp, decoded_coords, decoded_affine = super().forward(
            z=z,
            z_coords=z_coords,
            query_xp_coords=query_xp_coords,
            affine_z_el2z_coords=affine_z_el2z_coords,
            affine_xp_el2z_coords=affine_xp_el2z_coords,
            z_coord_normalizer_params=z_coord_normalizer_params,
            xp_coord_normalizer_params=xp_coord_normalizer_params,
            z_grid_sizes=z_grid_sizes,
            query_xp_grid_sizes=query_xp_grid_sizes,
            return_coord_space=True,
            query_xp_chunks=query_xp_chunks,
            _conv_inference_sliding_window_size=_conv_inference_sliding_window_size,
        )

        # Translate the output space to exactly match the query coordinates.
        # If in eval mode, this may interpolate the output.
        xp, decoded_coords, decoded_affine = self.output_resampler(
            xp,
            x_coords=decoded_coords,
            query_coords=query_xp_coords,
            affine_x_el2coords=decoded_affine,
            affine_query_el2coords=affine_xp_el2z_coords,
            return_coord_space=True,
        )

        if return_coord_space:
            r = DenseCoordSpace(values=xp, coords=decoded_coords, affine=decoded_affine)
        else:
            r = xp
        return r


def _interp_affines_shapes(
    affine1: mrinr.typing.SingleHomogeneousAffine3D,
    spatial_shape1: tuple[int, ...],
    affine2: mrinr.typing.SingleHomogeneousAffine3D,
    spatial_shape2: tuple[int, ...],
    n: int,
    shear_tol=1e-5,
    rotate_tol=1e-5,
) -> list[tuple[torch.Tensor, tuple[int, ...]]]:
    import transforms3d.affines

    A1 = affine1.detach().cpu().numpy()
    T1, R1, Z1, S1 = transforms3d.affines.decompose44(A1)
    A2 = affine1.detach().cpu().numpy()
    T2, R2, Z2, S2 = transforms3d.affines.decompose44(A2)

    for s in (S1, S2):
        if (np.abs(s) > shear_tol).any():
            raise NotImplementedError(
                f"ERROR: Shear transforms not supported, given shear params {s}"
            )

    # Linearly interpolate the translation and scale components.
    t = np.linspace(0.0, 1.0, n + 2, endpoint=True)[1:-1]
    # Spatial shapes, which will handle the scale components.
    spatial_shape_t = np.asarray(
        [
            np.interp(t, [0.0, 1.0], [shape1, shape2])
            for shape1, shape2 in zip(spatial_shape1, spatial_shape2)
        ]
    )
    spatial_shape_t = np.round(spatial_shape_t).astype(int).T

    # Translations
    T_t = np.asarray(
        [np.interp(t, [0.0, 1.0], [tr1, tr2]) for tr1, tr2 in zip(T1, T2)]
    ).T
    # Rotations
    # If rotations are close, don't perform slerp.
    if (np.abs(R1 - R2) <= rotate_tol).all():
        R_t = np.stack([R1] * n, axis=0)
    else:
        rots = scipy.spatial.transform.Rotation.from_matrix([R1, R2])
        slerp = scipy.spatial.transform.Slerp([0.0, 1.0], rots)
        R_t = slerp(t).as_matrix()

    # Compose the interpolation transformations.
    affine_tm1 = A1
    spatial_shape_tm1 = spatial_shape1
    interp_affine_shape = list()
    print(spatial_shape_t)
    print("---")
    print(T_t)
    print("---")
    print(R_t)
    for i_t in range(n):
        # Apply interpolated transforms.
        tf_affine = transforms3d.affines.compose(T_t[i_t], R_t[i_t], Z=np.ones_like(Z1))
        inter_affine = np.dot(tf_affine, affine_tm1)
        spatial_shape_t_i = tuple(spatial_shape_t[i_t])
        affine_t = mrinr.coords.resize_affine(
            torch.from_numpy(inter_affine).to(affine1),
            in_spatial_shape=spatial_shape_tm1,
            target_spatial_shape=spatial_shape_t_i,
            centered=True,
            return_coord_grid=False,
        )
        interp_affine_shape.append((affine_t, spatial_shape_t_i))
        affine_tm1 = affine_t.cpu().numpy()
        spatial_shape_tm1 = spatial_shape_t_i

    return interp_affine_shape
