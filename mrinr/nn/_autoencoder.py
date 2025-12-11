# -*- coding: utf-8 -*-
# Class definitions for autoencoder networks.

__all__ = [
    "InterpolResampler",
    "CMMCINREnsembleAutoencoder",
    # "CMCCMCINREnsembleAutoencoder",
    # "CMMCINRNeighborhoodAutoencoder",
    "_NormalizerParams",
]

import collections
import functools
import gc
import warnings
from typing import Any, Literal, Optional, Union

import einops
import monai
import monai.networks.nets
import numpy as np
import torch

import mrinr

_NormalizerParams = collections.namedtuple(
    "_NormalizerParams", ["min_shift", "size_scale"]
)


class InterpolResampler(torch.nn.Module):
    COORD_NDIM = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode,
        padding_mode: Optional[str] = "zeros",
        interpol_bound=None,
        override_out_of_bounds_val: Optional[float] = None,
        add_out_noise_std: Optional[float] = None,
        **kwargs,
    ):
        self._init_kwargs = {
            k: v
            for k, v in filter(
                lambda k_v: k_v[0] not in {"self", "__class__"}, locals().items()
            )
        }
        super().__init__()
        # Allow kwargs to catch extra init parameters, but warn if a non-None kwarg is
        # being sent over.
        if len(kwargs.keys()) > 0 and any(
            map(lambda v: v is not None, kwargs.values())
        ):
            warnings.warn(f"init ignoring extra kwargs {kwargs}")

        # Set up conv layer to create a learned linear combination ('interpolation') in
        # the channel dimension, while interp does the combination in the spatial
        # dimension.
        self.conv = torch.nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.mode = mode
        self.padding_mode = padding_mode
        self.interpol_bound = interpol_bound
        self.override_out_of_bounds_val = override_out_of_bounds_val
        self.add_out_noise_std = add_out_noise_std

        # Set for compatability with INR models.
        self.latent_features = in_channels
        self.out_features = out_channels

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(
        self,
        x: mrinr.typing.Volume,
        x_coords: Any,  # Ignored, for compatibility with INR3DEnsembledResidualDecoder
        query_coords: mrinr.typing.CoordGrid3D,
        affine_vox2coords: mrinr.typing.HomogeneousAffine3D,
        *args,
        **kwargs,
    ):
        batch_size = x.shape[0]
        # All coords and coord grids must be *coordinate-last* format.
        q = einops.rearrange(
            query_coords, "b ... c -> b (...) c", b=batch_size, c=self.COORD_NDIM
        )
        q = einops.rearrange(
            q,
            "b (x y z) c -> b x y z c",
            x=query_coords.shape[1],
            y=query_coords.shape[2],
            z=query_coords.shape[3],
        )
        y = self.conv(x)
        y = mrinr.vols.sample_vol(
            vol=y,
            coords_mm_xyz=q,
            affine_vox2mm=affine_vox2coords,
            mode=self.mode,
            padding_mode=self.padding_mode,
            interpol_bound=self.interpol_bound,
            align_corners=True,
            override_out_of_bounds_val=self.override_out_of_bounds_val,
        )
        # Add noise to the output, if the network is training and noise std was given.
        if (
            self.training
            and (self.add_out_noise_std is not None)
            and (self.add_out_noise_std > 0)
        ):
            eta = torch.zeros_like(y).normal_(mean=0, std=self.add_out_noise_std)
            y = y + eta
        return y


class CMMCINREnsembleAutoencoder(torch.nn.Module):
    COORD_NDIM = 3
    _MLP_SHORTCUT_INTERP_MODE = "trilinear"
    _MLP_SHORTCUT_PADDING_MODE = "border"

    def __init__(
        self,
        in_channels: int,
        conv2mlp_channels: int,
        latent_channels: int,
        out_channels: int,
        activate_fn: str,
        use_input_coords_as_channels: bool,
        conv_kwargs: dict[str, Any],
        inr_kwargs: dict[str, Any],
        inr_model: Literal["ensemble", "neighborhood", "interpolate"] = "ensemble",
        conv_model: Literal["carn"] = "carn",
        latent_carn_kwargs: Optional[dict[str, Any]] = None,
        latent_activate_fn: str = "tanh",
        use_mlp_shortcut: bool = False,
        quant_reg_bins: Optional[int] = None,
        activate_fn_kwargs: Optional[dict[str, Any]] = None,
        norm_layer: Optional[Literal["batch", "group", "instance"]] = None,
        mlp2conv_norm_layer_kwargs: Optional[dict[str, Any]] = None,
        latent_norm_layer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self._init_kwargs = {
            k: v
            for k, v in filter(
                lambda k_v: k_v[0] not in {"self", "__class__"}, locals().items()
            )
        }
        super().__init__()
        # Allow kwargs to catch extra init parameters, but warn if a non-None kwargs is
        # being sent over.
        if len(kwargs.keys()) > 0 and any(
            map(lambda v: v is not None, kwargs.values())
        ):
            warnings.warn(f"Network init ignoring extra kwargs {kwargs}")

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.conv2mlp_channels = conv2mlp_channels
        self.out_channels = out_channels
        if activate_fn_kwargs is None:
            self._activate_fn_kwargs = dict()
        else:
            self._activate_fn_kwargs = activate_fn_kwargs
        # Kwargs specific to the activation functions outside of the CARN/INR
        # sub-networks.
        # self._outside_activate_fn_kwargs = {"inplace": True} | self._activate_fn_kwargs
        self._outside_activate_fn_kwargs = {"inplace": False} | self._activate_fn_kwargs
        if mlp2conv_norm_layer_kwargs is None:
            mlp2conv_norm_layer_kwargs = dict()
        if latent_norm_layer_kwargs is None:
            latent_norm_layer_kwargs = dict()
        self._activate_fn_name = mrinr.nn.normalize_lookup_str(activate_fn)
        activate_fn_cls = mrinr.nn.ACTIVATE_FN_CLS_LOOKUP[self._activate_fn_name]
        self.activate_fn = activate_fn_cls(**self._outside_activate_fn_kwargs)
        self.use_input_coords_as_channels = use_input_coords_as_channels
        self.use_quant_reg_latent_space = quant_reg_bins is not None
        self.use_mlp_shortcut = use_mlp_shortcut

        if self.use_input_coords_as_channels:
            assert self.in_channels >= self.COORD_NDIM
        self._inr_model = str(inr_model).lower().strip()
        self._conv_model = (
            str(conv_model)
            .lower()
            .strip()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )
        # Normalize to American spelling.
        if "neighbour" in self._inr_model:
            self._inr_model = "neighborhood"
        assert (
            ("ensemble" in self._inr_model)
            or ("neighbor" in self._inr_model)
            or ("interp" in self._inr_model)
        )

        # Encoder stack
        if "carn" in self._conv_model:
            # Set carn models to use the same activation function as the encoder-decoder, if
            # not set explicitly.
            self.encoder_conv = mrinr.nn.CARNEncoder3d(
                in_channels=self.in_channels,
                out_channels=self.conv2mlp_channels,
                **(
                    {
                        "activate_fn": activate_fn,
                        "activate_fn_kwargs": activate_fn_kwargs,
                    }
                    | conv_kwargs
                ),
            )
        elif self._conv_model == "attentionunet":
            self.encoder_conv = monai.networks.nets.AttentionUnet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.conv2mlp_channels,
                **conv_kwargs,
            )
        elif self._conv_model == "unet":
            self.encoder_conv = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.conv2mlp_channels,
                **conv_kwargs,
            )
        if "ensemble" in self._inr_model:
            self.encoder_inr = mrinr.nn.INR3DEnsembledResidualDecoder(
                in_latent_features=self.encoder_conv.out_channels,
                out_features=self.latent_channels,
                **inr_kwargs,
            )
        elif "neighbor" in self._inr_model:
            self.encoder_inr = mrinr.nn.INR3DNeighborhoodResidualDecoder(
                in_latent_features=self.encoder_conv.out_channels,
                out_features=self.latent_channels,
                **inr_kwargs,
            )
        elif "interp" in self._inr_model:
            self.encoder_inr = InterpolResampler(
                in_channels=self.encoder_conv.out_channels,
                out_channels=self.latent_channels,
                **inr_kwargs,
            )
        if self.use_mlp_shortcut:
            self.encoder_mlp_shortcut_in2out_features = torch.nn.Conv3d(
                self.encoder_inr.latent_features, self.encoder_inr.out_features, 1
            )
            self.encoder_mlp_shortcut_combiner_conv = torch.nn.Conv3d(
                self.encoder_inr.out_features * 2, self.encoder_inr.out_features, 1
            )
        else:
            self.encoder_mlp_shortcut_in2out_features = None
            self.encoder_mlp_shortcut_combiner_conv = None

        self.latent_activate_fn = mrinr.nn.ACTIVATE_FN_CLS_LOOKUP[
            mrinr.nn.normalize_lookup_str(latent_activate_fn)
        ]()

        if self.use_quant_reg_latent_space:
            self.quant_reg = mrinr.nn.QuantizeRegularizationLayer(bins=quant_reg_bins)
        else:
            self.quant_reg = None

        # Decoder stack
        # Decoder MLP/INR
        if "ensemble" in self._inr_model:
            self.decoder_inr = mrinr.nn.INR3DEnsembledResidualDecoder(
                in_latent_features=self.latent_channels,
                out_features=self.conv2mlp_channels,
                **inr_kwargs,
            )
        elif "neighbor" in self._inr_model:
            self.decoder_inr = mrinr.nn.INR3DNeighborhoodResidualDecoder(
                in_latent_features=self.latent_channels,
                out_features=self.conv2mlp_channels,
                **inr_kwargs,
            )
        elif "interp" in self._inr_model:
            self.decoder_inr = InterpolResampler(
                in_channels=self.latent_channels,
                out_channels=self.conv2mlp_channels,
                **inr_kwargs,
            )
        if self.use_mlp_shortcut:
            self.decoder_mlp_shortcut_in2out_features = torch.nn.Conv3d(
                self.decoder_inr.latent_features, self.decoder_inr.out_features, 1
            )
            self.decoder_mlp_shortcut_combiner_conv = torch.nn.Conv3d(
                self.decoder_inr.out_features * 2, self.decoder_inr.out_features, 1
            )
        else:
            self.decoder_mlp_shortcut_in2out_features = None
            self.decoder_mlp_shortcut_combiner_conv = None

        decoder_conv_in_channels = self.decoder_inr.out_features
        # Decoder conv network
        if "carn" in self._conv_model:
            # Set carn models to use the same activation function as the encoder-decoder, if
            # not set explicitly.
            self.decoder_conv = mrinr.nn.CARNEncoder3d(
                in_channels=decoder_conv_in_channels,
                out_channels=self.out_channels,
                **(
                    {
                        "activate_fn": activate_fn,
                        "activate_fn_kwargs": activate_fn_kwargs,
                    }
                    | conv_kwargs
                ),
            )
        elif self._conv_model == "attentionunet":
            self.decoder_conv = monai.networks.nets.AttentionUnet(
                spatial_dims=3,
                in_channels=decoder_conv_in_channels,
                out_channels=self.out_channels,
                **conv_kwargs,
            )
        elif self._conv_model == "unet":
            self.decoder_conv = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=decoder_conv_in_channels,
                out_channels=self.out_channels,
                **conv_kwargs,
            )
        # Initialize norm layers.
        (
            self.encoder_conv2mlp_norm,
            self.latent_norm,
            self.decoder_mlp2conv_norm,
        ) = self._init_norm_layers(
            norm_layer=norm_layer,
            latent_norm_layer_kwargs=latent_norm_layer_kwargs,
            mlp2conv_norm_layer_kwargs=mlp2conv_norm_layer_kwargs,
        )
        # Init layers that transform objects in the latent space.
        if latent_carn_kwargs is not None:
            # Add activation function kwargs to the latent carn layers, if requested.
            latent_carn_kwargs = {
                "activate_fn": activate_fn,
                "activate_fn_kwargs": activate_fn_kwargs,
            } | latent_carn_kwargs
            # Force the in and out channels to match latent space size.
            latent_carn_kwargs = latent_carn_kwargs | {
                "in_channels": self.latent_channels,
                "out_channels": self.latent_channels,
            }
        (
            self.mlp2latent_transform,
            self.latent2mlp_transform,
        ) = self._init_latent_layers(
            latent_conv_kwargs=latent_carn_kwargs,
            activate_fn_cls=activate_fn_cls,
            activate_fn_kwargs=self._activate_fn_kwargs,
        )

        # Initialize the lazy convs.
        x = torch.randn(2, self.encoder_conv.in_channels, 40, 40, 40)
        y = torch.randn(2, self.decoder_conv.in_channels, 40, 40, 40)
        z = torch.randn(2, self.latent_channels, 40, 40, 40)
        with torch.no_grad():
            self.encoder_conv(x)
            self.decoder_conv(y)
            self.mlp2latent_transform(z)
            if self.latent2mlp_transform is not None:
                self.latent2mlp_transform(z)
        del x, y, z

    def _init_norm_layers(
        self, norm_layer, latent_norm_layer_kwargs, mlp2conv_norm_layer_kwargs
    ):
        """Optional model flags/switches.
        Encoder INR --> [normalization] --> latent_activation_fn --> Z
        and
        CARN --> [normalization] --> conv-mlp_activation_fn --> MLP
        and
        MLP --> [normalization] --> conv-mlp_activation_fn --> CARN"""
        if norm_layer is not None:
            if "batch" in norm_layer:
                encoder_conv2mlp_norm = torch.nn.BatchNorm3d(
                    self.encoder_conv.out_channels, **mlp2conv_norm_layer_kwargs
                )
                latent_norm = torch.nn.BatchNorm3d(
                    self.encoder_inr.out_features, **latent_norm_layer_kwargs
                )
                decoder_mlp2conv_norm = torch.nn.BatchNorm3d(
                    self.decoder_inr.out_features, **mlp2conv_norm_layer_kwargs
                )
            elif "group" in norm_layer:
                encoder_conv2mlp_norm = torch.nn.GroupNorm(
                    num_channels=self.encoder_conv.out_channels,
                    **mlp2conv_norm_layer_kwargs,
                )
                latent_norm = torch.nn.GroupNorm(
                    num_channels=self.encoder_inr.out_features,
                    **latent_norm_layer_kwargs,
                )
                decoder_mlp2conv_norm = torch.nn.GroupNorm(
                    num_channels=self.decoder_inr.out_features,
                    **mlp2conv_norm_layer_kwargs,
                )
            elif "instance" in norm_layer:
                encoder_conv2mlp_norm = torch.nn.InstanceNorm3d(
                    num_features=self.encoder_conv.out_channels,
                    **mlp2conv_norm_layer_kwargs,
                )
                latent_norm = torch.nn.InstanceNorm3d(
                    num_features=self.encoder_inr.out_features,
                    **latent_norm_layer_kwargs,
                )
                decoder_mlp2conv_norm = torch.nn.InstanceNorm3d(
                    num_features=self.decoder_inr.out_features,
                    **mlp2conv_norm_layer_kwargs,
                )
        else:
            encoder_conv2mlp_norm = None
            latent_norm = None
            decoder_mlp2conv_norm = None

        return encoder_conv2mlp_norm, latent_norm, decoder_mlp2conv_norm

    def _init_latent_layers(
        self, latent_conv_kwargs, activate_fn_cls, activate_fn_kwargs
    ):
        mlp2latent = collections.OrderedDict()
        if latent_conv_kwargs is not None:
            # If a conv net will be used on the latent space, then an activation
            # function will need to be applied to the MLP output.
            mlp2latent["inr_encoder_activation"] = activate_fn_cls(**activate_fn_kwargs)
            latent_enc_conv = mrinr.nn.CARNEncoder3d(**latent_conv_kwargs)
            mlp2latent["latent_encoder_conv"] = latent_enc_conv
        if self.latent_norm is not None:
            mlp2latent["latent_norm"] = self.latent_norm
            del self.latent_norm
        mlp2latent["latent_activation"] = self.latent_activate_fn
        del self.latent_activate_fn
        if self.quant_reg is not None:
            mlp2latent["quantized_regularization"] = self.quant_reg
            del self.quant_reg
        if latent_conv_kwargs is not None:
            latent2mlp_ = collections.OrderedDict()
            latent_dec_conv = mrinr.nn.CARNEncoder3d(**latent_conv_kwargs)
            latent2mlp_["latent_decoder_conv"] = latent_dec_conv
            # If a conv net will be used on the latent space, then an activation
            # function will need to be applied to the latent conv output.
            latent2mlp_["inr_decoder_activation"] = activate_fn_cls(
                **activate_fn_kwargs
            )
            latent2mlp = torch.nn.Sequential(latent2mlp_)
        else:
            latent2mlp = None
        return torch.nn.Sequential(mlp2latent), latent2mlp

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    @classmethod
    def __carn_total_receptive_field_size(
        cls, carn_net: "mrinr.nn.CARNEncoder3d"
    ) -> tuple:
        """Calculate the total receptive field size for a CARN module.

        This is very specific to the carn encoder class, it certainly will not
        generalize to other conv networks!

        Parameters
        ----------
        carn_net : mrinr.nn.CARNEncoder3d

        Returns
        -------
        tuple
            Receptive field size of each dimension
        """

        rf = np.ones(cls.COORD_NDIM)
        for name, m in carn_net.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.Conv3d,
                    torch.nn.LazyConv3d,
                ),
            ):
                k = np.array(m.kernel_size)
                d = np.array(m.dilation)
                s = np.array(m.stride)
                # Ensure that stride and dilation are 1.
                assert (d == 1).all()
                assert (s == 1).all()
                rf = rf + (k - 1)

        return tuple(rf.round().astype(int).tolist())

    @functools.cached_property
    def _side_padding_for_encoder_conv_receptive_field(self):
        rf = self.__carn_total_receptive_field_size(self.encoder_conv)
        pad_for_rf_side = np.ceil((np.array(rf) - 1) // 2).astype(int)
        # Add a couple of voxels for safety.
        pad_for_rf_side = pad_for_rf_side + 2
        return tuple(pad_for_rf_side.tolist())

    @functools.cached_property
    def _side_padding_for_decoder_conv_receptive_field(self):
        rf = self.__carn_total_receptive_field_size(self.decoder_conv)
        pad_for_rf_side = np.ceil((np.array(rf) - 1) // 2).astype(int)
        # Add a couple of voxels for safety.
        pad_for_rf_side = pad_for_rf_side + 2
        return tuple(pad_for_rf_side.tolist())

    @staticmethod
    @torch.no_grad
    def __equivariant_sliding_window_inference(
        x: torch.Tensor,
        predictor: torch.nn.Module,
        window_size: int,
        num_predictor_channels: int,
        single_side_pad_to_receptive_field: tuple,
        output_device: Optional[str] = None,
    ):
        # Formula for getting the number of windows in each dimension, assuming no offset.
        def _num_sliding_windows(
            splitter_pad_vol_padded_shape, side_crop, roi_size, overlap
        ):
            pad_shape = np.array(splitter_pad_vol_padded_shape)
            num_win = (pad_shape - overlap) / (roi_size - overlap)
            assert (num_win == num_win.round()).all()
            return num_win.round().astype(int)

        if output_device is None:
            output_device = x.device
        input_device = x.device
        roi_size = np.array((window_size,) * 3)
        side_rf = np.array(single_side_pad_to_receptive_field)
        overlap = side_rf * 2
        x_size = np.array(x.shape[-3:])

        # Pad by the receptive field amount on each side.
        x_pad = torch.nn.functional.pad(
            x.to(output_device),
            [side_rf[i_d] for _ in range(2) for i_d in range(len(side_rf) - 1, -1, -1)],
            mode="reflect",
        )

        splitter = monai.inferers.SlidingWindowSplitter(
            roi_size, overlap=overlap, offset=0, pad_mode="reflect"
        )
        sp_size = np.array(splitter.get_padded_shape(x_pad))
        n_patches = _num_sliding_windows(
            sp_size, side_crop=side_rf, roi_size=roi_size, overlap=overlap
        )

        y_size = x_size
        y_padded_size = n_patches * (roi_size - 2 * side_rf)
        y_padded = torch.zeros(
            (x.shape[0], num_predictor_channels) + tuple(y_padded_size.astype(int)),
            dtype=x.dtype,
            device=output_device,
        )

        # Select patch, run forward pass, subsample result patch, and store into y_padded.
        for p, loc in splitter(x_pad):
            ijk_splitter_patch = np.array(loc) // (roi_size - overlap)
            p_sub = predictor(p.to(input_device)).to(output_device)[
                ...,
                side_rf[0] : -side_rf[0],
                side_rf[1] : -side_rf[1],
                side_rf[2] : -side_rf[2],
            ]
            y_p_padded_start_idx = ijk_splitter_patch * (roi_size - 2 * side_rf)
            y_p_padded_end_idx = y_p_padded_start_idx + np.array(p_sub.shape[-3:])
            y_padded[
                ...,
                y_p_padded_start_idx[0] : y_p_padded_end_idx[0],
                y_p_padded_start_idx[1] : y_p_padded_end_idx[1],
                y_p_padded_start_idx[2] : y_p_padded_end_idx[2],
            ] = p_sub.to(output_device)

        # Remove extra end padding from y.
        y = y_padded[..., : y_size[0], : y_size[1], : y_size[2]]

        return y

    @torch.no_grad
    def _low_memory_sliding_window_conv_inference(
        self,
        conv_net: torch.nn.Module,
        x: torch.Tensor,
        out_channels: int,
        receptive_field_side_pad: tuple,
        sliding_window_size: Optional[int],
        allow_sliding_window: bool = True,
    ):
        try:
            # Wrap forward() call to strip out the Tensors from the traceback frame, in
            # the event of an OOM error.
            y = mrinr.utils.gpu_mem_restore(conv_net)(x)
        # From
        # <https://github.com/facebookresearch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L283>
        except RuntimeError as e:
            str_e = str(e).lower().replace(":", "").strip()
            if (
                "out of memory" in str_e
                and (sliding_window_size is not None)
                and allow_sliding_window
            ):
                for p in conv_net.parameters():
                    if p.grad is not None:
                        p.grad = None
                del e
                gc.collect()
                torch.cuda.empty_cache()
                # Store output on cpu, then transfer to gpu when the forward pass has
                # already been performed.
                y = self.__equivariant_sliding_window_inference(
                    x=x,
                    predictor=conv_net,
                    window_size=sliding_window_size,
                    num_predictor_channels=out_channels,
                    single_side_pad_to_receptive_field=receptive_field_side_pad,
                    output_device="cpu",
                )
                y = y.to(x.device)
            elif not allow_sliding_window:
                raise RuntimeError(
                    "Sliding window inference not allowed to resolve out of memory error!"
                ) from e
            else:
                raise e
        return y

    def forward(self, mode: Union[str, torch.Tensor], *args, **kwargs):
        if torch.is_tensor(mode):
            mode = mode.flatten().detach().cpu().int().numpy()
            if (mode == 0).all():
                mode = "encode"
            elif (mode == 1).all():
                mode = "decode"
            else:
                raise ValueError(f"ERROR: Invalid mode value {mode}")
        mode = mode.strip().lower()
        if mode == "encode":
            ret = self.encode(*args, **kwargs)
        elif mode == "decode":
            ret = self.decode(*args, **kwargs)

        return ret

    def encode(
        self,
        x: mrinr.typing.Volume,
        x_coords: mrinr.typing.CoordGrid3D,
        query_z_coords: mrinr.typing.CoordGrid3D,
        affine_vox2coords: mrinr.typing.HomogeneousAffine3D,
        x_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | _NormalizerParams,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | _NormalizerParams,
        x_grid_sizes: Optional[torch.Tensor] = None,
        query_z_grid_sizes: Optional[torch.Tensor] = None,
        query_z_chunks: Optional[int] = None,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ) -> torch.Tensor:
        if self.use_input_coords_as_channels:
            # Concat x and x coordinates.
            x_features = torch.cat(
                [x, einops.rearrange(x_coords, "b x y z coord -> b coord x y z")], dim=1
            )
        else:
            x_features = x

        # If the network is in eval mode and the sliding window size has been indicated,
        # try to perform low-memory inference with the conv network.
        if (
            isinstance(_conv_inference_sliding_window_size, int)
            and _conv_inference_sliding_window_size < 0
        ):
            _conv_inference_sliding_window_size = None
        if (
            (self._conv_model == "carn")
            and (not self.training)
            and (_conv_inference_sliding_window_size is not None)
        ):
            z = self._low_memory_sliding_window_conv_inference(
                conv_net=self.encoder_conv,
                x=x_features,
                out_channels=self.encoder_conv.out_channels,
                receptive_field_side_pad=self._side_padding_for_encoder_conv_receptive_field,
                sliding_window_size=_conv_inference_sliding_window_size,
                allow_sliding_window=True,
            )
        else:
            z = self.encoder_conv(x_features)
        if self.encoder_conv2mlp_norm is not None:
            z = self.encoder_conv2mlp_norm(z)
        z = self.activate_fn(z)
        # Create normalized (in range [0, 1]) coordinates for potential positional
        # encoding in the INR MLPs.
        z_coord_normalizer_params = _NormalizerParams(
            min_shift=z_coord_normalizer_params[0],
            size_scale=z_coord_normalizer_params[1],
        )
        x_coord_normalizer_params = _NormalizerParams(
            min_shift=x_coord_normalizer_params[0],
            size_scale=x_coord_normalizer_params[1],
        )
        if self.use_mlp_shortcut:
            z_interp = self.encoder_mlp_shortcut_in2out_features(z)
            z_interp = mrinr.vols.sample_vol(
                z_interp,
                query_z_coords,
                affine_vox2mm=affine_vox2coords,
                mode=self._MLP_SHORTCUT_INTERP_MODE,
                padding_mode=self._MLP_SHORTCUT_PADDING_MODE,
                align_corners=True,
            )
        z = self.encoder_inr(
            z,
            x_coords=x_coords,
            query_coords=query_z_coords,
            affine_vox2coords=affine_vox2coords,
            x_grid_sizes=x_grid_sizes,
            query_grid_sizes=query_z_grid_sizes,
            x_coord_normalizer_params=x_coord_normalizer_params,
            query_coord_normalizer_params=z_coord_normalizer_params,
            max_q_chunks=query_z_chunks,
        )
        if self.use_mlp_shortcut:
            z = self.encoder_mlp_shortcut_combiner_conv(torch.cat([z, z_interp], dim=1))
            # z = z + z_interp
            # Only use activation if the mlp2latent_transform will be used.
            if self.mlp2latent_transform is not None:
                z = self.activate_fn(z)
        if self.mlp2latent_transform is not None:
            z = self.mlp2latent_transform(z)
        return z

    def decode(
        self,
        z: mrinr.typing.Volume,
        z_coords: mrinr.typing.CoordGrid3D,
        query_xp_coords: mrinr.typing.CoordGrid3D,
        affine_vox2coords: mrinr.typing.HomogeneousAffine3D,
        z_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | _NormalizerParams,
        xp_coord_normalizer_params: tuple[torch.Tensor, torch.Tensor]
        | _NormalizerParams,
        z_grid_sizes: Optional[torch.Tensor] = None,
        xp_grid_sizes: Optional[torch.Tensor] = None,
        query_xp_chunks: Optional[int] = None,
        _conv_inference_sliding_window_size: Optional[int] = None,
    ):
        if self.latent2mlp_transform is not None:
            xp = self.latent2mlp_transform(z)
        else:
            xp = z
        # Sample the encoded space.
        # Create normalized (in range [0, 1]) coordinates for potential positional
        # encoding in the INR MLPs.
        z_coord_normalizer_params = _NormalizerParams(
            min_shift=z_coord_normalizer_params[0],
            size_scale=z_coord_normalizer_params[1],
        )
        xp_coord_normalizer_params = _NormalizerParams(
            min_shift=xp_coord_normalizer_params[0],
            size_scale=xp_coord_normalizer_params[1],
        )
        if self.use_mlp_shortcut:
            xp_interp = self.decoder_mlp_shortcut_in2out_features(xp)
            xp_interp = mrinr.vols.sample_vol(
                xp_interp,
                query_xp_coords,
                affine_vox2mm=affine_vox2coords,
                mode=self._MLP_SHORTCUT_INTERP_MODE,
                padding_mode=self._MLP_SHORTCUT_PADDING_MODE,
                align_corners=True,
            )
        xp = self.decoder_inr(
            xp,
            x_coords=z_coords,
            query_coords=query_xp_coords,
            affine_vox2coords=affine_vox2coords,
            x_grid_sizes=z_grid_sizes,
            query_grid_sizes=xp_grid_sizes,
            x_coord_normalizer_params=z_coord_normalizer_params,
            query_coord_normalizer_params=xp_coord_normalizer_params,
            max_q_chunks=query_xp_chunks,
        )
        if self.use_mlp_shortcut:
            xp = self.decoder_mlp_shortcut_combiner_conv(
                torch.cat([xp, xp_interp], dim=1)
            )
            # xp = xp + xp_interp
            # Use activation function if the mlp2conv_norm is used.
            if self.decoder_mlp2conv_norm is not None:
                xp = self.activate_fn(xp)
        if self.decoder_mlp2conv_norm is not None:
            xp = self.decoder_mlp2conv_norm(xp)
        xp = self.activate_fn(xp)
        if (
            isinstance(_conv_inference_sliding_window_size, int)
            and _conv_inference_sliding_window_size < 0
        ):
            _conv_inference_sliding_window_size = None
        if (
            (self._conv_model == "carn")
            and (not self.training)
            and (_conv_inference_sliding_window_size is not None)
        ):
            # TODO: Test and make sure this is valid; padding values within the network
            # TODO: is much different than padding inputs like in the encoder!
            xp = self._low_memory_sliding_window_conv_inference(
                conv_net=self.decoder_conv,
                x=xp,
                out_channels=self.decoder_conv.out_channels,
                receptive_field_side_pad=self._side_padding_for_decoder_conv_receptive_field,
                sliding_window_size=_conv_inference_sliding_window_size,
                # allow_sliding_window=False,
                allow_sliding_window=True,
            )
        else:
            xp = self.decoder_conv(xp)

        return xp


class ConvResizedCMMCINRAutoencoder(CMMCINREnsembleAutoencoder):
    def __init__(
        self,
        in_channels: int,
        conv2mlp_channels: int,
        latent_channels: int,
        conv_resize_factor: int,
        out_channels: int,
        activate_fn: str,
        use_input_coords_as_channels: bool,
        conv_kwargs: dict[str, Any],
        inr_kwargs: dict[str, Any],
        inr_model: Literal["ensemble", "neighborhood", "interpolate"] = "ensemble",
        conv_model: Literal["carn"] = "carn",
        latent_carn_kwargs: Optional[dict[str, Any]] = None,
        latent_activate_fn: str = "tanh",
        use_mlp_shortcut: bool = False,
        quant_reg_bins: Optional[int] = None,
        activate_fn_kwargs: Optional[dict[str, Any]] = None,
        norm_layer: Optional[Literal["batch", "group", "instance"]] = None,
        mlp2conv_norm_layer_kwargs: Optional[dict[str, Any]] = None,
        latent_norm_layer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self._init_kwargs = {
            k: v
            for k, v in filter(
                lambda k_v: k_v[0] not in {"self", "__class__"}, locals().items()
            )
        }
        super().__init__(
            in_channels=in_channels,
            conv2mlp_channels=conv2mlp_channels,
            latent_channels=latent_channels,
            out_channels=out_channels,
            activate_fn=activate_fn,
            use_input_coords_as_channels=use_input_coords_as_channels,
            conv_kwargs=conv_kwargs,
            inr_kwargs=inr_kwargs,
            inr_model=inr_model,
            conv_model=conv_model,
            latent_carn_kwargs=latent_carn_kwargs,
            latent_activate_fn=latent_activate_fn,
            use_mlp_shortcut=use_mlp_shortcut,
            quant_reg_bins=quant_reg_bins,
            activate_fn_kwargs=activate_fn_kwargs,
            norm_layer=norm_layer,
            mlp2conv_norm_layer_kwargs=mlp2conv_norm_layer_kwargs,
            latent_norm_layer_kwargs=latent_norm_layer_kwargs,
            **kwargs,
        )
        # Allow kwargs to catch extra init parameters, but warn if a non-None kwargs is
        # being sent over.
        if len(kwargs.keys()) > 0 and any(
            map(lambda v: v is not None, kwargs.values())
        ):
            warnings.warn(f"Network init ignoring extra kwargs {kwargs}")
