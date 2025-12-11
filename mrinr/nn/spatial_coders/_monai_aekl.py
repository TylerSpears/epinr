# -*- coding: utf-8 -*-
# Modules defining the KL encoder, decoder, and autoencoder classes from monai.
import itertools
from typing import Any, Literal, Optional, Sequence, Union

import einops
import monai
import monai.networks.blocks
import monai.networks.nets
import monai.networks.nets.autoencoderkl as maekl
import torch

import mrinr

__all__ = ["MonaiCoordConvKLEncoder", "MonaiCoordConvKLDecoder"]


class MonaiCoordConvKLEncoder(maekl.Encoder):
    """Taken from Monai's `monai.networks.nets.autoencoderkl.Encoder` class, modified to
    specify coordinate transformations given by conv/downsampling layers, and allow for
    multiple factors of downsampling.

    Original docstring:

    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    _is_coord_aware: bool = True

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        channels: Sequence[int],
        norm_num_groups: int,
        attention_levels: Sequence[bool],
        downsamples: bool | Sequence[bool] = True,
        norm_eps: float = 1e-6,
        with_nonlocal_attn: bool = True,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ):
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        torch.nn.Module.__init__(self)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        blocks: list[torch.nn.Module] = []
        # Initial convolution
        blocks.append(
            monai.networks.blocks.Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        if isinstance(downsamples, bool):
            dwn_samples = list(itertools.repeat(downsamples, len(channels) - 1))
        else:
            dwn_samples = downsamples
        if len(dwn_samples) != len(channels) - 1:
            raise ValueError(
                f"Downsamples must be indicated for N - 1 ({len(channels) - 1}) layers, "
                + f"but got {len(dwn_samples)} values."
            )
        dwn_samples = list(dwn_samples) + [False]
        self._resize_coord_grid = False
        self._downscale_factor = 1
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    maekl.AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        monai.networks.blocks.SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                if dwn_samples[i]:
                    blocks.append(
                        maekl.AEKLDownsample(
                            spatial_dims=spatial_dims, in_channels=input_channel
                        )
                    )
                    self._resize_coord_grid = True
                    self._downscale_factor *= 2
                # If no downsampling is indicated, add a regular conv block with stride
                # 1.
                else:
                    blocks.append(
                        torch.nn.Sequential(
                            monai.networks.blocks.Convolution(
                                spatial_dims=spatial_dims,
                                in_channels=input_channel,
                                out_channels=input_channel,
                                strides=1,
                                kernel_size=3,
                                padding=1,
                                conv_only=True,
                            ),
                        )
                    )
        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                maekl.AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )

            blocks.append(
                monai.networks.blocks.SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                maekl.AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )
        # Normalise and convert to latent size
        blocks.append(
            torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=channels[-1],
                eps=norm_eps,
                affine=True,
            )
        )
        blocks.append(
            monai.networks.blocks.Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = torch.nn.ModuleList(blocks)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(
        self,
        x: torch.Tensor,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
    ) -> Union[torch.Tensor, "mrinr.nn.spatial_coders.DenseCoordSpace"]:
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )
        y = x
        for block in self.blocks:
            y = block(y)

        if not return_coord_space:
            r = y
        else:
            in_spatial_shape = tuple(x.shape)[-self.spatial_dims :]
            out_spatial_shape = tuple(y.shape)[-self.spatial_dims :]
            if self._resize_coord_grid and (in_spatial_shape != out_spatial_shape):
                affine_y_el2coords, y_coords = mrinr.coords.resize_affine(
                    affine_x_el2coords=affine_x_el2coords,
                    in_spatial_shape=in_spatial_shape,
                    target_spatial_shape=out_spatial_shape,
                    centered=True,
                    return_coord_grid=True,
                )
            else:
                y_coords = x_coords
                affine_y_el2coords = affine_x_el2coords
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=y_coords, affine=affine_y_el2coords
            )

        return r


class MonaiCoordConvKLDecoder(maekl.Decoder):
    """Taken from Monai's `monai.networks.nets.autoencoderkl.Decoder` class, modified to
    specify coordinate transformations given by conv/upsampling layers, and allow for
    multiple factors of upsampling.

    Original docstring:

    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    _is_coord_aware: bool = True

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        channels: Sequence[int],
        norm_num_groups: int,
        attention_levels: Sequence[bool],
        upsamples: bool | Sequence[bool] = True,
        norm_eps: float = 1e-6,
        with_nonlocal_attn: bool = True,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
    ) -> None:
        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        torch.nn.Module.__init__(self)
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        # reversed_block_out_channels = list(reversed(channels))
        block_out_channels = list(reversed(channels))

        blocks: list[torch.nn.Module] = []

        # Initial convolution
        blocks.append(
            monai.networks.blocks.Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                maekl.AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=block_out_channels[0],
                )
            )
            blocks.append(
                monai.networks.blocks.SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                maekl.AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=block_out_channels[0],
                )
            )

        # reversed_attention_levels = list(reversed(attention_levels))
        # reversed_num_res_blocks = list(reversed(num_res_blocks))
        attention_levels = list(attention_levels)
        num_res_blocks = list(num_res_blocks)
        block_out_ch = block_out_channels[0]
        if isinstance(upsamples, bool):
            up_samples = list(itertools.repeat(upsamples, len(channels) - 1))
        else:
            up_samples = upsamples
        if len(up_samples) != len(channels) - 1:
            raise ValueError(
                f"Upsamples must be indicated for N - 1 ({len(channels) - 1}) layers, "
                + f"but got {len(up_samples)} values."
            )
        up_samples = list(up_samples) + [False]
        # reversed_up_samples = list(reversed(up_samples))
        self._resize_coord_grid = False
        self._upscale_factor = 1
        for i in range(len(block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = block_out_channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(num_res_blocks[i]):
                blocks.append(
                    maekl.AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if attention_levels[i]:
                    blocks.append(
                        monai.networks.blocks.SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                if up_samples[i]:
                    self._resize_coord_grid = True
                    self._upscale_factor *= 2
                    if use_convtranspose:
                        blocks.append(
                            monai.networks.blocks.Upsample(
                                spatial_dims=spatial_dims,
                                mode="deconv",
                                in_channels=block_in_ch,
                                out_channels=block_in_ch,
                            )
                        )
                    else:
                        post_conv = monai.networks.blocks.Convolution(
                            spatial_dims=spatial_dims,
                            in_channels=block_in_ch,
                            out_channels=block_in_ch,
                            strides=1,
                            kernel_size=3,
                            padding=1,
                            conv_only=True,
                        )
                        blocks.append(
                            monai.networks.blocks.Upsample(
                                spatial_dims=spatial_dims,
                                mode="nontrainable",
                                in_channels=block_in_ch,
                                out_channels=block_in_ch,
                                interp_mode="nearest",
                                scale_factor=2.0,
                                post_conv=post_conv,
                                align_corners=None,
                            )
                        )
                # If no upsampling is requested, add a regular conv block with stride 1.
                else:
                    c = monai.networks.blocks.Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        out_channels=block_in_ch,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                    )

        blocks.append(
            torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_in_ch,
                eps=norm_eps,
                affine=True,
            )
        )
        blocks.append(
            monai.networks.blocks.Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = torch.nn.ModuleList(blocks)

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def forward(
        self,
        x: torch.Tensor,
        x_coords: Optional[torch.Tensor] = None,
        affine_x_el2coords: Optional[torch.Tensor] = None,
        *,
        return_coord_space: bool = False,
    ) -> Union[torch.Tensor, "mrinr.nn.spatial_coders.DenseCoordSpace"]:
        if return_coord_space and (x_coords is None or affine_x_el2coords is None):
            raise ValueError(
                "If `return_coord_space` is True, `x_coords` and `affine_x_el2coords` "
                + "must be provided."
            )
        y = x
        for block in self.blocks:
            y = block(y)

        if not return_coord_space:
            r = y
        else:
            in_spatial_shape = tuple(x.shape)[-self.spatial_dims :]
            out_spatial_shape = tuple(y.shape)[-self.spatial_dims :]
            if self._resize_coord_grid and (in_spatial_shape != out_spatial_shape):
                affine_y_el2coords, y_coords = mrinr.coords.resize_affine(
                    affine_x_el2coords=affine_x_el2coords,
                    in_spatial_shape=in_spatial_shape,
                    target_spatial_shape=out_spatial_shape,
                    centered=True,
                    return_coord_grid=True,
                )
            else:
                y_coords = x_coords
                affine_y_el2coords = affine_x_el2coords
            r = mrinr.nn.spatial_coders.DenseCoordSpace(
                values=y, coords=y_coords, affine=affine_y_el2coords
            )

        return r
