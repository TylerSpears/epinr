# -*- coding: utf-8 -*-
# Module for classes and functions that relate to generative models (GANs, DDPM, etc.).
import torch
import torch.nn.functional as F

__all__ = [
    "CondInstanceNorm3d",
    "LPTNDiscriminatorBlock",
]


class CondInstanceNorm3d(torch.nn.Module):
    def __init__(self, latent_size: int, num_features: int, **kwargs_inst_norm):
        super().__init__()
        self._latent_size = latent_size
        self.instance_norm = torch.nn.InstanceNorm3d(num_features, **kwargs_inst_norm)
        self.activate_fn = torch.nn.Tanh()
        self.shift_conv = torch.nn.Conv3d(
            self._latent_size, num_features, kernel_size=1, bias=True
        )
        self.scale_conv = torch.nn.Conv3d(
            self._latent_size, num_features, kernel_size=1, bias=True
        )

    def forward(self, x, z_noise):
        batch_size = x.shape[0]
        z = z_noise.view(batch_size, self._latent_size, 1, 1, 1)
        shift = self.shift_conv(z)
        shift = self.activate_fn(shift)

        scale = self.scale_conv(z)
        scale = self.activate_fn(scale)

        y_norm = self.instance_norm(x)
        y = y_norm * scale + shift

        return y


class LPTNDiscriminatorBlock(torch.nn.Module):
    def __init__(
        self, num_input_channels: int, num_output_channels: int, normalize: bool = True
    ):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            num_input_channels, num_output_channels, kernel_size=3, stride=2, padding=1
        )

        if normalize:
            self.normalizer = torch.nn.InstanceNorm3d(
                num_output_channels, eps=1e-7, affine=True
            )
        else:
            self.normalizer = None

    def forward(self, x):
        y = self.conv(x)
        if self.normalizer:
            # Normalization requires spatial inputs greater than size 1, which can
            # happen when several discriminator blocks are performing subsequent
            # downsampling (stride > 1).
            if torch.prod(torch.as_tensor(y.shape[-3:])) > 1:
                y = self.normalizer(y)
        y = F.leaky_relu(y, negative_slope=0.2)

        return y
