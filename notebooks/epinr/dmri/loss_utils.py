import math
from typing import Optional

import torch
import einops
import kornia
import numpy as np

import mrinr


class WeightedNMIParzenLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        num_bins: int = 32,
        sigma_ratio: float = 0.5,
        reduction: str = "mean",
        eps: float = 1e-7,
        norm_mi: bool = True,
        norm_images: bool = True,
    ):
        super().__init__(reduction=reduction)
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        self.spatial_dims = 3
        self.num_bins = int(num_bins)
        # shape (num_bins)
        bin_centers = torch.linspace(0.0, 1.0, self.num_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_centers: torch.Tensor

        sigma = torch.mean(self.bin_centers[1:] - self.bin_centers[:-1]) * sigma_ratio
        self.register_buffer("sigma", sigma)
        self.sigma: torch.Tensor
        self.eps = eps
        self.norm_mi = norm_mi
        self.norm_images = norm_images

    @staticmethod
    def spatial_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Min-max normalize x to [0, 1] along spatial dimensions."""
        x_min = einops.reduce(x, "b c x y z -> b c 1 1 1", "min")
        x_max = einops.reduce(x, "b c x y z -> b c 1 1 1", "max")
        x_normalized = (x - x_min) / (x_max - x_min + eps)
        return x_normalized

    def parzen_windowing_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """Parzen Gaussian weighting function to approximate histogram differentiably.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape B x C x X x Y x Z.

            *NOTE* Input volume must have intensities normalized to [0, 1]. Intensities
            will be clamped to [0, 1] internally.
        Returns
        -------
        torch.Tensor
            Discrete probability distributions for each voxel, shape
            (B*C) x (X*Y*Z) x num_bins. Distributions are not averaged over spatial
            dimensions.
        """
        y = torch.clamp(x, 0.0, 1.0)
        # Move independent dims to the front, and merge spatial dims into a sampling
        # dimension, plus a singleton dimension for broadcasting hist. bins.
        y = einops.rearrange(y, "b c ... -> (b c) (...) 1")
        w_parzen = (1 / (self.sigma * math.sqrt(2 * math.pi))) * torch.exp(
            -0.5 * ((y - self.bin_centers[None, None, :]) / self.sigma) ** 2
        )
        # Normalize over bins.
        p = w_parzen / torch.maximum(
            torch.sum(w_parzen, dim=-1, keepdim=True),
            w_parzen.new_tensor([self.eps]),
        )
        # Wait to average over sampling dimensions until estimating the joint histogram.
        return p

    def weighted_nmi(
        self,
        pred: mrinr.typing.ScalarVolume,
        target: mrinr.typing.ScalarVolume,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
        norm_mi: bool = True,
        normalize_images: bool = True,
    ) -> torch.Tensor:
        """Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
            weight_mask: the shape should be B[1DHW] or B[NDHW], optional.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(
                f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})"
            )

        if normalize_images:
            # Normalize pred and target to [0, 1] along spatial dims, keeping batch and
            # channel dims independent.
            x = self.spatial_normalize(pred, eps=self.eps)
            y = self.spatial_normalize(target, eps=self.eps)
        else:
            x = pred
            y = target

        # Parzen windowing, without averaging over samples. Outputs are
        # shape (B*C)(D*H*W)(num_bins).
        p_pred = self.parzen_windowing_gaussian(x)
        p_target = self.parzen_windowing_gaussian(y)

        # Estimate joint histogram P_pred,target weighted by the weight mask if
        # provided.
        if weight_mask is not None:
            w = einops.rearrange(
                weight_mask.expand_as(pred), "b c x y z -> (b c) (x y z) 1 1"
            )
            # Normalize the weight mask to be in the range [0, 1].
            w = w / torch.maximum(
                w.max(dim=1, keepdim=True).values, w.new_tensor([self.eps])
            )
        else:
            w = 1.0
        p_joint = (
            w * einops.einsum(p_pred, p_target, "bc xyz i, bc xyz j -> bc xyz i j")
        ).sum(1)
        # Normalize joint histogram.
        p_joint = p_joint / torch.maximum(
            p_joint.sum(dim=(-2, -1), keepdim=True),
            p_joint.new_tensor([self.eps]),
        )
        # Estimate marginal histograms.
        p_pred_marginal = p_joint.sum(dim=-1)
        p_target_marginal = p_joint.sum(dim=-2)

        # Compute entropies.
        H_pred = -torch.sum(
            p_pred_marginal
            * torch.log(
                torch.maximum(p_pred_marginal, p_pred_marginal.new_tensor([self.eps]))
            ),
            dim=-1,
        )
        H_target = -torch.sum(
            p_target_marginal
            * torch.log(
                torch.maximum(
                    p_target_marginal, p_target_marginal.new_tensor([self.eps])
                )
            ),
            dim=-1,
        )
        H_joint = -torch.sum(
            p_joint * torch.log(torch.maximum(p_joint, p_joint.new_tensor([self.eps]))),
            dim=(-2, -1),
        )

        # NMI or plain MI.
        if norm_mi:
            mi = (H_pred + H_target) / torch.maximum(
                H_joint, H_joint.new_tensor([self.eps])
            )
        else:
            mi = H_pred + H_target - H_joint

        if self.reduction == "sum":
            # sum over the batch and channel ndims
            r = torch.sum(mi)
        elif self.reduction == "none":
            # No reduction of independent dims.
            r = einops.rearrange(mi, "(b c) -> b c", b=pred.shape[0], c=pred.shape[1])
        elif self.reduction == "mean":
            # average over the batch and channel ndims
            r = torch.mean(mi)
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction}, "
                'available options are ["mean", "sum", "none"].'
            )
        return r

    def forward(
        self,
        pred: mrinr.typing.ScalarVolume,
        target: mrinr.typing.ScalarVolume,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
    ) -> torch.Tensor:
        # Loss is negative NMI.
        return -self.weighted_nmi(
            pred,
            target,
            weight_mask=weight_mask,
            norm_mi=self.norm_mi,
            normalize_images=self.norm_images,
        )


class NCC(torch.nn.modules.loss._Loss):
    # Normalized Cross Correlation
    # Taken from <https://github.com/MIAGroupUT/IDIR/blob/main/objectives/ncc.py>,
    # which itself was taken from <https://github.com/BDdeVos/TorchIR>
    class _StableStd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            assert tensor.numel() > 1
            ctx.tensor = tensor.detach()
            res = torch.std(tensor).detach()
            ctx.result = res.detach()
            return res

        @staticmethod
        def backward(ctx, grad_output):
            tensor = ctx.tensor.detach()
            result = ctx.result.detach()
            e = 1e-6
            assert tensor.numel() > 1
            return (
                (2.0 / (tensor.numel() - 1.0))
                * (grad_output.detach() / (result.detach() * 2 + e))
                * (tensor.detach() - tensor.mean().detach())
            )

    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.forward = self.metric

    def ncc(self, x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        stablestd = self._StableStd.apply
        std = stablestd(x1) * stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    def metric(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return -self.ncc(fixed, warped)


class DoGLaplacian(torch.nn.Module):
    """Difference of Gaussians Laplacian filter."""

    def __init__(
        self,
        sigma_low: float | tuple,
        truncate: float | tuple,
        border_type: str = "replicate",
        normalized: bool = False,
    ) -> None:
        super().__init__()

        self.border_type = border_type
        self.normalized = normalized

        if isinstance(sigma_low, (float, int)):
            sigma_low = [sigma_low, sigma_low, sigma_low]
        sigma_low = np.asarray(sigma_low, dtype=np.float32)
        # Scale by 1.6 to get the high sigma that approximates the Laplacian.
        sigma_high = sigma_low * 1.6
        if isinstance(truncate, (float, int)):
            truncate = [truncate, truncate, truncate]
        truncate = np.asarray(truncate, dtype=np.float32)
        kernel_size_low = 2 * np.ceil(truncate * sigma_low).astype(int) + 1
        kernel_size_high = 2 * np.ceil(truncate * sigma_high).astype(int) + 1

        self.sigma_low = tuple(sigma_low.tolist())
        self.sigma_high = tuple(sigma_high.tolist())
        self.kernel_size_low = tuple(kernel_size_low.tolist())
        self.kernel_size_high = tuple(kernel_size_high.tolist())

        gaussian_kernel_low = kornia.filters.get_gaussian_kernel3d(
            kernel_size=kernel_size_low, sigma=self.sigma_low, dtype=torch.float32
        )
        self.register_buffer("gaussian_kernel_low", gaussian_kernel_low)
        self.gaussian_kernel_low: torch.Tensor
        gaussian_kernel_high = kornia.filters.get_gaussian_kernel3d(
            kernel_size=kernel_size_high, sigma=self.sigma_high, dtype=torch.float32
        )
        self.register_buffer("gaussian_kernel_high", gaussian_kernel_high)
        self.gaussian_kernel_high: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DoG Laplacian filter to input tensor.

        Args:
            x: Input tensor of shape (B, C, X, Y, Z).

        Returns:
            Tensor of same shape as input, after applying DoG Laplacian filter.
        """
        low_pass = kornia.filters.filter3d(
            x,
            self.gaussian_kernel_low,
            border_type=self.border_type,
            normalized=self.normalized,
        )
        high_pass = kornia.filters.filter3d(
            x,
            self.gaussian_kernel_high,
            border_type=self.border_type,
            normalized=self.normalized,
        )
        dog_laplacian = low_pass - high_pass
        return dog_laplacian


#! Not yet implemented
class WeightedNCCLoss(torch.nn.modules.loss._Loss):
    def __init__(self, eps: float = 1e-8):
        """Compute weighted normalized cross-correlation (NCC) between two 3D images.
        Args:
            img1, img2: tensors of shape (H, W, D)
            weight: optional tensor of same shape (H, W, D), spatial weights in [0, 1].
                    If None, uniform weights are used.
            eps: small constant for numerical stability.

        Returns:
            scalar tensor: weighted NCC value
        """
        raise NotImplementedError("Weighted NCC loss is not yet implemented.")
        super().__init__(reduction="mean")
        self.eps = eps

    def weighted_ncc(
        self,
        pred: mrinr.typing.ScalarVolume,
        target: mrinr.typing.ScalarVolume,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
    ):
        # assert img1.shape == img2.shape, "Input images must have the same shape"

        # assert x1.shape == x2.shape, "Inputs are not of similar shape"
        # cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        # stablestd = self._StableStd.apply
        # std = stablestd(x1) * stablestd(x2)
        # ncc = cc / (std + e)
        # return ncc

        x = einops.rearrange(pred, "b c x y z -> b c (x y z)")
        y = einops.rearrange(target, "b c x y z -> b c (x y z)")
        if weight_mask is not None:
            w = einops.rearrange(
                weight_mask.expand_as(pred), "b c x y z -> b c (x y z)"
            )
        else:
            w = torch.ones_like(x)

        if weight is None:
            w = torch.ones_like(i1)
        else:
            w = weight.flatten()
            w = w / (w.sum() + self.eps)  # normalize weights

        # Weighted means
        mu1 = torch.sum(w * i1)
        mu2 = torch.sum(w * i2)

        # Weighted covariance and variances
        v1 = i1 - mu1
        v2 = i2 - mu2

        cov12 = torch.sum(w * v1 * v2)
        var1 = torch.sum(w * v1 * v1)
        var2 = torch.sum(w * v2 * v2)

        # Normalized cross-correlation
        ncc = cov12 / (torch.sqrt(var1 * var2) + self.eps)
        return ncc

    def forward(
        self,
        pred: mrinr.typing.ScalarVolume,
        target: mrinr.typing.ScalarVolume,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
    ) -> torch.Tensor:
        # Loss is negative NCC.
        return -self.weighted_ncc(pred, target, weight_mask=weight_mask)


class CoordSampledWeightedMINDLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        patch_size: int = 3,
        patch_comparison_distance: int = 2,
        reduction: str = "mean",
        eps: float = 1e-7,
        **patch_sampling_kwargs,
    ):
        raise NotImplementedError("Weighted MIND loss is not yet implemented.")
        super().__init__(reduction=reduction)
        self.spatial_dims = 3
        self.eps = eps
        self._patch_sampling_kwargs = patch_sampling_kwargs
        if not self._patch_sampling_kwargs:
            self._patch_sampling_kwargs = dict(
                mode_or_interpolation="linear",
                padding_mode_or_bound="zeros",
                interp_lib="torch",
            )

    @staticmethod
    def spatial_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Min-max normalize x to [0, 1] along spatial dimensions."""
        x_min = einops.reduce(x, "b c x y z -> b c 1 1 1", "min")
        x_max = einops.reduce(x, "b c x y z -> b c 1 1 1", "max")
        x_normalized = (x - x_min) / (x_max - x_min + eps)
        return x_normalized

    def forward(
        self,
        moving_coords: mrinr.typing.CoordGrid3D,
        fixed_coords: mrinr.typing.CoordGrid3D,
        moving_volume: mrinr.typing.SingleScalarVolume,
        fixed_volume: mrinr.typing.SingleScalarVolume,
        moving_affine: mrinr.typing.SingleHomogeneousAffine3D,
        fixed_affine: mrinr.typing.SingleHomogeneousAffine3D,
        weight_mask: Optional[mrinr.typing.ScalarVolume] = None,
    ) -> torch.Tensor:
        if moving_volume.ndim == 5:
            moving_volume = moving_volume.squeeze(0)
        if fixed_volume.ndim == 5:
            fixed_volume = fixed_volume.squeeze(0)
        if moving_affine.ndim == 3:
            moving_affine = moving_affine.squeeze(0)
        if fixed_affine.ndim == 3:
            fixed_affine = fixed_affine.squeeze(0)
