# -*- coding: utf-8 -*-
# Pytorch loss/objective functions.
import collections
import inspect
import itertools
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import monai
import monai.data.utils
import monai.losses
import monai.networks
import monai.networks.nets
import monai.transforms
import numpy as np
import pyrsistent
import pytorch_msssim
import torch

from ._utils import normalize_lookup_str


def jenson_shannon_diverge(
    log_input_dist: torch.Tensor,
    log_target_dist: torch.Tensor,
    reduction=None,
):
    kl = torch.nn.KLDivLoss(reduction=reduction, log_target=True)
    log_P = log_target_dist
    log_Q = log_input_dist

    log_M = (
        log_P
        + torch.log1p(torch.exp(log_Q - log_P))
        - torch.log(torch.tensor([0.5]).to(log_Q))
    )

    jsd = (
        torch.exp(
            torch.log(kl(log_M, log_P))
            + torch.log1p(torch.exp(kl(log_M, log_Q) - kl(log_M, log_P)))
        )
        / 2
    )

    return jsd


class KLDivergenceIdGaussianLoss(torch.nn.Module):
    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        b = mu.shape[0]

        # Compute KL divergence between input and standard independent multivariate
        # Gaussian, simplified to the diagonal covariance case.
        pointwise_kl = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar))

        # Sum over latent dimensions, then average over batch. Equal to the "batchmean"
        # in the pytorch kl divergence loss.
        return pointwise_kl.sum() / b


class MaskedMSELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction is None:
            reduction = "none"
        self._reduction = str(reduction).strip().lower()
        assert self._reduction in {"none", "mean", "sum"}
        self._mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        se = self._mse_loss(input, target)
        if self._reduction == "none":
            l = se * mask if mask is not None else se
        else:
            if mask is not None:
                se.masked_fill_(~mask, torch.nan)
            if self._reduction == "mean":
                l = torch.nanmean(se)
            elif self._reduction == "sum":
                l = torch.nansum(se)
        return l


class MaskedSmoothL1Loss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", beta: float = 1):
        super().__init__()
        if reduction is None:
            reduction = "none"
        self._reduction = str(reduction).strip().lower()
        assert self._reduction in {"none", "mean", "sum"}
        self._smooth_l1_loss = torch.nn.SmoothL1Loss(reduction="none", beta=beta)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        smooth_ae = self._smooth_l1_loss(input, target)
        if self._reduction == "none":
            l = smooth_ae * mask if mask is not None else smooth_ae
        else:
            if mask is not None:
                smooth_ae.masked_fill_(~mask, torch.nan)
            if self._reduction == "mean":
                l = torch.nanmean(smooth_ae)
            elif self._reduction == "sum":
                l = torch.nansum(smooth_ae)
        return l


class MaskedL1Loss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction is None:
            reduction = "none"
        self._reduction = str(reduction).strip().lower()
        assert self._reduction in {"none", "mean", "sum"}
        self._l1_loss = torch.nn.L1Loss(reduction="none")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        ae = self._l1_loss(input, target)
        if self._reduction == "none":
            l = ae * mask if mask is not None else ae
        else:
            if mask is not None:
                ae.masked_fill_(~mask, torch.nan)
            if self._reduction == "mean":
                l = torch.nanmean(ae)
            elif self._reduction == "sum":
                l = torch.nansum(ae)
        return l


class SSIMLoss(pytorch_msssim.SSIM):
    def forward(self, X, Y):
        return 1.0 - super().forward(X, Y)


class MS_SSIMLoss(pytorch_msssim.MS_SSIM):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        allow_padding: bool = False,
        **pad_kwargs,
    ):
        """Multi-Scale Structural-Similarity Index Metric Loss

        Original class docs tring
        --------------------------------------------------------------------------------
        class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images.
                (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will
                be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2).
                Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        --------------------------------------------------------------------------------

        Additional Parameters
        ----------
        allow_padding : bool, optional
            Allow center-padding inputs for valid ms-ssim, by default False
        **pad_kwargs
            Additional kwargs to pass to `torch.nn.functional.pad()`"""

        super().__init__(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=channel,
            spatial_dims=spatial_dims,
            weights=weights,
            K=K,
        )

        self._allow_padding = allow_padding
        self._spatial_dims = spatial_dims
        self._pad_kwargs = pad_kwargs
        if self._allow_padding:
            self._dim_min = (self.win_size - 1) * 2**4

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if self._allow_padding:
            spatial_shape = np.array(tuple(X.shape[-self._spatial_dims :]))
            if (spatial_shape <= self._dim_min).any():
                # Need to pad some dimensions.
                # Add 1 to dim_min to make the output shape strictly > the min.
                lower_pad = (
                    np.clip(
                        np.ceil(((self._dim_min + 1) - spatial_shape) / 2),
                        0,
                        np.inf,
                    )
                    .astype(int)
                    .tolist()
                )
                upper_pad = (
                    np.clip(
                        np.floor(((self._dim_min + 1) - spatial_shape) / 2),
                        0,
                        np.inf,
                    )
                    .astype(int)
                    .tolist()
                )
                # Pytorch pads dimensions in right->left order, so flip each individual
                # spatial index.
                reverse_lower_pad = list(reversed(lower_pad))
                reverse_upper_pad = list(reversed(upper_pad))
                padding = list()
                for i_dim in range(len(spatial_shape)):
                    for side_pad in (
                        reverse_lower_pad[i_dim],
                        reverse_upper_pad[i_dim],
                    ):
                        padding.append(side_pad)
                X = torch.nn.functional.pad(X, pad=padding, **self._pad_kwargs)
                Y = torch.nn.functional.pad(Y, pad=padding, **self._pad_kwargs)

        return 1.0 - super().forward(X, Y)


class MonaiPerceptualLoss(monai.losses.PerceptualLoss):
    def __init__(
        self, *args, norm_scale: float = 1.0, norm_bias: float = 0.0, **kwargs
    ):
        """Input and target Tensors may be optionally normalized if the data is not in
        the desired range according to the monai modules.

        For radimagenet, the input should be scaled between 0 and 1 as monai will
        rescale the input and target data. See
        <https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/losses/perceptual.py#L308>.
        This is different than the original implementation in
        <https://github.com/richzhang/PerceptualSimilarity?tab=readme-ov-file#aii-python-code>,
        where data were scaled to the range [-1, 1].

        For medicalnet, no input range is specified, but the Tensors will be
        standardized (mean 0, std 1) according to
        <https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/losses/perceptual.py#L279>.
        The original paper truncated data to the percentile range [0.5, 99.5] and
        scaled to a standard normal mean 0 and variance 1; from
        <https://arxiv.org/pdf/1904.00625> equation 2.

        NOTE: Scaling occurs before adding bias."""

        super().__init__(*args, **kwargs)

        self._scale = norm_scale
        self._bias = norm_bias

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(
            input=(input * self._scale) + self._bias,
            target=(target * self._scale) + self._bias,
        )


class MonaiPatchAdversarialHingeLoss(torch.nn.Module):
    _WEIGHT_NORM_CLASSES = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose3d,
    )

    PATCH_PADDING_MODE = "reflect"
    MAX_PAD_ITERS = 4

    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        patch_size: tuple[int, ...],
        layer_out_channels: tuple[int, ...] = (8, 16, 32, 64, 1),
        layer_strides: tuple[int, ...] = (1, 2, 2, 2, 1),
        kernel_size: int | tuple[int, ...] = 3,
        num_res_units: int = 2,
        dropout: float = 0.25,
        reduction: str = "mean",
        use_spectral_norm: bool = False,
        **discriminator_kwargs,
    ):
        import mrinr

        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.patch_size = patch_size
        assert len(layer_out_channels) == len(layer_strides), "Layer params mismatch!"
        assert layer_out_channels[-1] == 1, "Last layer must output a single channel!"

        self._adv_loss_fn = monai.losses.PatchAdversarialLoss(
            reduction=reduction,
            criterion=monai.losses.adversarial_loss.AdversarialCriterions.HINGE,
        )
        self.adversarial_net = monai.networks.nets.Discriminator(
            **(
                discriminator_kwargs
                | dict(
                    in_shape=(self.in_channels,) + tuple(self.patch_size),
                    channels=layer_out_channels,
                    kernel_size=kernel_size,
                    strides=layer_strides,
                    num_res_units=num_res_units,
                    dropout=dropout,
                    last_act=None,
                )
            )
        )

        if use_spectral_norm:
            for m in self.adversarial_net.modules():
                if isinstance(m, self._WEIGHT_NORM_CLASSES):
                    torch.nn.utils.parametrizations.spectral_norm(m, "weight")

        # Set all parameter grads to 0
        self.zero_grad(set_to_none=False)
        # Only used to
        self._calc_padder = monai.transforms.SpatialPad(
            spatial_size=self.patch_size,
            method="symmetric",
        )

    def get_extra_state(self) -> Any:
        return {"init_kwargs": self._init_kwargs}

    def set_extra_state(self, state):
        return

    def _crop_or_pad_patch(
        self, *patches: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # If the input is not the given patch size, randomly crop a patch from the
        # image.
        # Assume all patches are the same shape.
        patch = patches[0]
        if tuple(patch.shape[2:]) != self.patch_size:
            random_slice = monai.data.utils.get_random_patch(
                tuple(patch.shape[2:]), patch_size=self.patch_size, rand_state=None
            )
            patches = [p[..., *random_slice] for p in patches]
            patch = patches[0]

        # If cropping did not correct all dimensions, then pad what remains.
        if tuple(patch.shape[2:]) != self.patch_size:
            # Pad until the patch is the correct size.
            for _ in range(self.MAX_PAD_ITERS):
                # Calculate padding for each dimension, drop the channel dim.
                lr_padding = self._calc_padder.compute_pad_width(
                    tuple(patch.shape[2:])
                )[1:]
                if self.PATCH_PADDING_MODE == "reflect":
                    # Clip the padding amount to the maximum allowed by pytorch, which is
                    # the size of the input tensor.
                    lr_padding = np.clip(
                        np.array(lr_padding),
                        a_min=0,
                        a_max=np.array(patch.shape[2:]).reshape(-1, 1) - 1,
                    )
                    lr_padding = tuple(lr_padding.tolist())
                # Reverse the padding for pytorch F.pad, and flatten the tuple of tuples
                # to a single tuple.
                rl_padding = tuple(itertools.chain.from_iterable(reversed(lr_padding)))
                # Pad the patch.
                patches = [
                    torch.nn.functional.pad(
                        p, pad=rl_padding, mode=self.PATCH_PADDING_MODE
                    )
                    for p in patches
                ]
                patch = patches[0]
                if tuple(patch.shape[2:]) == self.patch_size:
                    break

        if len(patches) == 1:
            patches = patches[0]
        return patches

    def forward(self, x_fake: torch.Tensor, *ignored_args):
        # If the input is not the given patch size, randomly crop and/or pad the patch.
        if tuple(x_fake.shape[2:]) != self.patch_size:
            x_fake = self._crop_or_pad_patch(x_fake)

        # The hinge 'generator loss'
        d_g = self.adversarial_net(x_fake)
        return self._adv_loss_fn(d_g, target_is_real=True, for_discriminator=False)

    def adversarial_loss(self, x_fake: torch.Tensor, x_real: torch.Tensor):
        # If the input is not the given patch size, randomly crop a patch from the
        # image. Assume both inputs are the same shape.
        if tuple(x_fake.shape[2:]) != self.patch_size:
            x_fake, x_real = self._crop_or_pad_patch(x_fake, x_real)

        l_real = self._adv_loss_fn(
            self.adversarial_net(x_real), target_is_real=True, for_discriminator=True
        )
        l_fake = self._adv_loss_fn(
            self.adversarial_net(x_fake), target_is_real=False, for_discriminator=True
        )

        return l_real + l_fake


class MonaiPatchAdversarialLeastSquaresLoss(MonaiPatchAdversarialHingeLoss):
    def __init__(
        self,
        spatial_dims: Literal[2, 3],
        in_channels: int,
        patch_size: tuple[int, ...],
        layer_out_channels: tuple[int, ...] = (8, 16, 32, 64, 1),
        layer_strides: tuple[int, ...] = (1, 2, 2, 2, 1),
        kernel_size: int | tuple[int, ...] = 3,
        num_res_units: int = 2,
        dropout: float = 0.25,
        reduction: str = "mean",
        use_spectral_norm: bool = False,
        **discriminator_kwargs,
    ):
        import mrinr

        self._init_kwargs = mrinr.nn.get_module_init_kwargs(
            locals(), extra_kwargs_dict=dict()
        )
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            patch_size=patch_size,
            layer_out_channels=layer_out_channels,
            layer_strides=layer_strides,
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            dropout=dropout,
            reduction=reduction,
            use_spectral_norm=use_spectral_norm,
            **discriminator_kwargs,
        )

        self._adv_loss_fn = monai.losses.PatchAdversarialLoss(
            criterion=monai.losses.adversarial_loss.AdversarialCriterions.LEAST_SQUARE,
            no_activation_leastsq=False,
            reduction=reduction,
        )

    def adversarial_loss(self, x_fake: torch.Tensor, x_real: torch.Tensor):
        # If the input is not the given patch size, randomly crop a patch from the
        # image. Assume both inputs are the same shape.
        if tuple(x_fake.shape[2:]) != self.patch_size:
            x_fake, x_real = self._crop_or_pad_patch(x_fake, x_real)

        l_real = self._adv_loss_fn(
            self.adversarial_net(x_real), target_is_real=True, for_discriminator=True
        )
        l_fake = self._adv_loss_fn(
            self.adversarial_net(x_fake), target_is_real=False, for_discriminator=True
        )

        return (l_real + l_fake) / 2


_LossFnDescriptorT = Union[
    str,
    Callable,
    Tuple[str, dict[str, Any]],
    Callable,
    Tuple[Callable, dict[str, Any]],
    torch.nn.Module,
    None,
]


def _init_fn_obj(
    fn_obj: _LossFnDescriptorT, ensure_module: bool = True
) -> torch.nn.Module | Callable | None:
    class _ModuleFnWrapper(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.wrapped_fn = fn

        def forward(self, *args, **kwargs):
            return self.wrapped_fn(*args, **kwargs)

    # fn_obj is a string with the name of the loss function with default kwargs.
    if isinstance(fn_obj, str):
        r = LOSS_CLS_LOOKUP[normalize_lookup_str(fn_obj)]()
    # fn_obj is a tuple with the name and kwargs of the loss function.
    elif isinstance(fn_obj, tuple) and isinstance(fn_obj[0], str):
        fn_name, fn_kwargs = fn_obj
        r = LOSS_CLS_LOOKUP[normalize_lookup_str(fn_name)](**fn_kwargs)
    # fn_obj is a tuple with the __init__() method and the __init__ kwargs.
    elif (
        isinstance(fn_obj, tuple)
        and callable(fn_obj[0])
        and inspect.ismethod(fn_obj[0])
    ):
        fn_init, fn_kwargs = fn_obj
        r = fn_init(**fn_kwargs)
    # fn_obj is just a Module.
    elif isinstance(fn_obj, torch.nn.Module):
        r = fn_obj
    # fn_obj is a callable/function.
    elif callable(fn_obj):
        # If fn is bound to an object, assume fn_obj is the __init__ method and it
        # needs to be instantiated.
        if inspect.ismethod(fn_obj):
            r = fn_obj()
        # Otherwise, assume the object is just a function that returns a loss.
        elif ensure_module:
            r = _ModuleFnWrapper(fn_obj)
        else:
            r = fn_obj
    # fn_obj is None, so no loss function is used.
    elif fn_obj is None:
        r = None
    else:
        raise ValueError(f"Invalid loss function object '{fn_obj}'")

    return r


class WeightedMaskLoss(torch.nn.Module):
    def __init__(
        self,
        unreduced_loss_fn: _LossFnDescriptorT,
        reduction: str = "mean",
        **loss_kwargs,
    ):
        super().__init__()
        self._unreduced_loss = _init_fn_obj(unreduced_loss_fn, ensure_module=False)
        self._reduction = reduction
        if self._reduction is None:
            self._reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elementwise_l: torch.Tensor = self._unreduced_loss(input, target)
        if self._reduction == "none":
            l = elementwise_l * (weight_mask if weight_mask is not None else 1)
        else:
            if weight_mask is not None:
                elementwise_l = elementwise_l * weight_mask
                # All 0-weighted elements are excluded from the reduction.
                bool_mask = weight_mask != 0
                elementwise_l.masked_fill_(~bool_mask, torch.nan)

            if self._reduction == "mean":
                l = torch.nanmean(elementwise_l)
            elif self._reduction == "sum":
                l = torch.nansum(elementwise_l)
            else:
                raise ValueError(f"Invalid reduction '{self._reduction}'")
        return l


class WeightedSumLoss(torch.nn.Module):
    def __init__(
        self,
        loss_fns: dict[str, _LossFnDescriptorT],
        loss_weights: Sequence[float],
        force_enable_loss_fns: Optional[Sequence[str]] = None,
    ):
        """Weighted sum of multiple loss functions.

        Parameters
        ----------
        loss_fns : dict[str, _LossFnDescriptorT]
            Mapping from loss function names to loss function objects.

            The loss function object can be one of the following:
            - A string, which is the name of a loss function in `LOSS_CLS_LOOKUP`.
            - A tuple of a string and a dictionary, where the string is the name of a
                loss function in `LOSS_CLS_LOOKUP`, and the dictionary contains keyword
                arguments to pass to the loss function constructor.
            - A callable, which is a function that returns a loss function object.
            - A tuple of a callable and a dictionary, where the callable is a function
                that returns a loss function object, and the dictionary contains keyword
                arguments to pass to the loss function constructor.
            - A `torch.nn.Module` object, which is a loss function object.

        loss_weights : Sequence[float]
            Weights to apply to each loss function.
        force_enable_loss_fns : Optional[Sequence[str]], optional
            Loss function names to force to be enabled, by default None

            If a loss function is not in `force_enable_loss_fns` and has a weight of 0,
            then it will be disabled.

        Raises
        ------
        ValueError
            Raised when the lengths of `loss_fns` and `loss_weights` do not match.
        """
        super().__init__()
        weights = np.asarray(loss_weights).reshape(-1).astype(float).tolist()

        if not (len(loss_fns.keys()) == len(weights)):
            raise ValueError(
                "ERROR: All loss function params must have the same length, got "
                + f"len(loss_fns) = {len(loss_fns)}, "
                + f"len(loss_weights) = {len(weights)}, "
            )

        l_fns: dict[str, torch.nn.Module] = collections.OrderedDict()
        l_ws = list()
        disabled_fn_names = list()
        if force_enable_loss_fns is None:
            enabled_fns = set()
        else:
            enabled_fns = set(force_enable_loss_fns)
        all_fn_names = list()
        for i, fn_name in enumerate(loss_fns.keys()):
            w = weights[i]
            # Only enable functions that have a non-zero weight, unless explicitly
            # forced to be enabled.
            if w != 0.0 or fn_name in enabled_fns:
                fn = _init_fn_obj(loss_fns[fn_name])
                if fn is not None:
                    l_ws.append(w)
                    l_fns[fn_name] = fn
                    if isinstance(fn, torch.nn.Module):
                        self.register_module(fn_name, l_fns[fn_name])
                else:
                    disabled_fn_names.append(fn_name)
                    self.__setattr__(fn_name, None)
            else:
                disabled_fn_names.append(fn_name)
                self.__setattr__(fn_name, None)
            all_fn_names.append(fn_name)

        self.loss_fns = torch.nn.ModuleDict(l_fns)
        self.register_buffer(
            "loss_weights",
            torch.from_numpy(np.asarray(l_ws)).to(torch.float32),
            persistent=True,
        )
        self.loss_weights: torch.Tensor
        self._fn_name_to_w_idx = {k: i for i, k in enumerate(self.loss_fns.keys())}
        self._all_fn_names = set(all_fn_names)
        self._disabled_fn_names = set(disabled_fn_names)

    def extra_repr(self) -> str:
        terms = list()
        for fn, fn_name, w in zip(
            self.loss_fns.values(), self.loss_fns.keys(), self.loss_weights
        ):
            terms.append(
                f"[{fn_name}] {round(w.detach().cpu().item(), ndigits=7)}*{str(fn)}"
            )
        return " + ".join(terms)

    def get_loss_weight(self, fn_name: str) -> float:
        if fn_name not in self._all_fn_names:
            raise ValueError(
                f"Invalid loss function name '{fn_name}'. "
                + f"Expected one of '{tuple(self._all_fn_names)}"
            )
        return self.loss_weights[self._fn_name_to_w_idx[fn_name]].detach().cpu().item()

    def update_loss_weight(self, fn_name: str, weight: float) -> None:
        if fn_name not in self._all_fn_names:
            raise ValueError(
                f"Invalid loss function name '{fn_name}'. "
                + f"Expected one of '{tuple(self._all_fn_names)}"
            )
        # Update the loss term weight.
        with torch.no_grad():
            self.loss_weights[self._fn_name_to_w_idx[fn_name]] = float(weight)

    def enable_loss_fn(self, fn_name: str) -> None:
        if fn_name not in self._all_fn_names:
            raise ValueError(
                f"Invalid loss function name '{fn_name}'. "
                + f"Expected one of '{tuple(self._all_fn_names)}"
            )
        if self.__getattr__(fn_name) is None:
            raise ValueError(
                f"Loss function '{fn_name}' was never stored. "
                + "Try setting 'force_enable_loss_fns' in the class init."
            )
        self._disabled_fn_names = self._disabled_fn_names - {fn_name}

    def disable_loss_fn(self, fn_name: str) -> None:
        if fn_name not in self._all_fn_names:
            raise ValueError(
                f"Invalid loss function name '{fn_name}'. "
                + f"Expected one of '{tuple(self._all_fn_names)}"
            )
        self._disabled_fn_names = self._disabled_fn_names | {fn_name}

    def forward(
        self,
        # return_unweighted_loss_terms: bool = False,
        return_loss_terms: bool = False,
        **loss_fn_inputs: torch.Tensor
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]],
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        # Loss function names are keys, and the values are either a tuple of args and
        # kwargs, just args, or just kwargs.
        losses = list()
        weights = list()
        # fn_names_used = list()
        loss_record = dict()
        for i, (fn_name, args_kwargs) in enumerate(loss_fn_inputs.items()):
            # Skip any loss functions that have been disabled.
            if fn_name in self._disabled_fn_names or args_kwargs is None:
                continue
            # Single pytorch Tensor
            if torch.is_tensor(args_kwargs):
                args = (args_kwargs,)
                kwargs = dict()
            # Just kwargs
            elif isinstance(args_kwargs, dict):
                args = tuple()
                kwargs = args_kwargs
            # Args and kwargs
            elif (
                isinstance(args_kwargs, (tuple, list))
                and len(args_kwargs) == 2
                and isinstance(args_kwargs[0], dict)
            ):
                args, kwargs = args_kwargs
                args = tuple(args)
            # Just args
            elif isinstance(args_kwargs, (tuple, list)):
                args = tuple(args_kwargs)
                kwargs = dict()

            l_i = self.loss_fns[fn_name](*args, **kwargs)
            losses.append(l_i)
            weights.append(self.loss_weights[self._fn_name_to_w_idx[fn_name]])
            # fn_names_used.append(fn_name)
            loss_record[fn_name] = dict(
                value=l_i.detach().cpu().item(),
                weight=weights[-1].detach().cpu().item(),
            )

        # Return dot product over weights and loss functions.
        losses = torch.stack(losses, 0).flatten()
        weights = torch.stack(weights, 0).flatten()
        L = torch.tensordot(losses, weights, dims=([0], [0]))
        if return_loss_terms:
            r = L, loss_record
        else:
            r = L
        return r


LOSS_CLS_LOOKUP = pyrsistent.m(
    mse=torch.nn.MSELoss,
    l1=torch.nn.L1Loss,
    smoothl1=torch.nn.SmoothL1Loss,
    kl=torch.nn.KLDivLoss,
    nll=torch.nn.NLLLoss,
    crossentropy=torch.nn.CrossEntropyLoss,
    bce=torch.nn.BCELoss,
    bcewithlogits=torch.nn.BCEWithLogitsLoss,
    ssim=SSIMLoss,
    msssim=MS_SSIMLoss,
    maskedmse=MaskedMSELoss,
    maskedl1=MaskedL1Loss,
    maskedsmoothl1=MaskedSmoothL1Loss,
    klidgaussian=KLDivergenceIdGaussianLoss,
    monaipatchadversarialhinge=MonaiPatchAdversarialHingeLoss,
    monaipatchadversarialleastsquares=MonaiPatchAdversarialLeastSquaresLoss,
    monaiperceptualloss=MonaiPerceptualLoss,
    weightedsumloss=WeightedSumLoss,
)
