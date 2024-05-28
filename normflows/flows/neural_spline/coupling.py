"""
Implementations of various coupling layers.
Code taken from https://github.com/bayesiains/nsf
"""

import warnings

import numpy as np
import torch
from torch import nn
from vegas import AdaptiveMap

from ..base import Flow
from ... import utils
from ...utils import splines


class Coupling(Flow):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None):
        """Constructor.

        mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:

        - if `mask[i] > 0`, `input[i]` will be transformed.
        - if `mask[i] <= 0`, `input[i]` will be passed unchanged.

        Args:
          mask
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer(
            "identity_features", features_vector.masked_select(mask <= 0)
        )
        self.register_buffer(
            "transform_features", features_vector.masked_select(mask > 0)
        )

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier(),
        )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split, context
            )
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split, context
            )

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class PiecewiseCoupling(Coupling):
    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, utils.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()


# class PieceWiseVegasCoupling(PiecewiseCoupling):
class PieceWiseVegasCoupling(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        func,
        num_input_channels,
        integration_region,
        batchsize,
        num_increments=1000,
        niters=20,
    ):
        super().__init__()

        vegas_map = AdaptiveMap(integration_region, ninc=num_increments)
        y = np.random.uniform(0.0, 1.0, (batchsize, num_input_channels))
        vegas_map.adapt_to_samples(y, func(y), nitn=niters)

        self.register_buffer("y", torch.Tensor(y))
        self.register_buffer("grid", torch.Tensor(vegas_map.grid))
        self.register_buffer("inc", torch.Tensor(vegas_map.inc))
        self.register_buffer("dim", torch.tensor(num_input_channels))
        self.register_buffer("x", torch.empty_like(self.y))
        self.register_buffer("jac", torch.ones(batchsize))
        if num_increments < 1000:
            self.register_buffer("ninc", torch.tensor(1000))
        else:
            self.register_buffer("ninc", torch.tensor(num_increments))

    @torch.no_grad()
    def forward(self, y):
        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()
        dy_ninc = y_ninc - iy

        self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            mask = iy[:, d] < self.ninc
            if mask.any():
                self.x[mask, d] = (
                    self.grid[d, iy[mask, d]]
                    + self.inc[d, iy[mask, d]] * dy_ninc[mask, d]
                )
                self.jac[mask] *= self.inc[d, iy[mask, d]] * self.ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                self.x[mask_inv, d] = self.grid[d, self.ninc]
                self.jac[mask_inv] *= self.inc[d, self.ninc - 1] * self.ninc

        return self.x, torch.log(self.jac)

    @torch.no_grad()
    def inverse(self, x):
        self.jac.fill_(1.0)
        for d in range(self.dim):
            iy = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

            mask_valid = (iy > 0) & (iy <= self.ninc)
            mask_lower = iy <= 0
            mask_upper = iy > self.ninc

            # Handle valid range (0 < iy <= self.ninc)
            if mask_valid.any():
                iyi_valid = iy[mask_valid] - 1
                self.y[mask_valid, d] = (
                    iyi_valid
                    + (x[mask_valid, d] - self.grid[d, iyi_valid])
                    / self.inc[d, iyi_valid]
                ) / self.ninc
                self.jac[mask_valid] *= self.inc[d, iyi_valid] * self.ninc

            # Handle lower bound (iy <= 0)\
            if mask_lower.any():
                self.y[mask_lower, d] = 0.0
                self.jac[mask_lower] *= self.inc[d, 0] * self.ninc

            # Handle upper bound (iy > self.ninc)
            if mask_upper.any():
                self.y[mask_upper, d] = 1.0
                self.jac[mask_upper] *= self.inc[d, self.ninc - 1] * self.ninc

        return self.y, torch.log(1 / self.jac)


# class PieceWiseVegasCoupling(nn.Module):
#     def __init__(
#         self,
#         func,
#         num_input_channels,
#         integration_region,
#         batchsize,
#         apply_unconditional_transform=False,
#     ):
#         super().__init__()

#         self.integration_region = integration_region
#         self.apply_unconditional_transform = apply_unconditional_transform

#         if apply_unconditional_transform:
#             self.unconditional_transform = lambda: PieceWiseVegasCDF(
#                 func=func,
#                 num_input_channels=num_input_channels,
#                 integration_region=integration_region,
#                 batchsize=batchsize,
#             )
#         else:
#             self.unconditional_transform = None

#         # self.transform_net = nn.Sequential(
#         #     nn.Linear(num_input_channels, num_input_channels * 2),
#         #     nn.ReLU(),
#         #     nn.Linear(num_input_channels * 2, num_input_channels * 2),
#         # )

#     def forward(self, inputs, context=None):
#         return self._piecewise_vegas(inputs, inverse=False, context=context)

#     def inverse(self, inputs, context=None):
#         return self._piecewise_vegas(inputs, inverse=True, context=context)

#     def _piecewise_vegas(self, inputs, inverse=False, context=None):
#         transform_features = inputs
#         identity_features = inputs

#         transform_params = self.transform_net(transform_features)
#         if self.unconditional_transform:
#             transform_params = self.unconditional_transform()

#         outputs, logabsdet = PieceWiseVegasCDF(
#             num_input_channels=transform_features.shape[1],
#             integration_region=self.integration_region,
#         ).forward(transform_params)

#         if inverse:
#             outputs = torch.cat([identity_features, outputs], dim=-1)
#         else:
#             outputs = torch.cat([outputs, identity_features], dim=-1)

#         return outputs, logabsdet


class PiecewiseLinearCDF(Flow):
    def __init__(
        self,
        shape,
        num_bins=10,
        identity_init=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        # min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        # self.min_bin_height = min_bin_height

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            # self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            # self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

    @staticmethod
    def _share_across_batch(params, batch_size):
        return params[None, ...].expand(batch_size, *params.shape)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = self._share_across_batch(
            self.unnormalized_widths, batch_size
        )
        # unnormalized_heights = self._share_across_batch(
        #     self.unnormalized_heights, batch_size
        # )

        spline_fn = splines.linear_spline
        spline_kwargs = {}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            **spline_kwargs,
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseLinearCoupling(PiecewiseCoupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        img_shape=None,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        # min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        # self.min_bin_height = min_bin_height

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        # if apply_unconditional_transform:
        #     unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
        #         shape=[features] + (img_shape if img_shape else []),
        #         num_bins=num_bins,
        #         tails=tails_,
        #         tail_bound=tail_bound_,
        #         min_bin_width=min_bin_width,
        #         min_bin_height=min_bin_height,
        #         min_derivative=min_derivative,
        #     )
        # else:
        #     unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            # unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        # unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            # unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            # unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        spline_fn = splines.linear_spline
        spline_kwargs = {}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            # unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            # min_bin_height=self.min_bin_height,
            **spline_kwargs,
        )


class PiecewiseRationalQuadraticCDF(Flow):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound
        self.tails = tails

        if self.tails == "linear":
            num_derivatives = num_bins - 1
        elif self.tails == "circular":
            num_derivatives = num_bins
        else:
            num_derivatives = num_bins + 1

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    @staticmethod
    def _share_across_batch(params, batch_size):
        return params[None, ...].expand(batch_size, *params.shape)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = self._share_across_batch(
            self.unnormalized_widths, batch_size
        )
        unnormalized_heights = self._share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnormalized_derivatives = self._share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseRationalQuadraticCoupling(PiecewiseCoupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        if isinstance(tails, list) or isinstance(tails, tuple):
            self.tails = [tails[i] for i in transform_features]
            tails_ = [tails[i] for i in identity_features]
        else:
            self.tails = tails
            tails_ = tails

        if torch.is_tensor(tail_bound):
            tail_bound_ = tail_bound[identity_features]
        else:
            self.tail_bound = tail_bound
            tail_bound_ = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails_,
                tail_bound=tail_bound_,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound[transform_features])

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        # print("width:", unnormalized_widths)
        # print("height:", unnormalized_heights)
        # print("derivatives:", unnormalized_derivatives)
        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )
