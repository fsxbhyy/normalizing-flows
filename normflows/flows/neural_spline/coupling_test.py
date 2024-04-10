"""
Tests for the coupling Transforms.
Code partially taken from https://github.com/bayesiains/nsf
"""

import torch
import unittest

from normflows import nets as nn_

from normflows.flows.neural_spline import coupling
from normflows.flows.flow_test import FlowTest
from normflows import utils


def create_coupling_transform(shape, type,**kwargs):
    if len(shape) == 1:

        def create_net(in_features, out_features):
            return nn_.ResidualNet(
                in_features, out_features, hidden_features=30, num_blocks=5
            )

    else:

        def create_net(in_channels, out_channels):
            # return nn.Conv2d(in_channels, out_channels, kernel_size=1)
            return nn_.ConvResidualNet(
                in_channels=in_channels, out_channels=out_channels, hidden_channels=16
            )

    mask = utils.masks.create_mid_split_binary_mask(shape[0])
    if type == "linear":
        return coupling.PiecewiseLinearCoupling(mask=mask, transform_net_create_fn=create_net, **kwargs), mask
    elif type=="rationalQuadratic":
        return coupling.PiecewiseRationalQuadraticCoupling(mask=mask, transform_net_create_fn=create_net, **kwargs), mask


batch_size = 5



class PiecewiseCouplingTransformTest(FlowTest):
    shapes = [[20], [2, 4, 4]]

    def test_rqcs(self):
        for shape in self.shapes:
            for type in ["linear", "rationalQuadratic"]:
                for tails in [None, "linear"]:
                    with self.subTest(shape=shape, type=type, tails=tails):
                        inputs = torch.rand(batch_size, *shape)
                        if  type=="linear":
                            flow, _ = create_coupling_transform(shape, type)
                        else:
                            flow, _ = create_coupling_transform(shape, type, tails=tails)
                        self.checkForwardInverse(flow, inputs)

    def test_rqcs_unconditional(self):
        for shape in self.shapes:
            with self.subTest(shape=shape):
                inputs = torch.rand(batch_size, *shape)
                img_shape = shape[1:] if len(shape) > 1 else None
                flow, _ = create_coupling_transform(
                    shape, "rationalQuadratic", apply_unconditional_transform=True, img_shape=img_shape
                )
                self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()
