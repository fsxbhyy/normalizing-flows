import torch
import numpy as np
from vegas import AdaptiveMap


class VegasMap(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        func,
        # vegas_map,
        num_input_channels,
        integration_region,
        batchsize,
        num_increments=1000,
        niters=20,
    ):
        super().__init__()

        vegas_map = AdaptiveMap(integration_region, ninc=num_increments)
        y = np.random.uniform(0.0, 1.0, (batchsize, num_input_channels))
        # y = torch.rand(batchsize, num_input_channels, dtype=torch.float64)
        vegas_map.adapt_to_samples(y, func(y), nitn=niters)

        self.register_buffer("y", torch.Tensor(y))
        self.register_buffer("grid", torch.Tensor(vegas_map.grid))
        self.register_buffer("inc", torch.Tensor(vegas_map.inc))
        self.register_buffer("ninc", torch.tensor(num_increments))
        self.register_buffer("dim", torch.tensor(num_input_channels))
        self.register_buffer("x", torch.empty_like(self.y))
        self.register_buffer("jac", torch.ones(batchsize))

    @torch.no_grad()
    def forward(self, y):
        # y0 = y.detach().numpy()
        # y0 = np.array(y0, dtype=np.float64)
        # x = np.empty(y.shape, float)
        # jac = np.empty(y.shape[0], float)
        # self.vegas_map.map(y0, x, jac)

        # print("torch init:", x)
        # print(jac)

        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()
        dy_ninc = y_ninc - iy

        self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            mask = iy[:, d] < self.ninc
            if mask.any():
                self.x[mask, d] = (
                    # self.grid[iy[mask, d], d] + self.inc[iy[mask, d], d] * dy_ninc[mask, d]
                    self.grid[d, iy[mask, d]]
                    + self.inc[d, iy[mask, d]] * dy_ninc[mask, d]
                )
                # self.jac[mask] *= self.inc[iy[mask, d], d] * self.ninc
                self.jac[mask] *= self.inc[d, iy[mask, d]] * self.ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                # self.x[mask_inv, d] = self.grid[self.ninc, d]
                self.x[mask_inv, d] = self.grid[d, self.ninc]
                # self.jac[mask_inv] *= self.inc[self.ninc - 1, d] * self.ninc
                self.jac[mask_inv] *= self.inc[d, self.ninc - 1] * self.ninc

        return self.x, torch.log(self.jac)
        # return self.x, self.jac

    @torch.no_grad()
    def inverse(self, x):
        self.jac.fill_(1.0)
        for d in range(self.dim):
            # iy = torch.searchsorted(self.grid[:, d], x[:, d], right=True)
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
                    # + (x[mask_valid, d] - self.grid[iyi_valid, d]) / self.inc[iyi_valid, d]
                ) / self.ninc
                self.jac[mask_valid] *= self.inc[d, iyi_valid] * self.ninc
                # self.jac[mask_valid] *= self.inc[iyi_valid, d] * self.ninc

            # Handle lower bound (iy <= 0)\
            if mask_lower.any():
                self.y[mask_lower, d] = 0.0
                self.jac[mask_lower] *= self.inc[d, 0] * self.ninc
                # self.jac[mask_lower] *= self.inc[0, d] * self.ninc

            # Handle upper bound (iy > self.ninc)
            if mask_upper.any():
                self.y[mask_upper, d] = 1.0
                self.jac[mask_upper] *= self.inc[d, self.ninc - 1] * self.ninc
                # self.jac[mask_upper] *= self.inc[self.ninc - 1, d] * self.ninc

        return self.y, torch.log(1 / self.jac)
        # return self.y, self.jac
