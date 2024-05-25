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

        self.func = func

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

        # return self.x, torch.log(self.jac)
        return self.x, self.jac

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

        # return self.y, torch.log(1 / self.jac)
        return self.y, self.jac

    @torch.no_grad()
    def integrate_block(self, num_blocks, bins=25, hist_range=(0.0, 1.0)):
        print("Estimating integral from trained network")

        num_samples = self.y.shape[0]
        num_vars = self.y.shape[1]
        # Pre-allocate tensor for storing means and histograms
        means_t = torch.zeros(num_blocks)
        with torch.device("cpu"):
            if isinstance(bins, int):
                histr = torch.zeros(bins, num_vars)
                histr_weight = torch.zeros(bins, num_vars)
            else:
                histr = torch.zeros(bins.shape[0], num_vars)
                histr_weight = torch.zeros(bins.shape[0], num_vars)

        # Loop to fill the tensor with mean values
        for i in range(num_blocks):
            self.y = torch.rand(num_samples, num_vars)
            self.x, self.jac = self.forward(self.y)
            # for flow in self.flows:
            #     self.p.samples, self.p.log_det = flow(self.p.samples)
            #     self.p.log_q -= self.p.log_det
            # self.p.log_q = torch.exp(self.p.log_q)
            # self.p.var = self.p.prob(self.p.samples)
            res = torch.Tensor(self.func(self.x)) * self.jac
            means_t[i] = torch.mean(res, dim=0)

            z = self.x.detach().cpu()
            weights = res.detach().cpu()
            for d in range(num_vars):
                hist, bin_edges = torch.histogram(
                    z[:, d], bins=bins, range=hist_range, density=True
                )
                histr[:, d] += hist
                hist, bin_edges = torch.histogram(
                    z[:, d],
                    bins=bins,
                    range=hist_range,
                    weight=weights,
                    density=True,
                )
                histr_weight[:, d] += hist
        # Compute mean and standard deviation directly on the tensor
        mean_combined = torch.mean(means_t)
        std_combined = torch.std(means_t) / num_blocks**0.5

        return (
            mean_combined,
            std_combined,
            bin_edges,
            histr / num_blocks,
            histr_weight / num_blocks,
        )
