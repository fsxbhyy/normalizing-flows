import torch
from vegas import AdaptiveMap
from warnings import warn
from scipy.stats import kstest


class VegasMap(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        target,
        # vegas_map,
        num_input_channels,
        integration_region,
        batchsize,
        num_adapt_samples=1000000,
        num_increments=1000,
        niters=20,
    ):
        super().__init__()

        vegas_map = AdaptiveMap(integration_region, ninc=num_increments)
        # y = np.random.uniform(0.0, 1.0, (num_adapt_samples, num_input_channels))
        y = torch.rand(num_adapt_samples, num_input_channels, dtype=torch.float64)
        prob = torch.empty(num_adapt_samples)
        nblock = num_adapt_samples // batchsize
        for i in range(nblock):
            # prob.append(func(y[i * batchsize : (i + 1) * batchsize]))
            prob[i * batchsize : (i + 1) * batchsize] = target.prob(
                y[i * batchsize : (i + 1) * batchsize]
            )
        vegas_map.adapt_to_samples(
            y[: nblock * batchsize, :], prob[: nblock * batchsize], nitn=niters
        )

        self.register_buffer("y", torch.empty(batchsize, num_input_channels))
        self.register_buffer("grid", torch.Tensor(vegas_map.grid))
        self.register_buffer("inc", torch.Tensor(vegas_map.inc))
        self.register_buffer("ninc", torch.tensor(num_increments))
        self.register_buffer("dim", torch.tensor(num_input_channels))
        self.register_buffer("x", torch.empty(batchsize, num_input_channels))
        self.register_buffer("jac", torch.ones(batchsize))

        self.target = target

    @torch.no_grad()
    def forward(self, y):
        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()
        dy_ninc = y_ninc - iy

        x = torch.empty_like(y)
        jac = torch.ones(y.shape[0], device=x.device)
        # self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            mask = iy[:, d] < self.ninc
            if mask.any():
                x[mask, d] = (
                    self.grid[d, iy[mask, d]]
                    + self.inc[d, iy[mask, d]] * dy_ninc[mask, d]
                )
                jac[mask] *= self.inc[d, iy[mask, d]] * self.ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                x[mask_inv, d] = self.grid[d, self.ninc]
                jac[mask_inv] *= self.inc[d, self.ninc - 1] * self.ninc

        return x, jac

    @torch.no_grad()
    def inverse(self, x):
        # self.jac.fill_(1.0)
        y = torch.empty_like(x)
        jac = torch.ones(x.shape[0], device=x.device)
        for d in range(self.dim):
            iy = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

            mask_valid = (iy > 0) & (iy <= self.ninc)
            mask_lower = iy <= 0
            mask_upper = iy > self.ninc

            # Handle valid range (0 < iy <= self.ninc)
            if mask_valid.any():
                iyi_valid = iy[mask_valid] - 1
                y[mask_valid, d] = (
                    iyi_valid
                    + (x[mask_valid, d] - self.grid[d, iyi_valid])
                    / self.inc[d, iyi_valid]
                ) / self.ninc
                jac[mask_valid] *= self.inc[d, iyi_valid] * self.ninc

            # Handle lower bound (iy <= 0)\
            if mask_lower.any():
                y[mask_lower, d] = 0.0
                jac[mask_lower] *= self.inc[d, 0] * self.ninc

            # Handle upper bound (iy > self.ninc)
            if mask_upper.any():
                y[mask_upper, d] = 1.0
                jac[mask_upper] *= self.inc[d, self.ninc - 1] * self.ninc

        return y, jac

    @torch.no_grad()
    def integrate_block(self, num_blocks):
        print("Estimating integral from trained network")

        num_samples = self.y.shape[0]
        num_vars = self.y.shape[1]
        # Pre-allocate tensor for storing means and histograms
        means_t = torch.empty(num_blocks, device=self.y.device)

        # Loop to fill the tensor with mean values
        for i in range(num_blocks):
            self.y[:] = torch.rand(num_samples, num_vars, device=self.y.device)
            self.x[:], self.jac[:] = self.forward(self.y)

            res = torch.Tensor(self.target.prob(self.x)) * self.jac
            means_t[i] = torch.mean(res, dim=0)

        while (
            kstest(
                means_t.cpu(),
                "norm",
                args=(means_t.mean().item(), means_t.std().item()),
            )[1]
            < 0.05
        ):
            print("correlation too high, merge blocks")
            if num_blocks <= 64:
                warn(
                    "blocks too small, try increasing num_blocks",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            means_t = (
                means_t[torch.arange(0, num_blocks * 2, 2, device=self.y.device)]
                + means_t[torch.arange(1, num_blocks * 2, 2, device=self.y.device)]
            ) / 2.0
        statistic, p_value = kstest(
            means_t.cpu(), "norm", args=(means_t.mean().item(), means_t.std().item())
        )
        print(f"K-S test: statistic {statistic}, p-value {p_value}.")

        # Compute mean and standard deviation directly on the tensor
        mean_combined = torch.mean(means_t)
        std_combined = torch.std(means_t) / num_blocks**0.5

        return (
            mean_combined,
            std_combined,
        )

    @torch.no_grad()
    def integrate_block_histr(self, num_blocks, bins=25, hist_range=(0.0, 1.0)):
        print("Estimating integral from trained network")

        num_samples = self.y.shape[0]
        num_vars = self.y.shape[1]
        means_t = torch.empty(num_blocks, device=self.y.device)
        # Pre-allocate tensor for storing means and histograms
        with torch.device("cpu"):
            if isinstance(bins, int):
                histr = torch.zeros(bins, num_vars, device=self.y.device)
                histr_weight = torch.zeros(bins, num_vars, device=self.y.device)
            else:
                histr = torch.zeros(bins.shape[0], num_vars, device=self.y.device)
                histr_weight = torch.zeros(
                    bins.shape[0], num_vars, device=self.y.device
                )

        # Loop to fill the tensor with mean values
        for i in range(num_blocks):
            self.y[:] = torch.rand(num_samples, num_vars, device=self.y.device)
            self.x[:], self.jac[:] = self.forward(self.y)

            res = torch.Tensor(self.target.prob(self.x)) * self.jac
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

    def mcmc(self, len_chain=1000, burn_in=None, thinning=1, alpha=1.0, step_size=0.05):
        """
        Perform MCMC integration using batch processing. Using the Metropolis-Hastings algorithm to sample the distribution:
        Pi(x) = alpha * q(x) + (1 - alpha) * p(x),
        where q(x) is the learned distribution by the VEGAS map, and p(x) is the target distribution.

        Args:
            len_chain: Number of samples to draw.
            burn_in: Number of initial samples to discard.
            thinning: Interval to thin the chain.
            alpha: Annealing parameter.
            step_size: random walk step size.

        Returns:
            mean, error: Mean and standard variance of the integrated samples.
        """
        # epsilon = 1e-10  # Small value to ensure numerical stability
        device = self.y.device
        vars_shape = self.y.shape
        batch_size = vars_shape[0]
        num_vars = vars_shape[1]
        if burn_in is None:
            burn_in = len_chain // 4

        # Initialize chains
        self.y[:] = torch.rand(vars_shape, device=device)
        current_samples, current_qinv = self.forward(self.y)

        proposed_y = torch.empty(vars_shape, device=device)
        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_qinv = torch.empty(batch_size, device=device)

        current_weight = alpha / current_qinv + (1 - alpha) * torch.abs(
            self.target.prob(current_samples)
        )  # Pi(x) = alpha * q(x) + (1 - alpha) * p(x)
        new_weight = torch.empty(batch_size, device=device)

        bool_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        for _ in range(burn_in):
            # bool_mask[:] = torch.rand(batch_size, device=device) < 0.5
            bool_mask[:] = True
            num_rand = bool_mask.sum().item()

            # Propose new samples
            proposed_y[bool_mask, :] = torch.rand(num_rand, num_vars, device=device)
            proposed_y[~bool_mask, :] = (
                self.y[~bool_mask, :]
                + (torch.rand(batch_size - num_rand, num_vars, device=device) - 0.5)
                * step_size
            ) % 1.0

            proposed_samples[:], proposed_qinv[:] = self.forward(proposed_y)

            new_weight[:] = alpha / proposed_qinv + (1 - alpha) * torch.abs(
                self.target.prob(proposed_samples)
            )

            # Compute acceptance probabilities
            acceptance_probs = (
                new_weight / current_weight * proposed_qinv / current_qinv
            )

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            self.y = torch.where(accept.unsqueeze(1), proposed_y, self.y)
            current_samples = torch.where(
                accept.unsqueeze(1), proposed_samples, current_samples
            )
            current_weight = torch.where(accept, new_weight, current_weight)
            current_qinv = torch.where(accept, proposed_qinv, current_qinv)
            # self.p.log_q[accept] = proposed_log_q[accept]

        current_prob = self.target.prob(current_samples)
        new_prob = torch.empty_like(current_prob)
        values = torch.zeros(batch_size, device=device)
        ref_values = torch.zeros_like(values)
        abs_values = torch.zeros_like(values)
        var_p = torch.zeros_like(values)
        var_q = torch.zeros_like(values)
        num_measure = 0
        for i in range(len_chain):
            # bool_mask[:] = torch.rand(batch_size, device=device) < 0.5
            bool_mask[:] = True
            num_rand = bool_mask.sum().item()

            # Propose new samples
            proposed_y[bool_mask, :] = torch.rand(num_rand, num_vars, device=device)
            proposed_y[~bool_mask, :] = (
                self.y[~bool_mask, :]
                + (torch.rand(batch_size - num_rand, num_vars, device=device) - 0.5)
                * step_size
            ) % 1.0

            proposed_samples[:], proposed_qinv[:] = self.forward(proposed_y)
            new_prob[:] = self.target.prob(proposed_samples)

            new_weight[:] = alpha / proposed_qinv + (1 - alpha) * torch.abs(new_prob)
            acceptance_probs = (
                new_weight / current_weight * proposed_qinv / current_qinv
            )

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            if i % 400 == 0:
                print("acceptance rate: ", accept.sum().item() / batch_size)

            self.y = torch.where(accept.unsqueeze(1), proposed_y, self.y)
            current_prob = torch.where(accept, new_prob, current_prob)
            current_weight = torch.where(accept, new_weight, current_weight)
            current_qinv = torch.where(accept, proposed_qinv, current_qinv)

            # Measurement
            if i % thinning == 0:
                num_measure += 1

                values += current_prob / current_weight
                ref_values += 1 / (current_qinv * current_weight)
                abs_values += torch.abs(current_prob) / current_weight

                var_p += (current_prob / current_weight) ** 2
                var_q += 1 / (current_qinv * current_weight) ** 2

        values /= num_measure
        abs_values /= num_measure
        ref_values /= num_measure
        var_p /= num_measure
        var_q /= num_measure

        while (
            kstest(
                values.cpu(), "norm", args=(values.mean().item(), values.std().item())
            )[1]
            < 0.05
        ):
            print("correlation too high, merge blocks")
            if batch_size <= 64:
                warn(
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            batch_size //= 2
            even_idx = torch.arange(0, batch_size * 2, 2, device=device)
            odd_idx = torch.arange(1, batch_size * 2, 2, device=device)
            values = (values[even_idx] + values[odd_idx]) / 2.0
            abs_values = (abs_values[even_idx] + abs_values[odd_idx]) / 2.0
            ref_values = (ref_values[even_idx] + ref_values[odd_idx]) / 2.0
            var_p = (var_p[even_idx] + var_p[odd_idx]) / 2.0
            var_q = (var_q[even_idx] + var_q[odd_idx]) / 2.0
        print("new batch size: ", batch_size)

        statistic, p_value = kstest(
            values.cpu(), "norm", args=(values.mean().item(), values.std().item())
        )
        print(f"K-S test of values: statistic {statistic}, p-value {p_value}")

        statistic, p_value = kstest(
            ref_values.cpu(),
            "norm",
            args=(ref_values.mean().item(), ref_values.std().item()),
        )
        print(f"K-S test of ref_values: statistic {statistic}, p-value {p_value}")

        mean = torch.mean(values) / torch.mean(ref_values)
        abs_val_mean = torch.mean(abs_values) / torch.mean(ref_values)

        cov_matrix = torch.cov(torch.stack((values, ref_values)))
        print("covariance matrix: ", cov_matrix)
        ratio_var = (
            cov_matrix[0, 0] - 2 * mean * cov_matrix[0, 1] + mean**2 * cov_matrix[1, 1]
        ) / torch.mean(ref_values) ** 2
        ratio_err = (ratio_var / batch_size) ** 0.5

        values /= ref_values
        print("correlation of ratio values: ", calculate_correlation(values).item())
        _mean = torch.mean(values)
        _std = torch.std(values)
        error = _std / batch_size**0.5

        print("old result: {:.5e} +- {:.5e}".format(_mean.item(), error.item()))

        statistic, p_value = kstest(
            values.cpu(), "norm", args=(_mean.item(), _std.item())
        )
        print(
            "K-S test of ratio values: statistic {:.5e}, p-value {:.5e}",
            statistic,
            p_value,
        )

        abs_values /= ref_values
        err_absval = torch.std(abs_values) / batch_size**0.5
        print(
            "|f(x)| Integration results: {:.5e} +/- {:.5e}".format(
                abs_val_mean.item(), err_absval.item()
            )
        )

        err_var_p = torch.std(var_p) / batch_size**0.5
        err_var_q = torch.std(var_q) / batch_size**0.5
        print(
            "variance of p: {:.5e} +/- {:.5e}".format(
                var_p.mean().item(), err_var_p.item()
            )
        )
        print(
            "variance of q: {:.5e} +/- {:.5e}".format(
                var_q.mean().item(), err_var_q.item()
            )
        )

        return mean, ratio_err


# Function to calculate correlation between adjacent blocks
def calculate_correlation(x):
    x_centered = x[1:] - x.mean()
    y_centered = x[:-1] - x.mean()
    cov = torch.sum(x_centered * y_centered) / (len(x) - 1)
    return torch.abs(cov / torch.var(x))
