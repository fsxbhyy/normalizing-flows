import torch
import torch.nn as nn
import numpy as np
from warnings import warn
from scipy.stats import kstest

from . import distributions
from . import utils


# Function to calculate correlation between adjacent blocks
def calculate_correlation(x):
    x_centered = x[1:] - x.mean()
    y_centered = x[:-1] - x.mean()
    cov = torch.sum(x_centered * y_centered) / (len(x) - 1)
    return torch.abs(cov / torch.var(x))


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def IS_chi2(self, num_samples=1):
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        # utils.set_requires_grad(self, False)
        prob = torch.abs(self.p.prob(z))
        q = torch.exp(log_q)
        pmean = torch.mean(prob / q)
        prob = prob / pmean
        # print("test:", prob, "\n", ISratio, "\n", log_q, "\n")
        # print( -torch.mean(ISratio.detach()*log_q))
        # utils.set_requires_grad(self, True)
        return torch.mean(torch.square(prob.detach() - q) / q / q.detach())

    def IS_forward_kld(self, num_samples=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        if num_samples is None:
            num_samples = self.p.batchsize
        utils.set_requires_grad(self, False)
        z, _ = self.q0(num_samples)
        for flow in self.flows:
            z, _ = flow(z)
        utils.set_requires_grad(self, True)
        z_, log_q = self.inverse_and_log_det(z)
        # utils.set_requires_grad(self, False)
        prob = torch.abs(self.p.prob(z))
        q = torch.exp(log_q)
        pmean = torch.mean(prob / q)
        prob = prob / pmean
        logp = torch.where(prob > 1e-16, torch.log(prob), torch.log(prob + 1e-16))
        ISratio = prob / q
        # print("test:", prob, "\n", ISratio, "\n", log_q, "\n")
        # print( -torch.mean(ISratio.detach()*log_q))
        # utils.set_requires_grad(self, True)
        return torch.mean(ISratio.detach() * (logp.detach() - log_q))

    def IS_forward_kld_direct(self, z):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """

        z_, log_q = self.inverse_and_log_det(z)
        # utils.set_requires_grad(self, False)
        prob = torch.abs(self.p.prob(z))
        q = torch.exp(log_q)
        pmean = torch.mean(prob / q)
        prob = prob / pmean
        logp = torch.where(prob > 1e-16, torch.log(prob), torch.log(prob + 1e-16))
        ISratio = prob / q
        # print("test:", prob, "\n", ISratio, "\n", log_q, "\n")
        # print( -torch.mean(ISratio.detach()*log_q))
        # utils.set_requires_grad(self, True)
        return torch.mean(ISratio.detach() * (logp.detach() - log_q))

    # def forward_kld_mc(self, steps=10):
    #     z = self.p.sample(steps)

    #     self.p.log_q.fill_(0.0)
    #     for i in range(len(self.flows) - 1, -1, -1):
    #         z, self.p.log_det = self.flows[i].inverse(z)
    #         self.p.log_q += self.p.log_det
    #     self.p.log_q += self.q0.log_prob(z)
    #     return -torch.mean(self.p.log_q)

    def MCvar(self, num_samples=1):
        z, log_q_ = self.q0(num_samples)
        log_J = torch.zeros_like(log_q_)
        # log_J += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_J += log_det
        log_p = self.p.log_prob(z)
        return torch.mean(torch.exp(2 * log_p + 2 * log_J))

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det

        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """Alpha divergence when sampling from q

        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)

        Returns:
          Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q
        # self.p.samples[:], self.p.log_q[:] = self.q0(num_samples)
        # for flow in self.flows:
        #     self.p.samples[:], log_det = flow(self.p.samples)
        #     self.p.log_q -= log_det
        # return self.p.samples, self.p.log_q

    @torch.no_grad()
    def integrate(self):
        """Importance sampling integration with flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          mean, variance
        """
        num_samples = self.p.batchsize
        z, log_q = self.sample(num_samples)
        q = torch.exp(log_q)
        func = self.p.prob(z)
        return torch.mean(func / q, dim=0)

    @torch.no_grad()
    def integrate_block(
        self, num_blocks, bins=25, hist_range=(0.0, 1.0), correlation_threshold=0.2
    ):
        print("Estimating integral from trained network")

        device = self.p.samples.device
        num_samples = self.p.batchsize
        means_t = torch.zeros(num_blocks, device=device)
        # Pre-allocate tensor for storing means and histograms
        # num_vars = self.p.ndims
        # with torch.device("cpu"):
        #     if isinstance(bins, int):
        #         histr = torch.zeros(bins, num_vars)
        #         histr_weight = torch.zeros(bins, num_vars)
        #     else:
        #         histr = torch.zeros(bins.shape[0], num_vars)
        #         histr_weight = torch.zeros(bins.shape[0], num_vars)

        partition_z = torch.tensor(0.0, device=device)
        for i in range(num_blocks):
            self.p.samples[:], self.p.log_q[:] = self.q0(num_samples)
            for flow in self.flows:
                self.p.samples[:], log_det = flow(self.p.samples)
                self.p.log_q -= log_det
            self.p.val[:] = self.p.prob(self.p.samples)
            q = torch.exp(self.p.log_q)
            res = self.p.val / q
            means_t[i] = torch.mean(res, dim=0)

            partition_z += torch.mean(torch.abs(self.p.val) / q, dim=0)
            # log_p = torch.log(torch.clamp(prob_abs, min=1e-16))
            # loss += prob_abs / q / z * (log_p - self.p.log_q - torch.log(z))

            # z = self.p.samples.detach().cpu()
            # weights = (res / res.abs()).detach().cpu()
            # for d in range(num_vars):
            #     hist, bin_edges = torch.histogram(
            #         z[:, d], bins=bins, range=hist_range, density=True
            #     )
            #     histr[:, d] += hist
            #     hist, bin_edges = torch.histogram(
            #         z[:, d],
            #         bins=bins,
            #         range=hist_range,
            #         weight=weights,
            #         density=True,
            #     )
            #     histr_weight[:, d] += hist
        partition_z /= num_blocks
        # while calculate_correlation(means_t) > correlation_threshold:
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
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            means_t = (
                means_t[torch.arange(0, num_blocks * 2, 2, device=device)]
                + means_t[torch.arange(1, num_blocks * 2, 2, device=device)]
            ) / 2.0
        print("new block number: ", num_blocks)
        statistic, p_value = kstest(
            means_t.cpu(), "norm", args=(means_t.mean().item(), means_t.std().item())
        )
        print(f"K-S test: statistic {statistic}, p-value {p_value}.")
        mean_combined = torch.mean(means_t)
        error_combined = torch.std(means_t) / num_blocks**0.5

        return (
            mean_combined,
            error_combined,
            # bin_edges,
            # histr / num_blocks,
            # histr_weight / num_blocks,
            partition_z,
        )

    @torch.no_grad()
    def loss_block(self, num_blocks, partition_z=1.0):
        num_samples = self.p.batchsize

        loss = torch.tensor(0.0, device=self.p.samples.device)
        for _ in range(num_blocks):
            self.p.samples, self.p.log_q = self.q0(num_samples)
            for flow in self.flows:
                self.p.samples, log_det = flow(self.p.samples)
                self.p.log_q -= log_det
            self.p.val = self.p.prob(self.p.samples)

            prob_abs = torch.abs(self.p.val)
            log_p = torch.log(torch.clamp(prob_abs, min=1e-16))
            loss += torch.mean(
                prob_abs
                / torch.exp(self.p.log_q)
                / partition_z
                * (log_p - self.p.log_q - torch.log(partition_z))
            )
        return loss / num_blocks

    @torch.no_grad()
    def histogram(self, extvar_dim, bins, range=(0.0, 1.0), has_weight=True):
        """Plots histogram of samples from flow-based approximate distribution

        Args:
          extvar_dim: Dimension of variable to plot histogram for
          bins: int or 1D Tensor. If int, defines the number of equal-width bins. If tensor, defines the sequence of bin edges including the rightmost edge.
          range: Range of the bins.
          has_weight: Flag whether to use weights for histogram. If True, weights are proportional to the probability of each sample. If False, weights are all equal.
        """
        num_samples = self.p.batchsize
        z, log_q = self.sample(num_samples)
        weights = self.p.prob(z) / torch.abs(self.p.prob(z))

        z = self.p.samples.detach().cpu()
        weights = weights.detach().cpu()

        if has_weight:
            histr, bins = torch.histogram(
                z[:, extvar_dim], bins=bins, range=range, weight=weights, density=True
            )
        else:
            histr, bins = torch.histogram(
                z[:, extvar_dim], bins=bins, range=range, density=True
            )

        return histr, bins

    @torch.no_grad()
    def mcmc_sample(self, steps=1, init=False, alpha=0.1):
        batch_size = self.p.batchsize
        device = self.p.samples.device
        vars_shape = self.p.samples.shape
        epsilon = 1e-16  # Small value to ensure numerical stability
        if init:  # Initialize chains
            self.p.samples[:], self.p.log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                self.p.samples[:], log_det = flow(self.p.samples)
                self.p.log_q -= log_det

        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_log_q = torch.empty(batch_size, device=device)

        current_weight = alpha * torch.exp(self.p.log_q) + (1 - alpha) * torch.abs(
            self.p.prob(self.p.samples)
        )
        new_weight = torch.empty(batch_size, device=device)
        acceptance_probs = torch.empty(batch_size, device=device)
        accept = torch.empty(batch_size, device=device, dtype=torch.bool)

        for _ in range(steps):
            # Propose new samples using the normalizing flow
            proposed_samples[:], proposed_log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                proposed_samples[:], proposed_log_det = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_weight[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                self.p.prob(proposed_samples)
            )
            current_weight[:] = torch.clamp(current_weight, min=epsilon)
            new_weight[:] = torch.clamp(new_weight, min=epsilon)
            # Compute acceptance probabilities
            acceptance_probs[:] = (
                new_weight / current_weight * torch.exp(self.p.log_q - proposed_log_q)
            )  # Pi(x') / Pi(x) * q(x) / q(x')

            # Accept or reject the proposals
            accept[:] = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_weight = torch.where(accept, new_weight, current_weight)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
        return self.p.samples

    @torch.no_grad()
    def mcmc_integration(
        self,
        num_blocks=100,
        len_chain=1000,
        burn_in=None,
        thinning=1,
        alpha=1.0,
        step_size=0.0,
        norm_std=0.2,
    ):
        """
        Perform MCMC integration using batch processing. Using the Metropolis-Hastings algorithm to sample the distribution:
        Pi(x) = alpha * q(x) + (1 - alpha) * p(x),
        where q(x) is the learned distribution by the normalizing flow, and p(x) is the target distribution.

        Args:
            num_blocks: Number of blocks to divide the batch into.
            len_chain: Number of samples to draw.
            burn_in: Number of initial samples to discard.
            thinning: Interval to thin the chain.
            alpha: Annealing parameter.
            step_size: random walk step size.

        Returns:
            mean, error: Mean and standard variance of the integrated samples.
        """
        epsilon = 1e-16  # Small value to ensure numerical stability
        batch_size = self.p.batchsize
        device = self.p.samples.device
        vars_shape = self.p.samples.shape
        if burn_in is None:
            burn_in = len_chain // 4

        # Initialize chains
        current_z, self.p.log_q[:] = self.q0(batch_size)
        self.p.samples[:] = current_z
        for flow in self.flows:
            self.p.samples[:], log_det = flow(self.p.samples)
            self.p.log_q -= log_det
        proposed_z = torch.empty(vars_shape, device=device)
        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_log_q = torch.empty(batch_size, device=device)

        current_weight = alpha * torch.exp(self.p.log_q) + (1 - alpha) * torch.abs(
            self.p.prob(self.p.samples)
        )  # Pi(x) = alpha * q(x) + (1 - alpha) * p(x)
        torch.clamp(current_weight, min=epsilon, out=current_weight)
        new_weight = torch.empty(batch_size, device=device)
        acceptance_probs = torch.empty(batch_size, device=device)
        accept = torch.empty(batch_size, device=device, dtype=torch.bool)

        for i in range(burn_in):
            # Propose new samples using the normalizing flow
            proposed_z[:], proposed_log_q[:] = self.q0(batch_size)
            # proposed_z[:] = (current_z + (proposed_z - 0.5) * step_size) % 1.0
            proposed_z[:] = (
                current_z
                + torch.normal(step_size, norm_std, size=vars_shape, device=device)
            ) % 1.0
            proposed_samples[:] = proposed_z
            for flow in self.flows:
                proposed_samples[:], proposed_log_det = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_weight[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                self.p.prob(proposed_samples)
            )
            torch.clamp(new_weight, min=epsilon, out=new_weight)
            # Compute acceptance probabilities
            acceptance_probs[:] = (
                new_weight / current_weight * torch.exp(self.p.log_q - proposed_log_q)
            )  # Pi(x') / Pi(x) * q(x) / q(x')

            # Accept or reject the proposals
            accept[:] = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_z = torch.where(accept.unsqueeze(1), proposed_z, current_z)
            current_weight = torch.where(accept, new_weight, current_weight)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
            # self.p.log_q[accept] = proposed_log_q[accept]

            # if i % 100 == 0 and i > 0:
            #     accept_rate = accept.sum().item() / batch_size
            #     if accept_rate < 0.4:
            #         step_size *= 0.9
            #     else:
            #         step_size *= 1.1

        self.p.val = self.p.prob(self.p.samples)
        new_prob = torch.empty_like(self.p.val)

        ref_values = torch.zeros(num_blocks, device=device)
        values = torch.zeros(num_blocks, device=device)
        abs_values = torch.zeros(num_blocks, device=device)
        var_p = torch.zeros(num_blocks, device=device)
        var_q = torch.zeros(num_blocks, device=device)
        cov_pq = torch.zeros(num_blocks, device=device)
        block_size = batch_size // num_blocks
        num_measure = 0
        for i in range(len_chain):
            # Propose new samples using the normalizing flow
            proposed_z[:], proposed_log_q[:] = self.q0(batch_size)
            # proposed_z[:] = (current_z + (proposed_z - 0.5) * step_size) % 1.0
            proposed_z[:] = (
                current_z
                + torch.normal(step_size, norm_std, size=vars_shape, device=device)
            ) % 1.0
            proposed_samples[:] = proposed_z
            for flow in self.flows:
                proposed_samples[:], proposed_log_det = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_prob[:] = self.p.prob(proposed_samples)
            new_weight[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                new_prob
            )
            torch.clamp(new_weight, min=epsilon, out=new_weight)
            # Compute acceptance probabilities
            acceptance_probs[:] = (
                new_weight / current_weight * torch.exp(self.p.log_q - proposed_log_q)
            )  # Pi(x') / Pi(x) * q(x) / q(x')

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            if i % 400 == 0:
                print("acceptance rate: ", accept.sum().item() / batch_size)

            current_z = torch.where(accept.unsqueeze(1), proposed_z, current_z)
            self.p.val = torch.where(accept, new_prob, self.p.val)
            current_weight = torch.where(accept, new_weight, current_weight)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
            # self.p.log_q[accept] = proposed_log_q[accept]

            # Measurement
            if i % thinning == 0:
                num_measure += 1
                for j in range(num_blocks):
                    start = j * block_size
                    end = (j + 1) * block_size
                    values[j] += torch.mean(
                        self.p.val[start:end] / current_weight[start:end]
                    )
                    ref_values[j] += torch.mean(
                        torch.exp(self.p.log_q[start:end]) / current_weight[start:end]
                    )
                    abs_values[j] += torch.mean(
                        torch.abs(self.p.val[start:end] / current_weight[start:end])
                    )

                    var_p[j] += torch.mean(
                        (self.p.val[start:end] / current_weight[start:end]) ** 2
                    )
                    var_q[j] += torch.mean(
                        (torch.exp(self.p.log_q[start:end]) / current_weight[start:end])
                        ** 2
                    )
                    cov_pq[j] += torch.mean(
                        self.p.val[start:end]
                        * torch.exp(self.p.log_q[start:end])
                        / current_weight[start:end] ** 2
                    )
        values /= num_measure
        ref_values /= num_measure
        abs_values /= num_measure
        var_p /= num_measure
        var_q /= num_measure
        cov_pq /= num_measure

        while (
            kstest(
                values.cpu(), "norm", args=(values.mean().item(), values.std().item())
            )[1]
            < 0.05
            or kstest(
                ref_values.cpu(),
                "norm",
                args=(ref_values.mean().item(), ref_values.std().item()),
            )[1]
            < 0.05
        ):
            print("correlation too high, merge blocks")
            if num_blocks <= 64:
                warn(
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            even_idx = torch.arange(0, num_blocks * 2, 2, device=device)
            odd_idx = torch.arange(1, num_blocks * 2, 2, device=device)
            values = (values[even_idx] + values[odd_idx]) / 2.0
            abs_values = (abs_values[even_idx] + abs_values[odd_idx]) / 2.0
            ref_values = (ref_values[even_idx] + ref_values[odd_idx]) / 2.0
            var_p = (var_p[even_idx] + var_p[odd_idx]) / 2.0
            var_q = (var_q[even_idx] + var_q[odd_idx]) / 2.0
            cov_pq = (cov_pq[even_idx] + cov_pq[odd_idx]) / 2.0
        print("new block number: ", num_blocks)

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

        ratio_mean = torch.mean(values) / torch.mean(ref_values)
        abs_val_mean = torch.mean(abs_values) / torch.mean(ref_values)

        cov_matrix = torch.cov(torch.stack((values, ref_values)))
        print("covariance matrix: ", cov_matrix)
        ratio_var = (
            cov_matrix[0, 0]
            - 2 * ratio_mean * cov_matrix[0, 1]
            + ratio_mean**2 * cov_matrix[1, 1]
        ) / torch.mean(ref_values) ** 2
        ratio_err = (ratio_var / batch_size) ** 0.5

        values /= ref_values
        # print("correlation of combined values: ", calculate_correlation(values))
        _mean = torch.mean(values)
        _std = torch.std(values)
        error = _std / num_blocks**0.5

        print("old result: {:.5e} +- {:.5e}".format(_mean.item(), error.item()))

        statistic, p_value = kstest(
            values.cpu(), "norm", args=(_mean.item(), _std.item())
        )
        print(f"K-S test of ratio values: statistic {statistic}, p-value {p_value}")

        abs_values /= ref_values
        err_absval = torch.std(abs_values) / num_blocks**0.5
        print(
            "|f(x)| Integration results: {:.5e} +/- {:.5e}".format(
                abs_val_mean.item(), err_absval.item()
            )
        )

        err_var_p = torch.std(var_p) / num_blocks**0.5
        err_var_q = torch.std(var_q) / num_blocks**0.5
        err_cov_pq = torch.std(cov_pq) / num_blocks**0.5
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
        print(
            "covariance of pq: {:.5e} +/- {:.5e}".format(
                cov_pq.mean().item(), err_cov_pq.item()
            )
        )

        I_alpha = alpha + (1 - alpha) * abs_val_mean.item()
        print("I_alpha: ", I_alpha)

        var_pq_mean = I_alpha**2 * (
            var_p.mean() + var_q.mean() * ratio_mean**2 - 2 * ratio_mean * cov_pq.mean()
        )
        var_pq = (alpha + (1 - alpha) * abs_values) ** 2 * (
            var_p + var_q * values**2 - 2 * values * cov_pq
        )
        var_pq_err = torch.std(var_pq) / num_blocks**0.5
        print(
            "Composite variance: {:.5e} +/- {:.5e}  ".format(
                var_pq_mean.item(), var_pq_err.item()
            ),
            var_pq.mean(),
        )
        print(
            "theoretical estimated error : {:.5e}".format(
                (var_pq_mean / num_measure).item() ** 0.5
            )
        )

        return ratio_mean, ratio_err

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class ConditionalNormalizingFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """

    def forward(self, z, context=None):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context=None):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        return x, log_det

    def sample(self, num_samples=1, context=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context=None):
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def forward_kld(self, x, context=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, context=None, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)


class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model, providing the
    class to be conditioned on only to the base distribution,
    as done e.g. in [Glow](https://arxiv.org/abs/1807.03039)
    """

    def __init__(self, q0, flows):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x, y):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return -torch.mean(log_q)

    def sample(self, num_samples=1, y=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, y)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
         param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, q0, flows, merges, transform=None, class_cond=True):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of list of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        self.q0 = nn.ModuleList(q0)
        self.num_levels = len(self.q0)
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.transform = transform
        self.class_cond = class_cond

    def forward_kld(self, x, y=None):
        """Estimates forward KL divergence, see see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of targets, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

    def forward(self, x, y=None):
        """Get negative log-likelihood for maximum likelihood training

        Args:
          x: Batch of data
          y: Batch of targets, if applicable

        Returns:
            Negative log-likelihood of the batch
        """
        return -self.log_prob(x, y)

    def forward_and_log_det(self, z):
        """Get observed variable x from list of latent variables z

        Args:
            z: List of latent variables

        Returns:
            Observed variable x, log determinant of Jacobian
        """
        log_det = torch.zeros(len(z[0]), dtype=z[0].dtype, device=z.device)
        for i in range(len(self.q0)):
            if i == 0:
                z_ = z[0]
            else:
                z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                log_det += log_det_
            for flow in self.flows[i]:
                z_, log_det_ = flow(z_)
                log_det += log_det_
        if self.transform is not None:
            z_, log_det_ = self.transform(z_)
            log_det += log_det_
        return z_, log_det

    def inverse_and_log_det(self, x):
        """Get latent variable z from observed variable x

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        if self.transform is not None:
            x, log_det_ = self.transform.inverse(x)
            log_det += log_det_
        z = [None] * len(self.q0)
        for i in range(len(self.q0) - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_
        return z, log_det

    def sample(self, num_samples=1, y=None, temperature=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None
          temperature: Temperature parameter for temp annealed sampling

        Returns:
          Samples, log probability
        """
        if temperature is not None:
            self.set_temperature(temperature)
        for i in range(len(self.q0)):
            if self.class_cond:
                z_, log_q_ = self.q0[i](num_samples, y)
            else:
                z_, log_q_ = self.q0[i](num_samples)
            if i == 0:
                log_q = log_q_
                z = z_
            else:
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
            for flow in self.flows[i]:
                z, log_det = flow(z)
                log_q -= log_det
        if self.transform is not None:
            z, log_det = self.transform(z)
            log_q -= log_det
        if temperature is not None:
            self.reset_temperature()
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det
        for i in range(len(self.q0) - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                log_q += log_det
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
            if self.class_cond:
                log_q += self.q0[i].log_prob(z_, y)
            else:
                log_q += self.q0[i].log_prob(z_)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def set_temperature(self, temperature):
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, "temperature"):
                q0.temperature = temperature
            else:
                raise NotImplementedError(
                    "One base function does not "
                    "support temperature annealed sampling"
                )

    def reset_temperature(self):
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)


class NormalizingFlowVAE(nn.Module):
    """
    VAE using normalizing flows to express approximate distribution
    """

    def __init__(self, prior, q0=distributions.Dirac(), flows=None, decoder=None):
        """Constructor of normalizing flow model

        Args:
          prior: Prior distribution of te VAE, i.e. Gaussian
          decoder: Optional decoder
          flows: Flows to transform output of base encoder
          q0: Base Encoder
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.flows = nn.ModuleList(flows)
        self.q0 = q0

    def forward(self, x, num_samples=1):
        """Takes data batch, samples num_samples for each data point from base distribution

        Args:
          x: data batch
          num_samples: number of samples to draw for each data point

        Returns:
          latent variables for each batch and sample, log_q, and log_p
        """
        z, log_q = self.q0(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior.log_prob(z)
        if self.decoder is not None:
            log_p += self.decoder.log_prob(x, z)
        # Separate batch and sample dimension again
        z = z.view(-1, num_samples, *z.size()[1:])
        log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        log_p = log_p.view(-1, num_samples, *log_p.size()[1:])
        return z, log_q, log_p
