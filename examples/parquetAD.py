import os
import pandas as pd
import numpy as np
import torch
import re
import normflows as nf
from nsf_integrator import generate_model, train_model
from funcs_sigma import *
import vegas

from matplotlib import pyplot as plt
import tracemalloc
# from torch.utils.viz._cycles import warn_tensor_cycles

# warn_tensor_cycles()

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
# include(os.path.join(root_dir, f"func_sigma_o100.py"))
# from absl import app, flags
num_loops = [2, 6, 15, 39, 111, 448]
order = 3
beta = 10.0
batch_size = 100000


def _StringtoIntVector(s):
    pattern = r"[-+]?\d+"
    return [int(match) for match in re.findall(pattern, s)]


class FeynmanDiagram(nf.distributions.Target):
    @torch.no_grad()
    def __init__(self, order, loopBasis, leafstates, leafvalues, batchsize):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        # Unpack leafstates for clarity
        lftype, lforders, leaf_tau_i, leaf_tau_o, leafMomIdx = leafstates

        # Register buffers for leaf state information
        self.register_buffer("lftype", lftype)
        self.register_buffer("lforders", lforders)
        self.register_buffer("leaf_tau_i", leaf_tau_i)
        self.register_buffer("leaf_tau_o", leaf_tau_o)
        self.register_buffer("leafMomIdx", leafMomIdx)

        # Physical constants setup
        pi = np.pi
        self.register_buffer("eps0", torch.tensor(1 / (4 * pi)))
        self.register_buffer("e0", torch.sqrt(torch.tensor(2.0)))
        self.register_buffer("mass2", torch.tensor(0.5))
        self.register_buffer("me", torch.tensor(0.5))
        self.register_buffer("spin", torch.tensor(2.0))
        self.register_buffer("rs", torch.tensor(2.0))
        # self.register_buffer("dim", torch.tensor(3))
        self.dim = 3

        # Derived constants
        self.register_buffer("kF", (9 * pi / (2 * self.spin)) ** (1 / 3) / self.rs)
        self.register_buffer("EF", self.kF**2 / (2 * self.me))
        self.register_buffer("mu", self.EF)
        self.register_buffer("beta", beta / self.EF)
        if beta == 1.0:
            _mu = -0.021460754987022185
        elif beta == 10.0:
            _mu = 0.9916412363704453
        else:
            _mu = 1.0
        self.register_buffer("mu", _mu * self.EF)
        self.register_buffer("maxK", 10 * self.kF)

        print(
            "param:",
            self.dim,
            self.beta,
            self.me,
            self.mass2,
            self.mu,
            self.e0,
            self.eps0,
        )

        self.batchsize = batchsize
        self.innerLoopNum = order
        self.totalTauNum = order
        self.ndims = self.innerLoopNum * self.dim + self.totalTauNum - 1

        self.register_buffer(
            "loopBasis", loopBasis
        )  # size=(self.innerLoopNum + 1, loopBasis.shape[1])
        self.register_buffer(
            "loops", torch.empty((self.batchsize, self.dim, loopBasis.shape[1]))
        )
        self.register_buffer(
            "leafvalues",
            torch.broadcast_to(leafvalues, (self.batchsize, leafvalues.shape[0])),
        )
        self.register_buffer(
            "p", torch.zeros([self.batchsize, self.dim, self.innerLoopNum + 1])
        )
        self.register_buffer("tau", torch.zeros_like(self.leafvalues))
        self.register_buffer("kq2", torch.zeros_like(self.leafvalues))
        self.register_buffer("invK", torch.zeros_like(self.leafvalues))
        self.register_buffer("dispersion", torch.zeros_like(self.leafvalues))
        self.register_buffer(
            "isfermi", torch.full_like(self.leafvalues, True, dtype=torch.bool)
        )
        self.register_buffer(
            "isbose", torch.full_like(self.leafvalues, True, dtype=torch.bool)
        )
        self.register_buffer("leaf_fermi", torch.zeros_like(self.leafvalues))
        self.register_buffer("leaf_bose", torch.zeros_like(self.leafvalues))
        self.register_buffer("factor", torch.ones([self.batchsize]))
        self.register_buffer("root", torch.ones([self.batchsize]))

        self.register_buffer("samples", torch.zeros([self.batchsize, self.ndims]))
        self.register_buffer("log_q", torch.zeros([self.batchsize]))
        self.register_buffer("log_det", torch.zeros([self.batchsize]))
        self.register_buffer("val", torch.zeros([self.batchsize]))

        # Convention of variables: first totalTauNum - 1 variables are tau. The rest are momentums in shperical coordinate.
        self.p[:, 0, 0] += self.kF
        self.extk = self.kF
        self.extn = 0
        self.targetval = 4.0

    @torch.no_grad()
    def kernelFermiT(self):
        sign = torch.where(self.tau > 0, 1.0, -1.0)

        a = torch.where(
            self.tau > 0,
            torch.where(self.dispersion > 0, -self.tau, self.beta - self.tau),
            torch.where(self.dispersion > 0, -(self.beta + self.tau), -self.tau),
        )
        b = torch.where(self.dispersion > 0, -self.beta, self.beta)

        # Use torch operations to ensure calculations are done on GPU if tensors are on GPU
        self.leaf_fermi[:] = sign * torch.exp(self.dispersion * a)
        self.leaf_fermi /= 1 + torch.exp(self.dispersion * b)

    @torch.no_grad()
    def extract_mom(self, var):
        p_rescale = var[
            :, self.totalTauNum - 1 : self.totalTauNum - 1 + self.innerLoopNum
        ]
        theta = (
            var[
                :,
                self.totalTauNum - 1 + self.innerLoopNum : self.totalTauNum
                - 1
                + 2 * self.innerLoopNum,
            ]
            * np.pi
        )
        phi = (
            var[
                :,
                self.totalTauNum - 1 + 2 * self.innerLoopNum : self.totalTauNum
                - 1
                + 3 * self.innerLoopNum,
            ]
            * 2
            * np.pi
        )
        # print((p_rescale / (1+1e-6 - p_rescale)**2)**2 * torch.sin(theta))
        # self.factor = torch.prod((p_rescale / (1+1e-6 - p_rescale)**2)**2 * torch.sin(theta), dim = 1)
        # print("factor:", self.factor)
        # p_rescale /= (1.0 + 1e-10 - p_rescale)

        p_rescale *= self.maxK
        self.factor[:] = torch.prod(p_rescale**2 * torch.sin(theta), dim=1)
        self.p[:, 0, 1:] = p_rescale * torch.sin(theta)
        self.p[:, 1, 1:] = self.p[:, 0, 1:]
        self.p[:, 0, 1:] *= torch.cos(phi)
        self.p[:, 1, 1:] *= torch.sin(phi)
        self.p[:, 2, 1:] = p_rescale * torch.cos(theta)

    @torch.no_grad()
    def _evalleaf(self, var):
        self.isfermi[:] = self.lftype == 1
        self.isbose[:] = self.lftype == 2
        # update momentum
        self.extract_mom(var)  # varK should have shape [batchsize, dim, innerLoopMom]
        torch.matmul(self.p, self.loopBasis, out=self.loops)

        self.tau[:] = torch.where(
            self.leaf_tau_o == 0, 0.0, var[:, self.leaf_tau_o - 1]
        )
        self.tau -= torch.where(self.leaf_tau_i == 0, 0.0, var[:, self.leaf_tau_i - 1])
        self.tau *= self.beta

        kq = self.loops[:, :, self.leafMomIdx]
        # print("test1:", lftype, self.p, self.loops.shape, self.loopBasis[:, :2], kq)
        self.kq2[:] = torch.sum(kq * kq, dim=1)
        self.dispersion[:] = self.kq2 / (2 * self.me) - self.mu
        # print("test2:",self.loops.shape, eps.shape, tau.shape, leaf_tau_i.shape)
        # order = lforders[0]
        self.kernelFermiT()
        # print("var", kq2, self.mu, kernelFermiT(tau, eps, self.beta), tau, eps, self.beta)
        # Calculate bosonic leaves
        self.invK[:] = 1.0 / (self.kq2 + self.mass2)
        self.leaf_bose[:] = ((self.e0**2 / self.eps0) * self.invK) * (
            self.mass2 * self.invK
        ) ** self.lforders[1]
        # print("Before update:", self.leafvalues)
        # print(self.isfermi)
        # print(self.isbose)
        # print(self.leaf_fermi)
        # print(self.leaf_bose)
        # self.leafvalues[self.isfermi] = self.leaf_fermi[self.isfermi]
        # self.leafvalues[self.isbose] = self.leaf_bose[self.isbose]
        # print("After update:", self.leafvalues)
        # exit(-1)
        self.leafvalues = torch.where(self.isfermi, self.leaf_fermi, self.leafvalues)
        self.leafvalues = torch.where(self.isbose, self.leaf_bose, self.leafvalues)

    @torch.no_grad()
    def prob(self, var):
        self._evalleaf(var)
        self.root[:] = (
            # torch.stack(func_sigma_o200.graphfunc(self.leafvalues), dim=0).sum(dim=0)
            torch.stack(func_sigma_o300.graphfunc(self.leafvalues), dim=0).sum(dim=0)
            * self.factor
            * (self.maxK * 2 * np.pi**2) ** (self.innerLoopNum)
            * (self.beta) ** (self.totalTauNum - 1)
            / (2 * np.pi) ** (self.dim * self.innerLoopNum)
        )
        return self.root

    @torch.no_grad()
    def log_prob(self, var):
        self.prob(var)
        return torch.log(torch.clamp(torch.abs(self.root), min=1e-10))


def load_leaf_info(root_dir, name, key_str):
    df = pd.read_csv(os.path.join(root_dir, f"leafinfo_{name}_{key_str}.csv"))
    with torch.no_grad():
        leaftypes = torch.tensor(df.iloc[:, 1].to_numpy())
        leaforders = torch.tensor([_StringtoIntVector(x) for x in df.iloc[:, 2]]).T
        inTau_idx = torch.tensor(df.iloc[:, 3].to_numpy() - 1)
        outTau_idx = torch.tensor(df.iloc[:, 4].to_numpy() - 1)
        loop_idx = torch.tensor(df.iloc[:, 5].to_numpy() - 1)
        leafvalues = torch.tensor(df.iloc[:, 0].to_numpy())
    return (leaftypes, leaforders, inTau_idx, outTau_idx, loop_idx), leafvalues


def main(argv):
    del argv

    partition = [(order, 0, 0)]
    name = "sigma"
    df = pd.read_csv(os.path.join(root_dir, f"loopBasis_{name}_maxOrder6.csv"))
    with torch.no_grad():
        # loopBasis = torch.tensor([df[col].iloc[:maxMomNum].tolist() for col in df.columns[:num_loops[order-1]]]).T
        loopBasis = torch.Tensor(
            df.iloc[: order + 1, : num_loops[order - 1]].to_numpy()
        )
    leafstates = []
    leafvalues = []

    for key in partition:
        key_str = "".join(map(str, key))
        state, values = load_leaf_info(root_dir, name, key_str)
        leafstates.append(state)
        leafvalues.append(values)

    diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batch_size)

    nfm = generate_model(diagram, num_hidden_channels=32, num_bins=8)
    for name, param in nfm.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    epochs = 100
    blocks = 400

    # torch.cuda.memory._record_memory_history()
    tracemalloc.start()
    with torch.no_grad():
        mean, err, _, _, _ = nfm.integrate_block(blocks)

    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * diagram.batchsize, mean, err, nfm.p.targetval
        )
    )
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    print("[ Top 20 ]")
    for stat in top_stats[:20]:
        print(stat)

    train_model(nfm, epochs, diagram.batchsize)

    num_bins = 25
    with torch.no_grad():
        mean, err, bins, histr, histr_weight = nfm.integrate_block(blocks, num_bins)

    print(bins)
    torch.save(histr, "histogram_o{0}_beta{1}.pt".format(order, beta))
    torch.save(histr_weight, "histogramWeight_o{0}_beta{1}.pt".format(order, beta))

    plt.figure(figsize=(15, 12))
    for ndim in range(diagram.ndims):
        plt.stairs(
            histr_weight[:, ndim].numpy(), bins.numpy(), label="{0} Dim".format(ndim)
        )
    plt.legend()
    plt.savefig("histogram_o{0}_beta{1}.png".format(order, beta))
    # nfm.train()
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * diagram.batchsize, mean, err, nfm.p.targetval
        )
    )
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


if __name__ == "__main__":
    main(1)
    # app.run(main)
