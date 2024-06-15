import os
import pandas as pd
import numpy as np
import torch
import re
import normflows as nf

from realNVP_integrator import generate_model, train_model, train_model_annealing
from functools import partial
from nsf_multigpu import *
from funcs_sigma import *
import time

from matplotlib import pyplot as plt
import tracemalloc
# from torch.utils.viz._cycles import warn_tensor_cycles

# warn_tensor_cycles()

root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
# from absl import app, flags
num_loops = [2, 6, 15, 39, 111, 448]
num_roots = [1, 2, 3, 4, 5, 6]
traced_batchsize = [1000, 10000, 20000, 50000, 100000]
order = 1
beta = 1.0
batch_size = 10000
num_layers = 4
accum_iter = 10

init_lr = 8e-3
Nepochs = 100
Nblocks = 100

is_save = True
is_annealing = False
has_proposal_rnvp = False
# has_proposal_rnvp = True
multi_gpu = False

model_state_dict_path = "rnvp_o{0}_beta{1}_l1c32b8_state.pt".format(order, beta)

print(
    "beta:", beta, "order:", order, "batchsize:", batch_size, "num_layers:", num_layers
)


def _StringtoIntVector(s):
    pattern = r"[-+]?\d+"
    return [int(match) for match in re.findall(pattern, s)]


def chemical_potential(beta):
    if beta == 0.1:
        _mu = -37.301528674058275
    elif beta == 1.0:
        _mu = -0.021460754987022185
    elif beta == 2.0:
        _mu = 0.7431120842589388
    elif beta == 4.0:
        _mu = 0.9426157552012961
    elif beta == 8.0:
        _mu = 0.986801399943294
    elif beta == 10.0:
        _mu = 0.9916412363704453
    elif beta == 16.0:
        _mu = 0.9967680535828609
    elif beta == 32.0:
        _mu = 0.9991956396090637
    elif beta == 64.0:
        _mu = 0.9997991296749593
    elif beta == 128.0:
        _mu = 0.9999497960580543
    else:
        _mu = 1.0
    return _mu


class FeynmanDiagram(nf.distributions.Target):
    @torch.no_grad()
    def __init__(self, order, beta, loopBasis, leafstates, leafvalues, batchsize):
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
        self.dim = 3

        # Derived constants
        self.register_buffer("kF", (9 * pi / (2 * self.spin)) ** (1 / 3) / self.rs)
        self.register_buffer("EF", self.kF**2 / (2 * self.me))
        self.register_buffer("beta", beta / self.EF)
        self.register_buffer("mu", chemical_potential(beta) * self.EF)
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
        self.register_buffer("root", torch.ones([self.batchsize, num_roots[order - 1]]))

        self.register_buffer("samples", torch.zeros([self.batchsize, self.ndims]))
        self.register_buffer("log_q", torch.zeros([self.batchsize]))
        self.register_buffer("log_det", torch.zeros([self.batchsize]))
        self.register_buffer("val", torch.zeros([self.batchsize]))

        # Convention of variables: first totalTauNum - 1 variables are tau. The rest are momentums in shperical coordinate.
        self.p[:, 0, 0] += self.kF
        self.extk = self.kF
        self.extn = 0
        self.targetval = 4.0

        if batch_size in traced_batchsize:
            Sigma_module = torch.jit.load(
                os.path.join(root_dir, f"traced_Sigma_{batch_size:.0e}.pt")
            )
            if order == 1:
                self.eval_graph = Sigma_module.func100
            elif order == 2:
                self.eval_graph = Sigma_module.func200
            elif order == 3:
                self.eval_graph = Sigma_module.func300
            elif order == 4:
                self.eval_graph = Sigma_module.func400
            elif order == 5:
                self.eval_graph = Sigma_module.func500
            elif order == 6:
                self.eval_graph = Sigma_module.func600
            else:
                raise ValueError("Invalid order")
        else:
            if order == 1:
                self.eval_graph = torch.jit.script(eval_graph100)
            elif order == 2:
                self.eval_graph = torch.jit.script(eval_graph200)
            elif order == 3:
                self.eval_graph = torch.jit.script(eval_graph300)
            elif order == 4:
                self.eval_graph = torch.jit.script(eval_graph400)
            elif order == 5:
                self.eval_graph = torch.jit.script(eval_graph500)
            elif order == 6:
                self.eval_graph = torch.jit.script(eval_graph600)
            else:
                raise ValueError("Invalid order")

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

        self.factor[:] = torch.prod(
            (p_rescale * self.maxK) ** 2 * torch.sin(theta), dim=1
        )
        self.p[:, 0, 1:] = p_rescale * self.maxK * torch.sin(theta)
        self.p[:, 1, 1:] = self.p[:, 0, 1:]
        self.p[:, 0, 1:] *= torch.cos(phi)
        self.p[:, 1, 1:] *= torch.sin(phi)
        self.p[:, 2, 1:] = p_rescale * self.maxK * torch.cos(theta)

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
        self.kq2[:] = torch.sum(kq * kq, dim=1)
        self.dispersion[:] = self.kq2 / (2 * self.me) - self.mu
        self.kernelFermiT()
        # Calculate bosonic leaves
        self.invK[:] = 1.0 / (self.kq2 + self.mass2)
        self.leaf_bose[:] = ((self.e0**2 / self.eps0) * self.invK) * (
            self.mass2 * self.invK
        ) ** self.lforders[1]
        # self.leafvalues[self.isfermi] = self.leaf_fermi[self.isfermi]
        # self.leafvalues[self.isbose] = self.leaf_bose[self.isbose]
        self.leafvalues = torch.where(self.isfermi, self.leaf_fermi, self.leafvalues)
        self.leafvalues = torch.where(self.isbose, self.leaf_bose, self.leafvalues)

    @torch.no_grad()
    def prob(self, var):
        self._evalleaf(var)
        # self.eval_graph(self.root, self.leafvalues)
        self.root[:] = self.eval_graph(self.leafvalues)
        return self.root.sum(dim=1) * (
            self.factor
            * (self.maxK * 2 * np.pi**2) ** (self.innerLoopNum)
            * (self.beta) ** (self.totalTauNum - 1)
            / (2 * np.pi) ** (self.dim * self.innerLoopNum)
        )

    @torch.no_grad()
    def log_prob(self, var):
        return torch.log(torch.clamp(self.prob(var), min=1e-10))

    @torch.no_grad()
    def sample(self, steps: int = 10):
        for i in range(steps):
            proposed_samples = torch.rand(
                self.batchsize, self.ndims, device=self.samples.device
            )
            acceptance_probs = torch.clamp(
                torch.exp(
                    self.log_prob(proposed_samples) - self.log_prob(self.samples)
                ),
                max=1,
            )
            accept = (
                torch.rand(self.batchsize, device=self.samples.device)
                <= acceptance_probs
            )
            self.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.samples
            )

        return self.samples


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

    diagram = FeynmanDiagram(
        order, beta, loopBasis, leafstates[0], leafvalues[0], batch_size
    )

    rnvp = generate_model(diagram, num_layers)
    for name, param in rnvp.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    epochs = Nepochs
    blocks = Nblocks

    # torch.cuda.memory._record_memory_history()
    # tracemalloc.start()
    start_time = time.time()
    with torch.no_grad():
        mean, err, partition_z = rnvp.integrate_block(blocks)
    print("Initial integration time: {:.3f}s".format(time.time() - start_time))
    loss = rnvp.loss_block(20, partition_z)
    print("Initial loss: ", loss)

    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * diagram.batchsize, mean, err, rnvp.p.targetval
        )
    )
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # print("[ Top 20 ]")
    # for stat in top_stats[:20]:
    #     print(stat)
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    if has_proposal_rnvp:
        # proposal_model = torch.load("rnvp_o{0}_beta{1}.pt".format(order, beta))
        diagram = FeynmanDiagram(
            order, beta, loopBasis, leafstates[0], leafvalues[0], batch_size
        )
        proposal_model = generate_model(diagram)
        state_dict = torch.load(model_state_dict_path)
        # proposal_model.load_state_dict(state_dict)

        pmodel_state_dict = proposal_model.state_dict()
        partial_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in pmodel_state_dict and "p." not in k
        }
        pmodel_state_dict.update(partial_state_dict)

        proposal_model.load_state_dict(pmodel_state_dict)
        proposal_model.eval()

        start_time = time.time()
        if multi_gpu:
            trainfn = partial(
                train_model_parallel,
                rnvp=rnvp,
                max_iter=epochs,
                num_samples=diagram.batchsize,
                accum_iter=accum_iter,
                has_scheduler=True,
                proposal_model=proposal_model,
                save_checkpoint=True,
            )
            run_train(trainfn, world_size)
        else:
            train_model(
                rnvp,
                epochs,
                diagram.batchsize,
                proposal_model=proposal_model,
                accum_iter=accum_iter,
                init_lr=init_lr,
            )
    else:
        start_time = time.time()
        if multi_gpu:
            trainfn = partial(
                train_model_parallel,
                rnvp=rnvp,
                max_iter=epochs,
                num_samples=diagram.batchsize,
                accum_iter=accum_iter,
                has_scheduler=True,
                proposal_model=None,
                save_checkpoint=True,
            )
            run_train(trainfn, world_size)
        else:
            print("initial learning rate: ", init_lr)
            if is_annealing:
                train_model_annealing(
                    rnvp, epochs, diagram.batchsize, accum_iter, init_lr
                )
            else:
                train_model(rnvp, epochs, diagram.batchsize, accum_iter, init_lr)

    print("Training time: {:.3f}s".format(time.time() - start_time))

    print("Start computing integration...")
    start_time = time.time()
    num_hist_bins = 25
    if multi_gpu:
        rnvp = torch.load("checkpoint.pt")

    if is_save:
        # rnvp.save(
        #     "rnvp_o{0}_beta{1}_l{2}c{3}b{4}.pt".format(
        #         order, beta, num_blocks, num_hidden_channels, num_bins
        #     )
        # )
        torch.save(
            rnvp.state_dict(),
            "realnvp_o{0}_beta{1}_l{2}_state.pt".format(order, beta, num_layers),
        )

    with torch.no_grad():
        mean, err, partition_z = rnvp.integrate_block(blocks, num_hist_bins)
    print("Final integration time: {:.3f}s".format(time.time() - start_time))
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * diagram.batchsize, mean, err, rnvp.p.targetval
        )
    )

    start_time = time.time()
    mean_mcmc, err_mcmc = rnvp.mcmc_integration(
        num_blocks=blocks, len_chain=blocks, thinning=1, alpha=0.1
    )
    print("MCMC integration time: {:.3f}s".format(time.time() - start_time))
    print(
        "MCMC result with {:d} samples is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * diagram.batchsize, mean_mcmc, err_mcmc, rnvp.p.targetval
        )
    )

    loss = rnvp.loss_block(100, partition_z)
    print("Final loss: ", loss)
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


if __name__ == "__main__":
    main(1)
    # retrain(1)
    # app.run(main)
