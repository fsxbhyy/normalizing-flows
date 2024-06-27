import torch
import pandas as pd
import time
from parquetAD import FeynmanDiagram, load_leaf_info

# from nsf_integrator import generate_model
from nsf_annealing import generate_model
import os

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 2
beta = 16.0
nfm_batchsize = 20000
batch_size = 2000
# Nblocks = batch_size
Nblocks = 400
len_chain = 2000
therm_steps = len_chain // 2
step_size = 0.01
norm_std = 0.2
mix_rate = 0.1

print(
    f"batchsize {batch_size}, nblocks {Nblocks}, therm_steps {therm_steps}, Gaussian random-walk N({step_size}, {norm_std}^2)"
)

num_hidden_layers = 1
model_state_dict_path = "nfm_o{0}_beta{1}_l{2}c32b8_state1.pt".format(
    order, beta, num_hidden_layers
)
# model_state_dict_path = "nfm_o{0}_beta{1}_state_l2c32b8_anneal.pt".format(order, beta)

partition = [(order, 0, 0)]
name = "sigma"
df = pd.read_csv(os.path.join(root_dir, f"loopBasis_{name}_maxOrder6.csv"))
with torch.no_grad():
    # loopBasis = torch.tensor([df[col].iloc[:maxMomNum].tolist() for col in df.columns[:num_loops[order-1]]]).T
    loopBasis = torch.Tensor(df.iloc[: order + 1, : num_loops[order - 1]].to_numpy())
leafstates = []
leafvalues = []

for key in partition:
    key_str = "".join(map(str, key))
    state, values = load_leaf_info(root_dir, name, key_str)
    leafstates.append(state)
    leafvalues.append(values)


def main(blocks, beta, len_chain, batch_size, nfm_batchsize):
    print(
        f"{blocks} blocks, {len_chain} samples per markov chain, batch size {batch_size}"
    )
    diagram_nfm = FeynmanDiagram(
        order, beta, loopBasis, leafstates[0], leafvalues[0], nfm_batchsize
    )
    diagram = FeynmanDiagram(
        order, beta, loopBasis, leafstates[0], leafvalues[0], batch_size
    )

    print("Loading normalizing-flow model")
    start_time = time.time()
    # nfm = torch.load("nfm_o{0}_beta{1}.pt".format(order, beta))
    # nfm.eval()
    state_dict = torch.load(
        model_state_dict_path, map_location=device
    )  # ["model_state_dict"]
    partial_state_dict = {
        k: v for k, v in state_dict.items() if k in state_dict and "p." not in k
    }
    nfm = generate_model(
        diagram_nfm,
        num_blocks=num_hidden_layers,
        num_hidden_channels=32,
        num_bins=8,
    )
    nfm_state_dict = nfm.state_dict()
    nfm_state_dict.update(partial_state_dict)
    nfm.load_state_dict(nfm_state_dict)
    nfm.p = diagram
    nfm = nfm.to(device)
    nfm.eval()
    print("Loading model takes {:.3f}s \n".format(time.time() - start_time))

    print("Start computing integration...")
    start_time = time.time()
    num_hist_bins = 25
    with torch.no_grad():
        mean, err, partition_z = nfm.integrate_block(len_chain, num_hist_bins)
    print("Final integration time: {:.3f}s".format(time.time() - start_time))
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n".format(
            len_chain * batch_size, mean, err
        )
    )
    loss = nfm.loss_block(100, partition_z)
    print("Loss = ", loss, "\n")

    for alpha in [0.0, 0.1, 0.9, 1.0]:
        start_time = time.time()
        mean_mcmc, err_mcmc = nfm.mcmc_integration(
            num_blocks=blocks,
            len_chain=len_chain,
            thinning=1,
            alpha=alpha,
            burn_in=therm_steps,
            step_size=step_size,
            norm_std=norm_std,
            mix_rate=mix_rate,
        )
        print("MCMC integration time: {:.3f}s".format(time.time() - start_time))
        print("alpha = ", alpha)
        print(
            "MCMC result with {:d} samples is {:.5e} +/- {:.5e}. \n".format(
                len_chain * batch_size, mean_mcmc, err_mcmc
            )
        )


if __name__ == "__main__":
    main(Nblocks, beta, len_chain, batch_size, nfm_batchsize)
