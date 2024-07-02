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
order = 5
beta = 32.0
solution = 0.23  # order 2

nfm_batchsize = 50000
# batch_size = 32768
batch_size = 20000
# len_chain = 3052
len_chain = 1000
therm_steps = len_chain // 2
step_size = 0.1
mix_rate = 0.1

alpha_opt = abs(solution / (solution + 1))
accept_rate = 0.4

print(
    f"batchsize {batch_size}, therm_steps {therm_steps}, sampling_steps {len_chain}, mix_rate {mix_rate}"
)
if mix_rate != 1.0:
    print(f"Uniform random-walk U(-{step_size}, {step_size})")
print("\n")

num_hidden_layers = 1
model_state_dict_path = "nfm_o{0}_beta{1}_l{2}c32b8_state.pt".format(
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


def main(beta, len_chain, batch_size, nfm_batchsize):
    print(f"{len_chain} samples per markov chain, batch size {batch_size}")
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
    loss = nfm.loss_block(400, partition_z)
    print("Loss = ", loss, "\n")

    start_time = time.time()
    mean_mcmc, err_mcmc, adapt_step_size = nfm.mcmc_integration(
        len_chain=len_chain,
        thinning=1,
        alpha=0.0,
        burn_in=therm_steps,
        step_size=step_size,
        mix_rate=mix_rate,
        adaptive=True,
        adapt_acc_rate=accept_rate,
    )
    print("MCMC integration time: {:.3f}s".format(time.time() - start_time))
    print("alpha = 0")
    print(
        "MCMC result with {:d} samples is {:.5e} +/- {:.5e}. \n".format(
            len_chain * batch_size, mean_mcmc, err_mcmc
        )
    )
    for alpha in [0.1, 0.9, 1.0]:
        start_time = time.time()
        mean_mcmc, err_mcmc = nfm.mcmc_integration(
            len_chain=len_chain,
            thinning=1,
            alpha=alpha,
            burn_in=therm_steps,
            step_size=adapt_step_size,
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
    main(beta, len_chain, batch_size, nfm_batchsize)
