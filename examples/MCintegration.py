import torch
import pandas as pd
import time
from parquetAD import FeynmanDiagram
from parquetAD import FeynmanDiagram, load_leaf_info
import os

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 1
beta = 10.0
Nblocks = 100
batch_size = 1000
len_chain = 1000

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


def main(blocks, len_chain, batch_size):
    diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batch_size)

    print("Loading normalizing-flow model")
    start_time = time.time()
    nfm = torch.load("nfm_o{0}_beta{1}.pt".format(order, beta))
    nfm.eval()
    nfm.p = diagram
    nfm = nfm.to(device)
    print("Loading model takes {:.3f}s".format(time.time() - start_time))

    print("Start computing integration...")
    start_time = time.time()
    num_hist_bins = 25
    with torch.no_grad():
        mean, err, bins, histr, histr_weight, partition_z = nfm.integrate_block(
            len_chain, num_hist_bins
        )
    print("Final integration time: {:.3f}s".format(time.time() - start_time))
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n".format(
            len_chain * batch_size, mean, err
        )
    )

    for alpha in [0.0, 0.5, 1.0]:
        start_time = time.time()
        mean_mcmc, err_mcmc = nfm.mcmc_integration(
            num_blocks=blocks, len_chain=len_chain, thinning=1, alpha=alpha
        )
        print("MCMC integration time: {:.3f}s".format(time.time() - start_time))
        print("alpha = ", alpha)
        print(
            "MCMC result with {:d} samples is {:.5e} +/- {:.5e}. \n".format(
                blocks * batch_size, mean_mcmc, err_mcmc
            )
        )


if __name__ == "__main__":
    main(Nblocks, len_chain, batch_size)
