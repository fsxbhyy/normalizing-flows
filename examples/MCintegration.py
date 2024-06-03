import torch
import time
from parquetAD import FeynmanDiagram

order = 1
beta = 10.0
Nblocks = 100
len_chain = 1000


def main(blocks, len_chain):
    print("Loading normalizing-flow model")
    start_time = time.time()
    nfm = torch.load("nfm_o{0}_beta{1}.pt".format(order, beta))
    nfm.eval()
    print("Loading model takes {:.3f}s".format(time.time() - start_time))

    batch_size = nfm.p.batchsize
    print("Start computing integration...")

    start_time = time.time()
    num_hist_bins = 25
    with torch.no_grad():
        mean, err, bins, histr, histr_weight, partition_z = nfm.integrate_block(
            blocks, num_hist_bins
        )
    print("Final integration time: {:.3f}s".format(time.time() - start_time))
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n".format(
            blocks * batch_size, mean, err
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
    main(Nblocks, len_chain)
