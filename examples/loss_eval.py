import os
import pandas as pd
import numpy as np
import torch
import normflows as nf
from funcs_sigma import *
import time
from parquetAD import FeynmanDiagram
# from torch.utils.viz._cycles import warn_tensor_cycles

# warn_tensor_cycles()

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 1
beta = 10.0
batch_size = 100000
hidden_layers = 2
num_hidden_channels = 32
num_bins = 8

blocks = 100

nfm = torch.load("nfm_o{0}_beta{1}.pt".format(order, beta))
# nfm = torch.load("nfm_o{0}_beta{1}_r1.pt".format(order, beta))

print("Start computing integration...")
start_time = time.time()
num_hist_bins = 25
with torch.no_grad():
    mean, err, bins, histr, histr_weight, partition_z = nfm.integrate_block(
        blocks, num_hist_bins
    )
print("Final integration time: {:.3f}s".format(time.time() - start_time))
print(
    "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
        blocks * batch_size, mean, err, nfm.p.targetval
    )
)

loss = nfm.loss_block(100, partition_z)
print("Final loss: ", loss)
