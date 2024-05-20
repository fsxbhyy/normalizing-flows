import numpy as np
import os
import pandas as pd
import time  # For benchmarking
from parquetAD import FeynmanDiagram, load_leaf_info
import matplotlib.pyplot as plt

# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch

torch.set_printoptions(precision=10)  # Set displayed output precision to 10 digits
from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import (
    Trapezoid,
    Simpson,
    Boole,
    MonteCarlo,
    VEGAS,
)  # The available integrators
from torchquad.utils.set_precision import set_precision


def print_error(result, solution):
    print("Results:", result.item())
    print(f"Abs. Error: {(torch.abs(result - solution).item()):.8e}")
    print(f"Rel. Error: {(torch.abs((result - solution) / solution).item()):.8e}")


# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float32")

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 2
dim = 4 * order - 1
beta = 10.0
solution = 0.2773

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

# batchsize = 100000
batchsize = 3968
diagram = FeynmanDiagram(loopBasis, leafstates[0], leafvalues[0], batchsize)

integration_domain = [[0, 1]] * dim
vegas = VEGAS()
result = vegas.integrate(
    # diagram.prob, dim=dim, N=batchsize, integration_domain=integration_domain
    diagram.prob,
    dim=dim,
    N=100000,
    integration_domain=integration_domain,
)
print_error(result, solution)
