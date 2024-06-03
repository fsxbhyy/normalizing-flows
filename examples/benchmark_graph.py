import torch
import pandas as pd
from funcs_sigma import *
from parquetAD import FeynmanDiagram, load_leaf_info
import os
import torch.utils.benchmark as benchmark
import vegas
from vegas_torch import VegasMap
import numpy as np

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 1
dim = 4 * order - 1
beta = 10.0
batch_size = 100000
Neval = 10

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

# for batchsize in [10**i for i in range(0, 7)]:
batch_size = 1000000
diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batch_size)


# Vegas
@vegas.batchintegrand
def func(x):
    return torch.Tensor.numpy(diagram.prob(torch.Tensor(x)))


integration_domain = [[0, 1]] * dim
map_torch = VegasMap(func, dim, integration_domain, batch_size)

var = torch.rand(batch_size, dim, device=diagram.val.device)
t0 = benchmark.Timer(
    stmt="map_torch.forward(var)",
    globals={"map_torch": map_torch, "var": var},
    label="Self-energy diagram (order {0} beta {1})".format(order, beta),
    sub_label="sampling using VEAGS map",
)

nfm = torch.load("nfm_o{0}_beta{1}.pt".format(order, beta))
nfm.eval()
t1 = benchmark.Timer(
    stmt="nfm.forward(var)",
    globals={"nfm": nfm, "var": var},
    label="Self-energy diagram (order {0} beta {1})".format(order, beta),
    sub_label="sampling using normalizing flow",
)

t2 = benchmark.Timer(
    stmt="diagram.prob(var)",
    globals={"diagram": diagram, "var": var},
    label="Self-energy diagram (order {0} beta {1})".format(order, beta),
    sub_label="Evaluating integrand",
)

print(t0.timeit(Neval))
print(t1.timeit(Neval))
print(t2.timeit(Neval))
