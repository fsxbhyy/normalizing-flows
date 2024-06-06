import torch
import pandas as pd
from funcs_sigma import *
from parquetAD import FeynmanDiagram, load_leaf_info
import os
import torch.utils.benchmark as benchmark
import numpy as np

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 5
dim = 4 * order - 1
beta = 10.0
batch_size = 10000
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
diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batch_size)


var = torch.rand(batch_size, dim, device=device)

t1 = benchmark.Timer(
    stmt="diagram.prob(var)",
    globals={"diagram": diagram, "var": var},
    label="Self-energy diagram (order {0} beta {1})".format(order, beta),
    sub_label="Evaluating integrand (batchsize {0})".format(batch_size),
)

t2 = benchmark.Timer(
    stmt="diagram.prob(var, 1)",
    globals={"diagram": diagram, "var": var},
    label="Self-energy diagram (order {0} beta {1})".format(order, beta),
    sub_label="Evaluating integrand with jit (batchsize {0})".format(batch_size),
)

print(t1.timeit(Neval))
print(t2.timeit(Neval))
print(t1.timeit(Neval))
print(t2.timeit(Neval))
