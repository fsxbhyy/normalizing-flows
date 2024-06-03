import torch
import pandas as pd
from funcs_sigma import *
from parquetAD import FeynmanDiagram, load_leaf_info
import os
import torch.utils.benchmark as benchmark

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 3
dim = 4 * order - 1
beta = 10.0
batch_size = 1000000

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
nfm = torch.load("nfm_o3_beta10.0.pt")
nfm.eval()

diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batch_size)
var = torch.rand(batch_size, dim, device=diagram.val.device)
t0 = benchmark.Timer(stmt="diagram.prob(var)", globals={"diagram": diagram, "var": var})

t1 = benchmark.Timer(
    stmt="nfm.sample(batch_size)", globals={"nfm": nfm, "batch_size": batch_size}
)

print(t0.timeit(100))
print(t1.timeit(100))
