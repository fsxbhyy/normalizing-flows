import numpy as np
import os
import pandas as pd
import time  # For benchmarking
from parquetAD import FeynmanDiagram, load_leaf_info
import matplotlib.pyplot as plt
import vegas

# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch

torch.set_printoptions(precision=10)  # Set displayed output precision to 10 digits

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 3
dim = 4 * order - 1
beta = 10.0
solution = 0.2773  # order 2
# solution = -0.03115 # order 3

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

batchsize = 200000
niters = 20
# batchsize = 10**dim
diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batchsize)


@vegas.batchintegrand
def func(x):
    return torch.Tensor.numpy(diagram.prob(x))


@vegas.batchintegrand
def func0(x):
    print(x.shape)
    return x[:, 0] * x[:, 1] ** 2


integration_domain = [[0, 1]] * dim

# N_intervals = max(2, batchsize // (niters + 5) // 10)
m = vegas.AdaptiveMap(integration_domain, ninc=1000)
# m = vegas.AdaptiveMap(integration_domain, ninc=N_intervals)
print("intial grid:")
print(m.settings())

y = np.random.uniform(0.0, 1.0, (batchsize, dim))
y = np.array(y, dtype=float)

m.adapt_to_samples(y, func(y), nitn=niters)
# m.adapt_to_samples(y, func(y), nitn=10)
print(m.settings())
# print(m.extract_grid())
m.show_grid()
# m.show_grid(axes=[(2, 3)])

jac = np.empty(y.shape[0], float)
x = np.empty(y.shape, float)
m.map(y, x, jac)

jac1 = np.empty(x.shape[0], float)
y_inv = np.empty(x.shape, float)
m.invmap(x, y_inv, jac1)

fx = func(x)
fy = torch.Tensor(jac) * fx
print(torch.mean(fy), torch.std(fy) / batchsize**0.5)


def smc(f, neval, dim):
    "integrates f(y) over dim-dimensional unit hypercube"
    y = np.random.uniform(0, 1, (neval, dim))
    fy = f(y)
    return (np.average(fy), np.std(fy) / neval**0.5)


def g(y):
    jac = np.empty(y.shape[0], float)
    x = np.empty(y.shape, float)
    m.map(y, x, jac)
    return jac * func(x)


def block_results(data, nblocks=100):
    data = np.array(data)
    neval = len(data)
    val = []
    for i in range(nblocks):
        val.append(np.average(data[i * neval // nblocks : (i + 1) * neval // nblocks]))
    mean = np.average(val)
    std = np.std(val) / nblocks**0.5
    return (mean, std)


nblocks = 64
# with map
data = []
for i in range(nblocks):
    data.append(smc(g, batchsize, dim)[0])
data = np.array(data)
r = (np.average(data), np.std(data) / nblocks**0.5)
print("   SMC + map:", f"{r[0]:.6f} +- {r[1]:.6f}")

# without map
data = []
for i in range(nblocks):
    data.append(smc(func, batchsize, dim)[0])
data = np.array(data)
r = (np.average(data), np.std(data) / nblocks**0.5)
print("SMC (no map):", f"{r[0]:.6f} +- {r[1]:.6f}")

# integ = vegas.Integrator(m, alpha=0.0, beta=0.0)
# r = integ(func, neval=5e7, nitn=5)
# print(r.summary())
