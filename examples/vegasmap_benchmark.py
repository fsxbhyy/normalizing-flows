import numpy as np
import os
import pandas as pd
import time  # For benchmarking
from parquetAD import FeynmanDiagram, load_leaf_info
import matplotlib.pyplot as plt
import vegas
from vegas_torch import VegasMap

# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch

torch.set_printoptions(precision=10)  # Set displayed output precision to 10 digits

root_dir = os.path.join(os.path.dirname(__file__), "source_codeParquetAD/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 1
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

batchsize = 100000
niters = 20
# batchsize = 10**dim
diagram = FeynmanDiagram(order, loopBasis, leafstates[0], leafvalues[0], batchsize)


@vegas.batchintegrand
def func(x):
    return torch.Tensor.numpy(diagram.prob(torch.Tensor(x)))


@vegas.batchintegrand
def func0(x):
    return -(x[:, 0] ** 2) - x[:, 1] ** 2 + x[:, 0] + x[:, 1]


integration_domain = [[0, 1]] * dim

m = vegas.AdaptiveMap(integration_domain, ninc=1000)
# N_intervals = max(2, batchsize // (niters + 5) // 10)
# m = vegas.AdaptiveMap(integration_domain, ninc=N_intervals)
print("intial grid:")
print(m.settings())

y = np.random.uniform(0.0, 1.0, (batchsize, dim))

# m.adapt_to_samples(y, func0(y), nitn=niters)
m.adapt_to_samples(y, func(y), nitn=niters)
print(m.settings())
# m.show_grid()
# m.show_grid(axes=[(2, 3)])

y = np.random.uniform(0.0, 1.0, (batchsize, dim))

jac = np.empty(y.shape[0], float)
x = np.empty(y.shape, float)
m.map(y, x, jac)
# print("x:", x)
# print("jac:", jac)
# jac1 = np.empty(x.shape[0], float)
# y_inv = np.empty(x.shape, float)
# m.invmap(x, y_inv, jac1)
# print("inv y:", y_inv)
# print("inv jac:", jac1)

fx = func(x)
fy = torch.Tensor(jac) * fx
print(torch.mean(fy), torch.std(fy) / batchsize**0.5)


map_torch = VegasMap(func, dim, integration_domain, batchsize)
# map_torch = VegasMap(m, dim, integration_domain, batchsize)

x, jac = map_torch(torch.Tensor(y))
fx = func(x)
fy = jac * fx
print(torch.mean(fy), torch.std(fy) / batchsize**0.5)
# y = torch.Tensor(y)
# print(y)
# x, jac = map_torch(y)
# print("torch map x:", x, "\n jac:", jac)
# y0, jac = map_torch.inverse(x)
# print("torch invmap y:", y0, "\n jac:", jac)


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


def g_torch(y):
    x, jac = map_torch.forward(torch.Tensor(y))
    return jac.numpy() * func(x)


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
# with Veags map
start_time = time.time()
data = []
for i in range(nblocks):
    data.append(smc(g, batchsize, dim)[0])
data = np.array(data)
r = (np.average(data), np.std(data) / nblocks**0.5)
print("   SMC + map:", f"{r[0]:.6f} +- {r[1]:.6f}")
end_time = time.time()
wall_clock_time = end_time - start_time
print(f"Wall-clock time: {wall_clock_time:.3f} seconds")

# with Vegas map (torch)
start_time = time.time()
mean, std = map_torch.integrate_block(nblocks)
print("   Vegas map (torch):", f"{mean:.6f} +- {std:.6f}")
end_time = time.time()
wall_clock_time = end_time - start_time
print(f"Wall-clock time: {wall_clock_time:.3f} seconds")
# print(bins)
# torch.save(hist, "histogramVegas_o{0}_beta{1}.pt".format(order, beta))
# torch.save(hist_weight, "histogramWeightVegas_o{0}_beta{1}.pt".format(order, beta))

# data = []
# for i in range(nblocks):
#     data.append(smc(g_torch, batchsize, dim)[0])
# data = np.array(data)
# r = (np.average(data), np.std(data) / nblocks**0.5)
# print("   SMC + map (torch):", f"{r[0]:.6f} +- {r[1]:.6f}")
# end_time = time.time()
# wall_clock_time = end_time - start_time
# print(f"Wall-clock time: {wall_clock_time:.3f} seconds")

# without map
start_time = time.time()
data = []
for i in range(nblocks):
    data.append(smc(func, batchsize, dim)[0])
data = np.array(data)
r = (np.average(data), np.std(data) / nblocks**0.5)
print("   SMC (no map):", f"{r[0]:.6f} +- {r[1]:.6f}")
end_time = time.time()
wall_clock_time = end_time - start_time
print(f"Wall-clock time: {wall_clock_time:.3f} seconds")
# integ = vegas.Integrator(m, alpha=0.0, beta=0.0)
# r = integ(func, neval=5e7, nitn=5)
# print(r.summary())
