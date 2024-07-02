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

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")
torch.set_printoptions(precision=10)  # Set displayed output precision to 10 digits

root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 2
dim = 4 * order - 1
beta = 32.0
solution = 0.23  # order 2
# solution = -0.03115 # order 3
integration_domain = [[0, 1]] * dim
niters = 20
num_adapt_samples = 2000000

alpha_opt = abs(solution / (solution + 1))
batchsize = 32768
nblocks = 3052
therm_steps = 1000
mu = 0.0
step_size = 0.1
# type = "gaussian"  # "gaussian" or "uniform"
type = "uniform"  # "gaussian" or "uniform"
# type = None
mix_rate = 0.0

print(
    f"batchsize {batchsize}, nblocks {nblocks}, therm_steps {therm_steps}, mix_rate {mix_rate}"
)
if type == "gaussian":
    print(f"Gaussian random-walk N({mu}, {step_size}^2)")
elif type == "uniform":
    print(f"Uniform random-walk U(-{step_size}, {step_size})")
else:
    print("Global random sampling")
print("\n")

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

diagram_eval = FeynmanDiagram(
    order, beta, loopBasis, leafstates[0], leafvalues[0], batchsize
)

###### VegasMap by torch
map_torch = VegasMap(
    diagram_eval, dim, integration_domain, batchsize, num_adapt_samples
)
map_torch = map_torch.to(device)

# Importance sampling with Vegas map (torch)
map_torch = map_torch.to(device)
start_time = time.time()
mean, std = map_torch.integrate_block(nblocks)
print("   Importance sampling with VEGAS map (torch):", f"{mean:.6f} +- {std:.6f}")
end_time = time.time()
wall_clock_time = end_time - start_time
print(f"Wall-clock time: {wall_clock_time:.3f} seconds \n")

# Vegas-map MCMC
len_chain = nblocks
start_time = time.time()
mean, error, adapt_step_size = map_torch.mcmc(
    len_chain,
    alpha=0.0,
    burn_in=therm_steps,
    step_size=step_size,
    mu=mu,
    type=type,
    mix_rate=mix_rate,
    adaptive=True,
)
print("   VEGAS-map MCMC (alpha = 0):", f"{mean:.6f} +- {error:.6f}")
print("MCMC integration time: {:.3f}s \n".format(time.time() - start_time))
for alpha in [0.1, 0.9, 1.0]:
    start_time = time.time()
    mean, error = map_torch.mcmc(
        len_chain,
        alpha=alpha,
        burn_in=therm_steps,
        step_size=adapt_step_size,
        mu=mu,
        type=type,
        mix_rate=mix_rate,
    )  # , thinning=20
    print(f"   VEGAS-map MCMC (alpha = {alpha}):", f"{mean:.6f} +- {error:.6f}")
    print("MCMC integration time: {:.3f}s \n".format(time.time() - start_time))


# @vegas.batchintegrand
# def integrand_eval(x):
#     return torch.Tensor.numpy(diagram_eval.prob(torch.Tensor(x)))


# def smc(f, neval, dim):
#     "integrates f(y) over dim-dimensional unit hypercube"
#     y = np.random.uniform(0, 1, (neval, dim))
#     fy = f(y)
#     return (np.average(fy), np.std(fy) / neval**0.5)

# # without map
# start_time = time.time()
# data = []
# for i in range(nblocks):
#     data.append(smc(integrand_eval, batchsize, dim)[0])
# data = np.array(data)
# r = (np.average(data), np.std(data) / nblocks**0.5)
# print("   SMC (no map):", f"{r[0]:.6f} +- {r[1]:.6f}")
# end_time = time.time()
# wall_clock_time = end_time - start_time
# print(f"Wall-clock time: {wall_clock_time:.3f} seconds")

# print(bins)
# torch.save(hist, "histogramVegas_o{0}_beta{1}.pt".format(order, beta))
# torch.save(hist_weight, "histogramWeightVegas_o{0}_beta{1}.pt".format(order, beta))


# integ = vegas.Integrator(m, alpha=0.0, beta=0.0)
# r = integ(func, neval=5e7, nitn=5)
# print(r.summary())
