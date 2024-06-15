# Import required packages
import torch
import numpy as np
import normflows as nf
import benchmark
# from sklearn.datasets import make_moons

# from matplotlib import pyplot as plt

from tqdm import tqdm

K = 2
torch.manual_seed(0)

ndims = 2
alpha = 0.5
nsamples = 10000
epochs = 300
# if FLAGS.function == 'Gauss':
target = benchmark.Gauss(ndims, alpha)

latent_size = 2
hidden_units = 32
num_blocks = 2

flows = []
for i in range(K):
    flows += [
        nf.flows.AutoregressiveRationalQuadraticSpline(
            latent_size, num_blocks, hidden_units
        )
    ]
    flows += [nf.flows.LULinearPermute(latent_size)]

# masks = nf.utils.iflow_binary_masks(latent_size)
# print(masks)
# for mask in masks[::-1]:
#     flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, num_blocks, hidden_units, mask=mask)]

# Set base distribuiton
# q0 = nf.distributions.DiagGaussian(2, trainable=False)
q0 = nf.distributions.base.Uniform(ndims, 0.0, 1.0)

print(q0.low.device)
# Construct flow model
nfm = nf.NormalizingFlow(q0, flows, target)
print(nfm.q0.low.device)
# Move model on GPU if available
enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")
nfm = nfm.to(device)


max_iter = 1000
num_samples = 2**9
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-5)
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    # x_np, _ = make_moons(num_samples, noise=0.1)
    # x = torch.tensor(x_np).float().to(device)

    # Compute loss
    # loss = nfm.forward_kld(x)
    loss = nfm.reverse_kld(num_samples)
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())
