# Import required packages
import torch
import numpy as np
import normflows as nf
import benchmark
from scipy.special import erf, gamma
# from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py  # Make sure to import h5py

# from absl import app, flags

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# FLAGS = flags.FLAGS
# flags.DEFINE_string('function', 'Gauss', 'The function to integrate',
#                     short_name='f')
# flags.DEFINE_float('alpha', 0.5, 'The width of the Gaussians',
#                    short_name='a')
# flags.DEFINE_integer('ndims', 2, 'The number of dimensions for the integral',
#                      short_name='d')
# flags.DEFINE_integer('epochs', 300, 'Number of epochs to train',
#                      short_name='e')
# flags.DEFINE_integer('nsamples', 10000, 'Number of points to sample per epoch',
#                      short_name='s')

ndims = 2
alpha = 0.5
nsamples = 10000
epochs = 300
# if FLAGS.function == 'Gauss':
target = benchmark.Gauss(ndims,alpha)
# elif FLAGS.function == 'Camel':
#     target = benchmark.Camel(ndims,alpha)
# elif FLAGS.function == 'Sharp':
#     target = benchmark.Sharp()
# elif FLAGS.function == 'Sphere':
#     target = benchmark.Sphere(ndims)
# elif FLAGS.function == 'Tight':    
#     target = benchmark.Tight()
# Define flows
torch.manual_seed(13)
K = 3
ndims = 2
latent_size = ndims
hidden_units = 32
hidden_layers = ndims


flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
    flows += [nf.flows.LULinearPermute(latent_size)]
#for i in range(K):
#     flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
#     flows += [nf.flows.LULinearPermute(latent_size)]
# flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
# flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, reverse_mask=True)]

# masks = nf.utils.iflow_binary_masks(latent_size)
# # print(masks)
# for mask in masks[::-1]:
#     flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, mask=mask)]

# mask = masks[0] * 0 + 1
# print(mask)
# flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, mask=mask)]
# Set base distribuiton
base_dist = nf.distributions.base.Uniform(ndims, 0.0, 1.0, trainable=False)
    
# Construct flow model
nfm = nf.NormalizingFlow(base_dist, flows, target)


max_iter = epochs
num_samples = nsamples
 
   
nfm = nfm.to(device)
clip = 10.0

loss_hist = np.array([])

# grid_size = 100
# xx, yy = torch.meshgrid(torch.linspace(0.0, 1.0, grid_size), torch.linspace(0.0, 1.0, grid_size))
# zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
# #zz = zz.to(device)
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)

# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0

# plt.figure(figsize=(15, 15))
# plt.pcolormesh(xx, yy, prob.data.numpy())
# plt.gca().set_aspect('equal', 'box')
# plt.show()

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3)#, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

# for name, module in nfm.named_modules():
#     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
#     x_np, _ = make_moons(num_samples, noise=0.1)
#     x = torch.tensor(x_np).float().to(device)
    
    # Compute loss
#     if(it<max_iter/2):
#         loss = nfm.reverse_kld(num_samples)
#     else:
    # loss = nfm.IS_forward_kld(num_samples)
    loss = nfm.reverse_kld(num_samples)
    #loss = nfm.MCvar(num_samples)
    #print(loss)
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        torch.nn.utils.clip_grad_value_(nfm.parameters(), clip)
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    #print(loss_hist)
    #scheduler.step()
    
# Plot learned distribution
#zz = zz.to(device)
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)

# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0

# plt.figure(figsize=(15, 15))
# plt.pcolormesh(xx, yy, prob.data.numpy())
# plt.gca().set_aspect('equal', 'box')
# plt.show()
# Plot loss
# blocks = 10
# block_samples = 10000 
# nfm.eval()
# mean, err = nfm.integrate_block(block_samples, blocks)
# nfm.train()
# print("Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
#         blocks*block_samples,  mean, err, nfm.p.targetval))
    
# loss_hist = train_model(nfm, epochs, nsamples)

# plt.figure(figsize=(10, 10))
# plt.plot(loss_hist + np.log(mean.detach().numpy()), label='loss')
# plt.legend()
# plt.show()
    
