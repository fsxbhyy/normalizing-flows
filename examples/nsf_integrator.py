# Import required packages
import torch
import numpy as np
import normflows as nf
import benchmark
from scipy.special import erf, gamma
from tqdm import tqdm
import h5py  # Make sure to import h5py

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string('function', 'Gauss', 'The function to integrate',
                    short_name='f')
flags.DEFINE_float('alpha', 1.0, 'The width of the Gaussians',
                   short_name='a')
flags.DEFINE_integer('ndims', 2, 'The number of dimensions for the integral',
                     short_name='d')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('nsamples', 10000, 'Number of points to sample per epoch',
                     short_name='s')

def generate_model(target):
    # Define flows
    K = 2
    torch.manual_seed(0)
    ndims = target.ndims
    latent_size = ndims
    hidden_units = 4
    hidden_layers = ndims

    flows = []
    #for i in range(K):
    #     flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
    #     flows += [nf.flows.LULinearPermute(latent_size)]
    flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
    flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, reverse_mask=True)]
    # Set base distribuiton
    q0 = nf.distributions.base.Uniform(ndims, 0.0, 1.0)
        
    # Construct flow model
    nfm = nf.NormalizingFlow(q0, flows, target)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    nfm = nfm.to(device)
    return nfm
def train_model(nfm, max_iter = 1000, num_samples = 10000):
    # Train model
    clip = 10.0

    loss_hist = np.array([])

    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-2)#, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        
        # Get training samples
    #     x_np, _ = make_moons(num_samples, noise=0.1)
    #     x = torch.tensor(x_np).float().to(device)
        
        # Compute loss
    #     if(it<max_iter/2):
    #         loss = nfm.reverse_kld(num_samples)
    #     else:
        loss = nfm.MCvar(num_samples)

        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_value_(nfm.parameters(), clip)
            optimizer.step()
        
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        scheduler.step()

def main(argv):
    del argv
    ndims = FLAGS.ndims
    alpha = FLAGS.alpha
    nsamples = FLAGS.nsamples
    epochs = FLAGS.epochs
    if FLAGS.function == 'Gauss':
        target = benchmark.Gauss(ndims,alpha)
    elif FLAGS.function == 'Camel':
        target = benchmark.Camel(ndims,alpha)
    elif FLAGS.function == 'Sharp':
        target = benchmark.Sharp()
    elif FLAGS.function == 'Sphere':
        target = benchmark.Sphere(ndims)
    elif FLAGS.function == 'Tight':    
        target = benchmark.Tight()
    
    nfm = generate_model(target)   

    train_model(nfm, epochs, nsamples)
    nfm.eval()
    blocks = 100
    block_samples = 100000 
    mean, err = nfm.integrate_block(block_samples, blocks)
    nfm.train()
    print("Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks*block_samples,  mean, err, nfm.p.targetval))


if __name__ == '__main__':
    app.run(main)

    # Plot learned distribution
    # if (it + 1) % show_iter == 0:
    #     nfm.eval()
    #     log_prob = nfm.log_prob(zz)
    #     nfm.train()
    #     prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    #     prob[torch.isnan(prob)] = 0

    #     plt.figure(figsize=(15, 15))
    #     plt.pcolormesh(xx, yy, prob.data.numpy())
    #     plt.gca().set_aspect('equal', 'box')
    #     plt.show()
    # scheduler.step()
