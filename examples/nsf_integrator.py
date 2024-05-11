# Import required packages
import torch
import numpy as np
import normflows as nf
import benchmark
from scipy.special import erf, gamma

# from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py  # Make sure to import h5py

from absl import app, flags

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_string("function", "Gauss", "The function to integrate", short_name="f")
flags.DEFINE_float("alpha", 0.5, "The width of the Gaussians", short_name="a")
flags.DEFINE_integer(
    "ndims", 2, "The number of dimensions for the integral", short_name="d"
)
flags.DEFINE_integer("epochs", 300, "Number of epochs to train", short_name="e")
flags.DEFINE_integer(
    "nsamples", 10000, "Number of points to sample per epoch", short_name="s"
)


def generate_model(target, base_dist=None, num_hidden_channels=32, num_bins=8):
    # Define flows
    # torch.manual_seed(31)
    # K = 3
    ndims = target.ndims
    num_input_channels = ndims
    hidden_layers = ndims

    flows = []
    # for i in range(K):
    #     flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_input_channels, hidden_layers, num_hidden_channels)]
    #     flows += [nf.flows.LULinearPermute(num_input_channels)]
    # for i in range(K):
    #     flows += [nf.flows.CoupledRationalQuadraticSpline(num_input_channels, hidden_layers, num_hidden_channels)]
    #     flows += [nf.flows.LULinearPermute(num_input_channels)]
    # flows += [nf.flows.CoupledRationalQuadraticSpline(num_input_channels, hidden_layers, num_hidden_channels)]
    # flows += [nf.flows.CoupledRationalQuadraticSpline(num_input_channels, hidden_layers, num_hidden_channels, reverse_mask=True)]

    masks = nf.utils.iflow_binary_masks(num_input_channels)
    # print(masks)
    for mask in masks[::-1]:
        flows += [
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels,
                hidden_layers,
                num_hidden_channels,
                num_bins=num_bins,
                mask=mask,
            )
        ]

    # mask = masks[0] * 0 + 1
    # print(mask)
    # flows += [nf.flows.CoupledRationalQuadraticSpline(num_input_channels, hidden_layers, num_hidden_channels, mask=mask)]
    # Set base distribuiton
    if base_dist == None:
        base_dist = nf.distributions.base.Uniform(ndims, 0.0, 1.0)

    # Construct flow model
    nfm = nf.NormalizingFlow(base_dist, flows, target)
    nfm = nfm.to(device)
    return nfm


# def hook_fn(module, grad_input, grad_output):
#     print(f"--- Backward pass through module {module.__class__.__name__} ---")
#     print("Grad Input (input gradient to this layer):")
#     for idx, g in enumerate(grad_input):
#         print(f"Grad Input {idx}: {g.shape} - requires_grad: {g.requires_grad if g is not None else 'N/A'}")
#     print("Grad Output (gradient from this layer to next):")
#     for idx, g in enumerate(grad_output):
#         print(f"Grad Output {idx}: {g.shape} - requires_grad: {g.requires_grad if g is not None else 'N/A'}")
#     print("\n")


def train_model(nfm, max_iter=1000, num_samples=10000):
    # Train model
    # Move model on GPU if available

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

    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3)  # , weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

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
        loss = nfm.IS_forward_kld(num_samples)
        # loss = nfm.reverse_kld(num_samples)
        # loss = nfm.MCvar(num_samples)
        # print(loss)
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_value_(nfm.parameters(), clip)
            optimizer.step()

        # Log loss
        loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())
        # print(loss_hist)
        scheduler.step()

    # Plot learned distribution
    # zz = zz.to(device)
    # log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)

    # prob = torch.exp(log_prob)
    # prob[torch.isnan(prob)] = 0

    # plt.figure(figsize=(15, 15))
    # plt.pcolormesh(xx, yy, prob.data.numpy())
    # plt.gca().set_aspect('equal', 'box')
    # plt.show()
    # Plot loss
    print(loss_hist)


def main(argv):
    del argv
    ndims = FLAGS.ndims
    alpha = FLAGS.alpha
    nsamples = FLAGS.nsamples
    epochs = FLAGS.epochs
    if FLAGS.function == "Gauss":
        target = benchmark.Gauss(ndims, alpha)
    elif FLAGS.function == "Camel":
        target = benchmark.Camel(ndims, alpha)
    elif FLAGS.function == "Sharp":
        target = benchmark.Sharp()
    elif FLAGS.function == "Sphere":
        target = benchmark.Sphere(ndims)
    elif FLAGS.function == "Tight":
        target = benchmark.Tight()
    q0 = nf.distributions.base.Uniform(ndims, 0.0, 1.0)
    nfm = generate_model(target, q0)

    blocks = 10
    block_samples = 10000
    nfm.eval()
    mean, err = nfm.integrate_block(block_samples, blocks)
    nfm.train()
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * block_samples, mean, err, nfm.p.targetval
        )
    )

    loss_hist = train_model(nfm, epochs, nsamples)

    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist + np.log(mean.detach().numpy()), label="loss")
    plt.legend()
    plt.show()
    nfm.eval()
    mean, err = nfm.integrate_block(block_samples, blocks)
    nfm.train()
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * block_samples, mean, err, nfm.p.targetval
        )
    )


if __name__ == "__main__":
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
