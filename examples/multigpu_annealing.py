# Import required packages
import torch
import numpy as np
import normflows as nf
import benchmark
from scipy.special import erf, gamma
import vegas
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from tqdm import tqdm

# import h5py
# import idr_torch
from absl import app, flags
import socket

# enable_cuda = True
# device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")
def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
    return s.getsockname()[0]


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]



def setup():
    # get IDs of reserved GPU
    distributed_init_method = f"tcp://{get_ip()}:{get_open_port()}"
    dist.init_process_group(backend="nccl", init_method=distributed_init_method, world_size = int(os.environ["WORLD_SIZE"]), rank = int(os.environ["RANK"]))
    # init_method='env://',
    # world_size=int(os.environ["WORLD_SIZE"]),
    # rank=int(os.environ['SLURM_PROCID']))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


def train_model_parallel(
    nfm_input,
    max_iter=1000,
    num_samples=10000,
    accum_iter=10,
    init_lr=8e-3,
    has_scheduler=True,
    proposal_model=None,
    save_checkpoint=True,
    sample_interval=5,
):
    """
    Train a neural network model with gradient accumulation.

    Args:
        nfm: The neural network model to train.
        max_iter: The maximum number of training iterations.
        num_samples: The number of samples to use for training.
        accum_iter: The number of iterations to accumulate gradients.
        has_scheduler: Whether to use a learning rate scheduler.
        proposal_model: An optional proposal model for sampling.
        save_checkpoint: Whether to save checkpoints during training every 100 iterations.
    """

    global_rank = int(os.environ["RANK"])
    rank = int(os.environ["LOCAL_RANK"])
    if global_rank == 0:
        print(f"Running basic DDP example on rank {rank}.")

    nfm = DDP(nfm_input.to(rank), device_ids=[rank])

    order = nfm_input.p.innerLoopNum
    nfm.train()  # Set model to training mode
    loss_hist = []
    # writer = SummaryWriter()  # Initialize TensorBoard writer
    if global_rank == 0:
        print("start training \n")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(nfm.parameters(), lr=init_lr)  # , weight_decay=1e-5)
    if has_scheduler:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    # Use a learning rate warmup
    warmup_epochs = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    if proposal_model is not None:
        proposal_model.to(rank)
        proposal_model.mcmc_sample(500, init=True)

    # for name, module in nfm.named_modules():
    #     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
    for it in (range(max_iter)):
        start_time = time.time()

        optimizer.zero_grad()
        loss_accum = torch.zeros(1, requires_grad=False, device=rank)
        with nfm.no_sync():
            for _ in range(accum_iter-1):
                # Compute loss
                #     if(it<max_iter/2):
                #         loss = nfm.reverse_kld(num_samples)
                #     else:
                if proposal_model is None:
                    loss = nfm.IS_forward_kld(num_samples)
                else:
                    x = proposal_model.mcmc_sample(sample_interval)
                    loss = nfm.forward_kld(x)

                loss = loss / accum_iter
                loss_accum += loss
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()

        if proposal_model is None:
            loss = nfm.IS_forward_kld(num_samples)
        else:
            x = proposal_model.mcmc_sample(sample_interval)
            loss = nfm.forward_kld(x)

        loss = loss / accum_iter
        loss_accum += loss
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            nfm.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()

        if it % 10 == 0 and global_rank==0:
            print(
                f"Iteration {it}, Loss: {loss_accum.item()}, Learning Rate: {optimizer.param_groups[0]['lr']}, Running time: {time.time() - start_time:.3f}s"
            )

        # Scheduler step after optimizer step
        if it < warmup_epochs:
            scheduler_warmup.step()
        elif has_scheduler:
            scheduler.step(loss_accum)  # ReduceLROnPlateau
            # scheduler.step()  # CosineAnnealingLR

        # Log loss
        loss_hist.append(loss_accum.item())

        # # Log metrics
        # writer.add_scalar("Loss/train", loss.item(), it)
        # writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], it)

        # save checkpoint
        # if it % 100 == 0 and it > 0 and save_checkpoint:
        if it % 50 == 0 and save_checkpoint and global_rank==0:
            torch.save(
                {
                    "model_state_dict": nfm.module.state_dict()
                    if hasattr(nfm, "module")
                    else nfm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if has_scheduler
                    else None,
                    "loss_hist": loss_hist,
                },
                f"nfm_o{order}_checkpoint_{it}.pth",
                # f"checkpoint_{it}.pth",
            )

    # writer.close()
    if global_rank==0:
        print("training finished \n")
        # print(nfm.flows[0].pvct.grid)
        # print(nfm.flows[0].pvct.inc)
        print(loss_hist)
    cleanup()


def train_model_annealing(
    nfm_input,
    max_iter=1000,
    num_samples=10000,
    accum_iter=1,
    init_lr=8e-3,
    init_beta=0.5,
    final_beta=None,
    annealing_factor=1.25,
    steps_per_temp=50,
    proposal_model=None,
    save_checkpoint=True,
    sample_interval=5,
):
    if final_beta is None:
        final_beta = (nfm_input.p.beta * nfm_input.p.EF).item()
    assert final_beta > init_beta, "final_beta should be greater than init_beta"
    nfm_input.p.beta = init_beta / nfm_input.p.EF
    nfm_input.p.mu = chemical_potential(init_beta, nfm_input.p.dim) * nfm_input.p.EF
    order = nfm_input.p.innerLoopNum

    global_rank = int(os.environ["RANK"])
    rank = int(os.environ["LOCAL_RANK"])
    if global_rank == 0:
        print(f"Running basic DDP example on rank {rank}.")

    nfm = DDP(nfm_input.to(rank), device_ids=[rank])

    
    nfm.train()  # Set model to training mode
    current_beta = init_beta
    loss_hist = np.array([])
    # writer = SummaryWriter()  # Initialize TensorBoard writer
    if global_rank==0:
        print("start Annealing training, initial beta = ", init_beta, "\n")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(nfm.parameters(), lr=init_lr)  # , weight_decay=1e-5)
    # CosineAnnealingWarmRestarts scheduler
    T_0 = steps_per_temp  # Initial period for the first restart
    T_mult = 1  # Multiplicative factor for subsequent periods
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult
    )

    # ReduceLROnPlateau scheduler
    # scheduler_annealing = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.9, patience=5, verbose=True
    # )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
    # )

    # Use a learning rate warmup
    warmup_epochs = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    # for name, module in nfm.named_modules():
    #     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
    for it in (range(max_iter)):
        start_time = time.time()
        
        optimizer.zero_grad()

        loss_accum = torch.zeros(1, requires_grad=False, device=rank)
        with nfm.no_sync():
            for _ in range(accum_iter-1):
                # Compute loss
                if proposal_model is not None and current_beta == final_beta:
                    x = proposal_model.mcmc_sample(sample_interval)
                    loss = nfm.forward_kld(x)
                else:
                    loss = nfm.IS_forward_kld(num_samples)

                loss = loss / accum_iter
                loss_accum += loss
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()

        if proposal_model is not None and current_beta == final_beta:
            x = proposal_model.mcmc_sample(sample_interval)
            loss = nfm.forward_kld(x)
        else:
            loss = nfm.IS_forward_kld(num_samples)

        loss = loss / accum_iter
        loss_accum += loss
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            nfm.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        if it % 10 == 0 and global_rank ==0:
            print(
                f"Iteration {it}, beta: {current_beta}, Loss: {loss_accum.item()}, Learning Rate: {current_lr}, Running time: {time.time() - start_time:.3f}s"
            )

        # Scheduler step after optimizer step
        if it < warmup_epochs:
            scheduler_warmup.step()
        # elif current_beta < final_beta:
        #     scheduler_annealing.step(loss_accum)  # ReduceLROnPlateau
        else:
            # scheduler.step(loss_accum)  # ReduceLROnPlateau
            scheduler.step(it - warmup_epochs)  # CosineAnnealingLR

        # Log loss
        loss_hist = np.append(loss_hist, loss_accum.item())

        # save checkpoint
        # if it > warmup_epochs and (it - warmup_epochs) % 100 == 0 and save_checkpoint:
        if it > warmup_epochs and scheduler.T_cur == scheduler.T_0 and save_checkpoint and global_rank==0:
            print(
                f"Saving NF model at the end of a CosineAnnealingWarmRestarts cycle with beta={current_beta}..."
            )
            torch.save(
                {
                    "model_state_dict": nfm.module.state_dict()
                    if hasattr(nfm, "module")
                    else nfm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_hist": loss_hist,
                    "beta": current_beta,
                },
                f"nfm_o{order}_beta{current_beta}_cyclend_checkpoint{it}.pth",
            )

        if (
            it > warmup_epochs
            and (it - warmup_epochs) % steps_per_temp == 0
            and current_beta < final_beta
        ):
            current_beta = current_beta * annealing_factor
            if current_beta >= final_beta:
                current_beta = final_beta
                # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
                # )
                # scheduler.T_0 = max_iter - it
                # scheduler.T_i = max_iter - it
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0 * 2, T_mult=2
                )
            nfm.module.p.beta = current_beta / nfm.module.p.EF
            nfm.module.p.mu = chemical_potential(current_beta, nfm.module.p.dim) * nfm.module.p.EF

    # Final save
    if global_rank==0:
        torch.save(
            {
                "model_state_dict": nfm.module.state_dict()
                if hasattr(nfm, "module")
                else nfm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_hist": loss_hist,
            },
            f"nfm_o{order}_beta{current_beta}_final.pth",
        )
        
        print(f"Annealing training complete. Final beta: {current_beta}")
        print(loss_hist)




def train_model_parallel_example(
    nfm,
    max_iter=1000,
    num_samples=10000,
    accum_iter=10,
    has_scheduler=True,
    proposal_model=None,
    save_checkpoint=True,
):
    # setup()
    global_rank = int(os.environ["RANK"])
    rank = int(os.environ["LOCAL_RANK"])
    print("test:", rank)
    # Train model
    # Move model on GPU if available
    if global_rank == 0:
        print(f"Running basic DDP example on rank {rank}.")

    # dist.init_process_group(backend='nccl',
    #                    init_method='env://',
    #                    world_size=idr_torch.size,
    #                    rank=idr_torch.rank)
    # torch.cuda.set_device(rank)
    ddp_model = DDP(nfm.to(rank), device_ids=[rank])
    ddp_model = torch.compile(ddp_model)
    loss_hist = []
    # writer = SummaryWriter()

    if global_rank == 0:
        print("before training \n")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=8e-3
    )  # , weight_decay=1e-5)
    if has_scheduler:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=False
        )

    # Use a learning rate warmup
    warmup_epochs = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    if proposal_model is not None:
        proposal_model.to(rank)
        proposal_model.mcmc_sample(500, init=True)

    # for name, module in ddp_model.named_modules():
    #     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
    for it in range(max_iter):
        start_time = time.time()

        optimizer.zero_grad()

        loss_accum = torch.zeros(1, requires_grad=False, device=rank)
        with ddp_model.no_sync():
            for _ in range(accum_iter - 1):
                if proposal_model is None:
                    # loss = ddp_model.module.IS_forward_kld(num_samples)
                    z, _ = nfm.q0(num_samples)
                    z = ddp_model.forward(z.to(rank))
                    loss = nfm.IS_forward_kld_direct(z.detach())
                else:
                    x = proposal_model.mcmc_sample()
                    loss = ddp_model.module.forward_kld(x)

                loss = loss / accum_iter
                loss_accum += loss
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()

        # An extra forward-backward pass to trigger the gradient average.
        if proposal_model is None:
            # loss = ddp_model.module.IS_forward_kld(num_samples)
            z, _ = nfm.q0(num_samples)
            z = ddp_model.forward(z.to(rank))
            loss = nfm.IS_forward_kld_direct(z.detach())
        else:
            x = proposal_model.mcmc_sample()
            loss = ddp_model.module.forward_kld(x)

        loss = loss / accum_iter
        loss_accum += loss
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
        # if it % 50 == 0:
        #    for param in ddp_model.parameters():
        #        print("test_grad:", param.grad)
        #        break
        torch.nn.utils.clip_grad_norm_(
            ddp_model.module.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()
        # Scheduler step after optimizer step
        if it < warmup_epochs:
            scheduler_warmup.step()
        elif has_scheduler:
            scheduler.step(loss_accum)  # ReduceLROnPlateau
            # scheduler.step()  # CosineAnnealingLR
        # Log loss
        loss_hist.append(loss_accum.item())

        # # Log metrics
        # writer.add_scalar("Loss/train", loss.item(), it)
        # writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], it)

        if it % 10 == 0 and global_rank == 0:
            print(
                f"Iteration {it}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]['lr']}, Running time: {time.time() - start_time:.3f}s"
            )

        # save checkpoint
        if (
            (it % 100 == 0 or it == max_iter - 1)
            and save_checkpoint
            and global_rank == 0
        ):
            torch.save(
                # {
                #    "model_state_dict": ddp_model.state_dict(),
                #    "optimizer_state_dict": optimizer.state_dict(),
                #    "scheduler_state_dict": scheduler.state_dict()
                #    if has_scheduler
                #    else None,
                #    "loss_hist": loss_hist,
                #    "it": it,
                # },
                # ddp_model.state_dict(),
                ddp_model.module,
                f"checkpoint.pt",
            )
        # dist.barrier()
    # writer.close()
    if global_rank == 0:
        print("after training \n")
        # print(ddp_model.flows[0].pvct.grid)
        # print(ddp_model.flows[0].pvct.inc)
        print(loss_hist)
    cleanup()
