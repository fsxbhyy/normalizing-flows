import numpy as np

import torch

def kernelFermiT(τ, ω, β):
    # Ensure that τ, ω, β are tensors and possibly move them to the appropriate device (e.g., GPU)
    τ = torch.as_tensor(τ).to(ω.device)
    β = torch.as_tensor(β).to(ω.device)

    # Condition checks in PyTorch, keeping the results in tensor format
    condition_τ_range = (-β < τ) & (τ <= β)
    if not torch.all(condition_τ_range):
        raise ValueError("τ values must be in the range (-β, β]")

    sign = torch.where(τ >= 0, 1.0, -1.0)
    ω_positive = ω > 0

    a = torch.where(ω_positive, -τ, β - τ)
    b = torch.where(ω_positive, -β, β)

    # Use torch operations to ensure calculations are done on GPU if tensors are on GPU
    exp_ωa = torch.exp(ω * a)
    exp_ωb = torch.exp(ω * b)
    result = sign * exp_ωa / (1 + exp_ωb)

    return result

def kernelFermiT(τ, ω, β):
    if not (-β < τ <= β):
        raise ValueError(f"τ={τ} must be (-β, β] where β={β}")
    
    if τ >= 0.0:
        sign = 1.0
        if ω > 0.0:
            a, b = -τ, -β
        else:
            a, b = β - τ, β
    else:
        sign = -1.0
        if ω > 0.0:
            a, b = -(β + τ), -β
        else:
            a, b = -τ, β

    return sign * np.exp(ω * a) / (1 + np.exp(ω * b))
