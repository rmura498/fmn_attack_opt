import torch
from torch import nn, Tensor
from typing import Union


def simplex_projection(x: Tensor, ε: Union[float, Tensor] = 1) -> Tensor:
    """
    Simplex projection based on sorting.
    Parameters
    ----------
    x : Tensor
        Batch of vectors to project on the simplex.
    ε : float or Tensor
        Size of the simplex, default to 1 for the probability simplex.
    Returns
    -------
    projected_x : Tensor
        Batch of projected vectors on the simplex.
    """
    u = x.sort(dim=1, descending=True)[0]
    ε = ε.unsqueeze(1) if isinstance(ε, Tensor) else torch.tensor(ε, device=x.device)
    indices = torch.arange(x.size(1), device=x.device)
    cumsum = torch.cumsum(u, dim=1).sub_(ε).div_(indices + 1)
    K = (cumsum < u).long().mul_(indices).amax(dim=1, keepdim=True)
    τ = cumsum.gather(1, K)
    return (x - τ).clamp_(min=0)


def l0_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l0 projection"""
    delta = delta.flatten(1)
    delta_abs = delta.abs()
    sorted_indices = delta_abs.argsort(dim=1, descending=True).gather(1, (epsilon.long().unsqueeze(1) - 1).clamp_(min=0))
    thresholds = delta_abs.gather(1, sorted_indices)
    delta.mul_(delta_abs >= thresholds)


def l1_projection_(delta: Tensor, epsilon: Tensor, inplace: bool = False) -> Tensor:
    """In-place l1 projection"""
    print(delta.shape, " ", epsilon.shape)
    if (to_project := delta.norm(p=1, dim=1) > epsilon).any():
        x_to_project = delta[to_project]
        epsilon_ = epsilon[to_project] if isinstance(epsilon, Tensor) else torch.tensor([epsilon], device=delta.device)
        if not inplace:
            delta = delta.clone()
        simplex_proj = simplex_projection(x_to_project.abs(), epsilon=epsilon_)
        delta[to_project] = simplex_proj.copysign_(x_to_project)
        return delta
    else:
        return delta


def l2_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l2 projection"""
    delta = delta.flatten(1)
    l2_norms = delta.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
    delta.mul_(epsilon.unsqueeze(1) / l2_norms).clamp_(max=1)


def linf_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place linf projection"""
    delta = delta.flatten(1)
    epsilon = epsilon.unsqueeze(1)
    torch.maximum(torch.minimum(delta, epsilon, out=delta), -epsilon, out=delta)

def l0_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    n_features = x0[0].numel()
    δ = x1 - x0
    l0_projection_(δ=δ, ε=n_features * ε)
    return δ


def l1_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    threshold = (1 - ε).unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    δ_abs = δ.abs()
    mask = δ_abs > threshold
    mid_points = δ_abs.sub_(threshold).copysign_(δ)
    mid_points.mul_(mask)
    return x0 + mid_points


def l2_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    return x0.flatten(1).mul(1 - ε).add_(ε * x1.flatten(1)).view_as(x0)


def linf_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    return x0 + torch.maximum(torch.minimum(δ, ε, out=δ), -ε, out=δ).view_as(x0)
