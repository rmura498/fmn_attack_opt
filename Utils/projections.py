import torch
from torch import nn, Tensor
from typing import Union


def simplex_projection(x: Tensor, epsilon: Union[float, Tensor] = 1) -> Tensor:
    """
    Simplex projection based on sorting.
    Parameters
    ----------
    x : Tensor
        Batch of vectors to project on the simplex.
    epsilon : float or Tensor
        Size of the simplex, default to 1 for the probability simplex.
    Returns
    -------
    projected_x : Tensor
        Batch of projected vectors on the simplex.
    """
    u = x.sort(dim=1, descending=True)[0]
    epsilon = epsilon.unsqueeze(1) if isinstance(epsilon, Tensor) else torch.tensor(epsilon, device=x.device)
    indices = torch.arange(x.size(1), device=x.device)
    cumsum = torch.cumsum(u, dim=1).sub_(epsilon).div_(indices + 1)
    K = (cumsum < u).long().mul_(indices).amax(dim=1, keepdim=True)
    τ = cumsum.gather(1, K)
    return (x - τ).clamp_(min=0)


def l1_ball_euclidean_projection(x: Tensor, epsilon: Union[float, Tensor], inplace: bool = False) -> Tensor:
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.
    Adapted from Tony Duan's implementation https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

    Parameters
    ----------
    x: Tensor
        Batch of tensors to project.
    epsilon: float or Tensor
        Radius of L1-ball to project onto. Can be a single value for all tensors in the batch or a batch of values.
    inplace : bool
        Can optionally do the operation in-place.

    Returns
    -------
    projected_x: Tensor
        Batch of projected tensors with the same shape as x.

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    if (to_project := x.norm(p=1, dim=1) > epsilon).any():
        x_to_project = x[to_project]
        epsilon_ = epsilon[to_project] if isinstance(epsilon, Tensor) else torch.tensor([epsilon], device=x.device)
        if not inplace:
            x = x.clone()
        simplex_proj = simplex_projection(x_to_project.abs(), epsilon=epsilon_)
        x[to_project] = simplex_proj.copysign_(x_to_project)
        return x
    else:
        return x

def l0_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l0 projection"""
    delta = delta.flatten(1)
    delta_abs = delta.abs()
    sorted_indices = delta_abs.argsort(dim=1, descending=True).gather(1, (epsilon.long().unsqueeze(1) - 1).clamp_(min=0))
    thresholds = delta_abs.gather(1, sorted_indices)
    delta.mul_(delta_abs >= thresholds)


def l1_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l1 projection"""
    l1_ball_euclidean_projection(x=delta.flatten(1), epsilon=epsilon, inplace=True)


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


def l0_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    n_features = x0[0].numel()
    delta = x1 - x0
    l0_projection_(delta=delta, epsilon=n_features * epsilon)
    return delta


def l1_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    threshold = (1 - epsilon).unsqueeze(1)
    delta = (x1 - x0).flatten(1)
    delta_abs = delta.abs()
    mask = delta_abs > threshold
    mid_points = delta_abs.sub_(threshold).copysign_(delta)
    mid_points.mul_(mask)
    return x0 + mid_points


def l2_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    epsilon = epsilon.unsqueeze(1)
    return x0.flatten(1).mul(1 - epsilon).add_(epsilon * x1.flatten(1)).view_as(x0)


def linf_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    epsilon = epsilon.unsqueeze(1)
    delta = (x1 - x0).flatten(1)
    return x0 + torch.maximum(torch.minimum(delta, epsilon, out=delta), -epsilon, out=delta).view_as(x0)