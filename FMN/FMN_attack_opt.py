import torch
import math
from functools import partial
from typing import Optional

from torch import nn, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from Utils.projections import l0_projection_, l1_projection_, l2_projection_, linf_projection_
from Utils.projections import l0_mid_points, l1_mid_points, l2_mid_points, linf_mid_points
from Utils.plots import plot_loss_epsilon_over_steps
from Utils.metrics import difference_of_logits


class FmnOpt:
    """
    Fast Minimum-Norm attack from https://arxiv.org/abs/2102.12827.
    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    norm : float
        Norm to minimize in {0, 1, 2 ,float('inf')}.
    targeted : bool
        Whether to perform a targeted attack or not.
    steps : int
        Number of optimization steps.
    alpha_init : float
        Initial step size.
    alpha_final : float
        Final step size after cosine annealing.
    gamma_init : float
        Initial factor by which epsilon is modified: epsilon = epsilon * (1 + or - gamma).
    gamma_final : float
        Final factor, after cosine annealing, by which epsilon is modified.
    starting_points : Tensor
        Optional warm-start for the attack.
    binary_search_steps : int
        Number of binary search steps to find the decision boundary between inputs and starting_points.
    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.
    """

    def __init__(self,
                 model: nn.Module,
                 inputs: Tensor,
                 labels: Tensor,
                 norm: float,
                 targeted: bool = False,
                 steps: int = 10,
                 alpha_init: float = 1.0,
                 alpha_final: Optional[float] = None,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 starting_points: Optional[Tensor] = None,
                 binary_search_steps: int = 10,
                 optimizer=torch.optim.SGD,
                 scheduler=CosineAnnealingLR
                 ):

        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.norm = norm
        self.targeted = targeted
        self.multiplier = 1 if self.targeted else -1
        self.steps = steps
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_view = None
        self.best_norm = None
        self.worst_norm = None
        self.best_adv = None
        self.adv_found = None
        self.delta = None
        self.epsilon = None
        self.dual = None
        self.projection = None
        self.mid_point = None

    def init_attack(self):

        _dual_projection_mid_points = {
            0: (None, l0_projection_, l0_mid_points),
            1: (float('inf'), l1_projection_, l1_mid_points),
            2: (2, l2_projection_, l2_mid_points),
            float('inf'): (1, linf_projection_, linf_mid_points),
        }
        if self.inputs.min() < 0 or self.inputs.max() > 1: raise ValueError(
            'Input values should be in the [0, 1] range.')
        device = self.inputs.device
        batch_size = len(self.inputs)
        self.batch_view = lambda tensor: tensor.view(batch_size, *[1] * (self.inputs.ndim - 1))
        self.dual, self.projection, self.mid_point = _dual_projection_mid_points[self.norm]
        self.alpha_final = self.alpha_init / 100 if self.alpha_final is None else self.alpha_final

        # If starting_points is provided, search for the boundary
        if self.starting_points is not None:
            is_adv = self.model(self.starting_points).argmax(dim=1)
            if not is_adv.all():
                raise ValueError('Starting points are not all adversarial.')
            lower_bound = torch.zeros(batch_size, device=device)
            upper_bound = torch.ones(batch_size, device=device)
            for _ in range(self.binary_search_steps):
                self.epsilon = (lower_bound + upper_bound) / 2
                mid_points = self.mid_point(x0=self.inputs, x1=self.starting_points, epsilon=self.epsilon)
                pred_labels = self.model(mid_points).argmax(dim=1)
                is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
                lower_bound = torch.where(is_adv, lower_bound, self.epsilon)
                upper_bound = torch.where(is_adv, self.epsilon, upper_bound)

            self.delta = self.mid_point(x0=self.inputs, x1=self.starting_points, epsilon=self.epsilon) - self.inputs
        else:
            self.delta = torch.zeros_like(self.inputs)
        self.delta.requires_grad_(True)

        if self.norm == 0:
            self.epsilon = torch.ones(batch_size,
                                      device=device) if self.starting_points is None else self.delta.flatten(1).norm(
                p=0,
                dim=0)
        else:
            self.epsilon = torch.full((batch_size,), float('inf'), device=device)

        # Init trackers
        self.worst_norm = torch.maximum(self.inputs, 1 - self.inputs).flatten(1).norm(p=self.norm, dim=1)
        self.best_norm = self.worst_norm.clone()
        self.best_adv = self.inputs.clone()

        self.adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

        self.epsilon.requires_grad_()
        print(f"Initial epsilon value: {self.epsilon.data.norm(p=self.norm)}")

    def run(self):

        loss_per_iter = []
        epsilon_per_iter = []

        delta_optim = self.optimizer([self.delta], lr=self.alpha_init)
        delta_scheduler = self.scheduler(delta_optim, T_max=self.steps, eta_min=self.alpha_final)

        for i in range(self.steps):
            delta_optim.zero_grad()

            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = self.delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_inputs = self.inputs + self.delta
            logits = self.model(adv_inputs)
            pred_labels = logits.argmax(dim=1)

            if i == 0:
                labels_infhot = torch.zeros_like(logits).scatter_(1, self.labels.unsqueeze(1), float('inf'))
                logit_diff_func = partial(difference_of_logits, labels=self.labels, labels_infhot=labels_infhot)

            logit_diffs = logit_diff_func(logits=logits)
            loss = -(self.multiplier * logit_diffs)

            loss_per_iter.append(loss.sum().clone().detach().numpy())
            epsilon_per_iter.append(torch.linalg.norm(self.epsilon).clone().detach().numpy())

            loss.sum().backward()

            is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
            is_smaller = delta_norm < self.best_norm
            is_both = is_adv & is_smaller
            self.adv_found.logical_or_(is_adv)
            best_norm = torch.where(is_both, delta_norm, self.best_norm)
            best_adv = torch.where(self.batch_view(is_both), adv_inputs.detach(), self.best_adv)

            alpha = delta_scheduler.get_last_lr()[0]
            # print(f"alpha: {alpha} - gamma: {gamma}\n")

            if self.norm == 0:
                self.epsilon = torch.where(is_adv,
                                           torch.minimum(
                                               torch.minimum(self.epsilon - 1, (self.epsilon * (1 - gamma)).floor_()),
                                               best_norm),
                                           torch.maximum(self.epsilon + 1, (self.epsilon * (1 + gamma)).floor_()))
                self.epsilon.clamp_(min=0)
            else:
                distance_to_boundary = loss.detach().abs() / self.delta.grad.data.flatten(1).norm(p=self.dual,
                                                                                                  dim=1).clamp_(
                    min=1e-12)
                self.epsilon = torch.where(is_adv,
                                           torch.minimum(self.epsilon * (1 - gamma), best_norm),
                                           torch.where(self.adv_found, self.epsilon * (1 + gamma),
                                                       delta_norm + distance_to_boundary))

            # clip epsilon
            self.epsilon = torch.minimum(self.epsilon, self.worst_norm)

            delta_optim.step()

            # project in place
            self.projection(delta=self.delta.data, epsilon=self.epsilon)

            # clamp
            self.delta.data.add_(self.inputs).clamp_(min=0, max=1).sub_(self.inputs)

            delta_scheduler.step()

            # mostrare epsilon al variare delle iterazioni
            # printare la loss
            # printare la distanza

        plot_loss_epsilon_over_steps(
            loss=loss_per_iter,
            epsilon=epsilon_per_iter,
            steps=self.steps
        )

        return self.best_adv
