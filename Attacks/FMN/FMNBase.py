import math
import torch
from torch import nn, Tensor
from torch.autograd import grad

from functools import partial
from typing import Optional

from Attacks.Attack import Attack
from Utils.projections import l0_projection_, l1_projection_, l2_projection_, linf_projection_
from Utils.projections import l0_mid_points, l1_mid_points, l2_mid_points, linf_mid_points
from Utils.metrics import difference_of_logits


class FMNBase(Attack):

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
                 binary_search_steps: int = 10
                 ):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.norm = norm
        self.targeted = targeted
        self.steps = steps
        self.alpha_init = alpha_init
        self.alpha_final = self.alpha_init / 100 if alpha_final is None else alpha_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps

        self.device = self.inputs.device
        self.batch_size = len(self.inputs)
        self.batch_view = lambda tensor: tensor.view(self.batch_size, *[1] * (self.inputs.ndim - 1))

        self._dual_projection_mid_points = {
            0: (None, l0_projection_, l0_mid_points),
            1: (float('inf'), l1_projection_, l1_mid_points),
            2: (2, l2_projection_, l2_mid_points),
            float('inf'): (1, linf_projection_, linf_mid_points),
        }

        _worst_norm = torch.maximum(inputs, 1 - inputs).flatten(1).norm(p=norm, dim=1)
        self.init_trackers = {
            'worst_norm': _worst_norm,
            'best_norm': _worst_norm.clone(),
            'best_adv': self.inputs.clone(),
            'adv_found': torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        }

        self.epsilon_per_iter = []
        self.delta_per_iter = []

        self.best_adv = None

    def _boundary_search(self):
        _, _, mid_point = self._dual_projection_mid_points[self.norm]

        is_adv = self.model(self.starting_points).argmax(dim=1)
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(self.batch_size, device=self.device)
        upper_bound = torch.ones(self.batch_size, device=self.device)
        for _ in range(self.binary_search_steps):
            epsilon = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=self.inputs, x1=self.starting_points, epsilon=epsilon)
            pred_labels = self.model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
            lower_bound = torch.where(is_adv, lower_bound, epsilon)
            upper_bound = torch.where(is_adv, epsilon, upper_bound)

        delta = mid_point(x0=self.inputs, x1=self.starting_points, epsilon=epsilon) - self.inputs

        return epsilon, delta, is_adv

    def _init_attack(self):
        epsilon = None
        delta = None
        is_adv = None

        if self.starting_points is not None:
            epsilon, delta, is_adv = self._boundary_search()
        else:
            delta = torch.zeros_like(self.inputs)

        delta.requires_grad_()

        return epsilon, delta, is_adv

    def run(self):
        epsilon, delta, is_adv = self._init_attack()
        multiplier = 1 if self.targeted else -1

        dual, projection, mid_point = self._dual_projection_mid_points[self.norm]

        for i in range(self.steps):
            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            alpha = self.alpha_final + (self.alpha_init - self.alpha_final) * cosine
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_inputs = self.inputs + delta
            logits = self.model(adv_inputs)
            pred_labels = logits.argmax(dim=1)

            if i == 0:
                labels_infhot = torch.zeros_like(logits).scatter_(1, self.labels.unsqueeze(1), float('inf'))
                logit_diff_func = partial(difference_of_logits, labels=self.labels, labels_infhot=labels_infhot)

            logit_diffs = logit_diff_func(logits=logits)
            loss = (multiplier * logit_diffs)

            delta_grad = grad(loss.sum(), delta, only_inputs=True)[0]

            is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
            is_smaller = delta_norm < best_norm
            is_both = is_adv & is_smaller
            self.init_trackers['adv_found'].logical_or_(is_adv)
            best_norm = torch.where(is_both, delta_norm, best_norm)
            best_adv = torch.where(self.batch_view(is_both), adv_inputs.detach(), best_adv)

            if self.norm == 0:
                epsilon = torch.where(is_adv,
                                      torch.minimum(torch.minimum(epsilon - 1,
                                                                  (epsilon * (1 - gamma)).floor_()),
                                                    best_norm),
                                      torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
                epsilon.clamp_(min=0)
            else:
                distance_to_boundary = loss.detach().abs() / delta_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
                epsilon = torch.where(is_adv,
                                      torch.minimum(epsilon * (1 - gamma), best_norm),
                                      torch.where(self.init_trackers['adv_found'],
                                                  epsilon * (1 + gamma),
                                                  delta_norm + distance_to_boundary)
                                      )

            # clip epsilon
            epsilon = torch.minimum(epsilon, self.init_trackers['worst_norm'])

            # normalize gradient
            grad_l2_norms = delta_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)
            delta_grad.div_(self.batch_view(grad_l2_norms))

            # gradient ascent step
            delta.data.add_(delta_grad, alpha=alpha)

            # project in place
            projection(delta=delta.data, epsilon=epsilon)

            # clamp
            delta.data.add_(self.inputs).clamp_(min=0, max=1).sub_(self.inputs)
