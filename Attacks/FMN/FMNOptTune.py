import math
import torch
from torch import nn, Tensor
from torch.optim import SGD,Adam

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from functools import partial
from typing import Optional

from Attacks.Attack import Attack
from Utils.projections import l0_projection_, l1_projection_, l2_projection_, linf_projection_
from Utils.projections import l0_mid_points, l1_mid_points, l2_mid_points, linf_mid_points
from Utils.metrics import difference_of_logits


class FMNOpt(Attack):

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
                 optimizer=SGD,
                 scheduler=CosineAnnealingLR,
                 optimizer_config=None,
                 scheduler_config=None
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

        self.optimizer = optimizer
        self.scheduler = scheduler

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
        delta = torch.zeros_like(self.inputs)

        is_adv = None

        if self.starting_points is not None:
            epsilon, delta, is_adv = self._boundary_search()

        if self.norm == 0:
            epsilon = torch.ones(self.batch_size, device=self.device) if self.starting_points is None else delta.flatten(1).norm(p=0, dim=0)
        else:
            epsilon = torch.full((self.batch_size,), float('inf'), device=self.device)

        return epsilon, delta, is_adv

    def run(self, config):
        dual, projection, _ = self._dual_projection_mid_points[self.norm]

        epsilon, delta, is_adv = self._init_attack()
        multiplier = 1 if self.targeted else -1

        delta.requires_grad_(True)

        # Initialize optimizer and scheduler
        self.optimizer = self.optimizer([delta], **config)
        self.scheduler = self.scheduler(self.optimizer, **config)

        print(f"epsilon init: {epsilon}")
        print("Starting the attack...\n")
        for i in range(self.steps):
            print(f"Attack completion: {i / self.steps * 100:.2f}%")
            self.optimizer.zero_grad()

            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_inputs = self.inputs + delta

            logits = self.model(adv_inputs)
            pred_labels = logits.argmax(dim=1)

            if self.save_data:
                _epsilon = epsilon.clone()
                _distance = torch.linalg.norm((adv_inputs - self.inputs).data.flatten(1), dim=1, ord=self.norm)

            if i == 0:
                labels_infhot = torch.zeros_like(logits).scatter_(1, self.labels.unsqueeze(1), float('inf'))
                logit_diff_func = partial(difference_of_logits, labels=self.labels, labels_infhot=labels_infhot)

            logit_diffs = logit_diff_func(logits=logits)
            loss = -(multiplier * logit_diffs)
            loss.sum().backward()

            delta_grad = delta.grad.data

            is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
            is_smaller = delta_norm < self.init_trackers['best_norm']
            is_both = is_adv & is_smaller
            self.init_trackers['adv_found'].logical_or_(is_adv)
            self.init_trackers['best_norm'] = torch.where(is_both, delta_norm, self.init_trackers['best_norm'])
            self.init_trackers['best_adv'] = torch.where(self.batch_view(is_both), adv_inputs.detach(),
                                                         self.init_trackers['best_adv'])

            if self.norm == 0:
                epsilon = torch.where(is_adv,
                                      torch.minimum(torch.minimum(epsilon - 1,
                                                                  (epsilon * (1 - gamma)).floor_()),
                                                    self.init_trackers['best_norm']),
                                      torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
                epsilon.clamp_(min=0)
            else:
                distance_to_boundary = loss.detach().abs() / delta_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
                epsilon = torch.where(is_adv,
                                      torch.minimum(epsilon * (1 - gamma), self.init_trackers['best_norm']),
                                      torch.where(self.init_trackers['adv_found'],
                                                  epsilon * (1 + gamma),
                                                  delta_norm + distance_to_boundary)
                                      )

            # clip epsilon
            epsilon = torch.minimum(epsilon, self.init_trackers['worst_norm'])

            # normalize gradient
            grad_l2_norms = delta_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)
            delta_grad.div_(self.batch_view(grad_l2_norms))

            self.optimizer.step()

            # project in place
            projection(delta=delta.data, epsilon=epsilon)

            # clamp
            delta.data.add_(self.inputs).clamp_(min=0, max=1).sub_(self.inputs)

            self.scheduler.step()

            # Saving data
            if self.save_data:
                self.attack_data['epsilon'].append(_epsilon)
                self.attack_data['distance'].append(_distance)

                del _epsilon, _distance

        # Computing the best distance (x-x0 for the adversarial) ~ should be equal to delta
        _distance = torch.linalg.norm((self.init_trackers['best_adv'] - self.inputs).data.flatten(1),
                                      dim=1, ord=self.norm)
        # _distance = torch.linalg.norm(_distance, ord=self.norm).item()

        if self.save_data:
            # Storing best adv labels (perturbed one)
            self.attack_data['pred_labels'].append(pred_labels)

            # Storing best adv
            self.attack_data['best_adv'] = self.init_trackers['best_adv'].clone()

        return torch.median(_distance).item(), self.attack_data['best_adv']
