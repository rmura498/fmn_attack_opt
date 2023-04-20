import math

import torch
from torch import nn, Tensor

from functools import partial
from typing import Optional, Union

from Attacks.FMN.FMNOptTune import FMNOptTune
from Utils.metrics import difference_of_logits


class FMNOptTuneSave(FMNOptTune):

    def __init__(self,
                 model: nn.Module,
                 inputs: Tensor,
                 labels: Tensor,
                 norm: Union[str, float],
                 targeted: bool = False,
                 steps: int = 10,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 starting_points: Optional[Tensor] = None,
                 binary_search_steps: int = 10,
                 optimizer='SGD',
                 scheduler='CosineAnnealingLR',
                 optimizer_config=None,
                 scheduler_config=None
                 ):
        super().__init__(
            model,
            inputs,
            labels,
            norm,
            targeted,
            steps,
            gamma_init,
            gamma_final,
            starting_points,
            binary_search_steps,
            optimizer,
            scheduler,
            optimizer_config,
            scheduler_config
        )

        self.attack_data = {
            'epsilon': [],
            'pred_labels': [],
            'distance': [],
            'inputs': [],
            'best_adv': []
        }

    def run(self):
        dual, projection, _ = self._dual_projection_mid_points[self.norm]

        epsilon, delta, is_adv = self._init_attack()
        multiplier = 1 if self.targeted else -1

        delta.requires_grad_(True)

        # Initialize optimizer and scheduler
        self._init_optimizer(objective=delta)
        self._init_scheduler()

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

            # saving epsilon and distance
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
            # Computing the best distance (x-x0 for the adversarial) ~ should be equal to delta
            _distance = torch.linalg.norm((self.init_trackers['best_adv'] - self.inputs).data.flatten(1),
                                          dim=1, ord=self.norm)

            if self.scheduler_name == 'ReduceLROnPlateau':
                self._scheduler_step(torch.median(_distance).item())
            else:
                self._scheduler_step()

            # Saving data
            self.attack_data['epsilon'].append(_epsilon)
            self.attack_data['distance'].append(_distance)

            del _epsilon

        logits = self.model(self.init_trackers['best_adv'])
        pred_labels = logits.argmax(dim=1)

        # Storing best adv labels (perturbed one)
        self.attack_data['pred_labels'].append(pred_labels)

        # Storing best adv
        self.attack_data['best_adv'] = self.init_trackers['best_adv'].clone()

        return torch.median(_distance).item(), self.init_trackers['best_adv']
