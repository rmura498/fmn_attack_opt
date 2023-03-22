from .TestAttack import TestAttack
from Experiments.TestAutoAttack import TestAutoAttack
from Attacks.FMN.FMNOpt import FMNOpt
from Attacks.FMN.FMNBase import FMNBase

from Utils.metrics import accuracy
from Utils.plots import plot_loss_epsilon_over_steps

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


class TestFMNAttack(TestAttack):
    def __init__(self,
                 model,
                 dataset,
                 attack=FMNOpt,
                 norm=0,
                 steps=10,
                 batch_size=10,
                 optimizer=SGD,
                 scheduler=CosineAnnealingLR):
        super().__init__(
            model,
            dataset,
            attack,
            norm,
            steps,
            batch_size,
            optimizer,
            scheduler
        )

        self.dl_test = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        self.samples, self.labels = next(iter(self.dl_test))
        # TODO: create a function where samples, labels are dropped

        self.attack = self.attack(
            model=self.model,
            inputs=self.samples.clone(),
            labels=self.labels,
            norm=self.norm,
            steps=self.steps
        )

        if hasattr(self.attack, 'optimizer') and hasattr(self.attack, 'scheduler'):
            self.attack.optimizer = optimizer
            self.attack.scheduler = scheduler

    def run(self):
        advs = self.attack.run()

        standard_acc = accuracy(self.model, self.samples, self.labels)
        model_robust_acc = accuracy(self.model, advs, self.labels)
        print("Standard Accuracy", standard_acc)
        print("[FMN] Robust accuracy: ", model_robust_acc)

    def plot(self, normalize=True, translate_loss=True, translate_distance=True):
        plot_loss_epsilon_over_steps(
            self.attack.loss_per_iter,
            self.attack.epsilon_per_iter,
            distance_to_boundary=self.attack.distance_to_boundary_per_iter,
            steps=self.steps,
            norm=self.norm,
            attack_name=self.attack.__class__.__name__,
            model_name=self.model.__class__.__name__,
            normalize=normalize,
            translate_loss=translate_loss,
            translate_distance=translate_distance
        )

