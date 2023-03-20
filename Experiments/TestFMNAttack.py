from .TestAttack import TestAttack

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

        self.dl_test = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
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

        model_robust_acc = accuracy(self.model, advs, self.labels)
        print("Robust accuracy: ", model_robust_acc)

    def plot(self):
        plot_loss_epsilon_over_steps(
            self.attack.loss_per_iter,
            self.attack.epsilon_per_iter,
            self.steps,
            norm=self.norm,
            attack_name=self.attack.__class__.__name__,
            model_name=self.model.__class__.__name__,
            normalize=True
        )

