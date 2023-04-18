import os.path
import pickle

from .TestAttack import TestAttack
from Experiments.TestAutoAttack import TestAutoAttack
from Attacks.FMN.FMNOpt import FMNOpt
from Attacks.FMN.FMNBase import FMNBase

from Utils.metrics import accuracy

import torch
from torch.optim import SGD, Adam, Adagrad, Adadelta
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import numpy as np


class TestFMNAttack(TestAttack):
    def __init__(self,
                 model,
                 dataset,
                 attack=FMNOpt,
                 norm=0,
                 steps=10,
                 batch_size=10,
                 optimizer='SGD',
                 scheduler='CosineAnnealingLR',
                 create_exp_folder=True,
                 alpha_init=1
                 ):
        super().__init__(
            model,
            dataset,
            attack,
            norm,
            steps,
            batch_size,
            optimizer,
            scheduler,
            create_exp_folder
        )

        self.optimizer_name = optimizer
        self.scheduler_name = scheduler

        self._schedulers = {
            'CosineAnnealingLR': CosineAnnealingLR,
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts
        }

        self.dl_test = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)
        self.samples, self.labels = next(iter(self.dl_test))

        self.attack = self.attack(
            model=self.model,
            inputs=self.samples.clone(),
            labels=self.labels,
            norm=self.norm,
            steps=self.steps,
            alpha_init=alpha_init,
            save_data=True
        )

        if hasattr(self.attack, 'optimizer') and hasattr(self.attack, 'scheduler'):
            self.attack.optimizer = self.optimizer
            self.attack.scheduler = self._schedulers[self.scheduler_name]

        self.standard_accuracy = None
        self.robust_accuracy = None

        self.best_adv = None

    """
best_result : Result(metrics={'distance': 0.04519006609916687, 'done': True, 'trial_id': '3a08bb24', 'experiment_tag': '18_dampening=0.1091,lr=0.9957,momentum=0.8017'}, error=None, log_dir=PosixPath('/home/lucas/ray_results/objective_2023-04-17_18-21-28/objective_3a08bb24_18_dampening=0.1091,lr=0.9957,momentum=0.8017_2023-04-17_18-28-00'))
, best config : {'lr': 0.995714859948475, 'momentum': 0.8017060210618113, 'dampening': 0.10910823490475516}
    """

    def run(self):
        distance, self.best_adv = self.attack.run(
            config={
                "lr": 0.7803572661328038,
                #"momentum": 0.9141681922121645,
                #"dampening": 0.0012792542095206721
            }
        )

        standard_acc = accuracy(self.model, self.samples, self.labels)
        model_robust_acc = accuracy(self.model, self.best_adv, self.labels)
        print("Standard Accuracy", standard_acc)
        print("[FMN] Robust accuracy: ", model_robust_acc)

        self.standard_accuracy = standard_acc
        self.robust_accuracy = model_robust_acc
        return self.best_adv

    def plot(self, normalize=True, translate_loss=True, translate_distance=True):
        pass
        '''
        plo_loss_epsilon_over_steps(
            self.attack.loss_per_iter,
            self.attack.epsilon_per_iter,
            distance_to_boundary=self.attack.distance_to_boundary_per_iter,
            steps=self.steps,
            batch_size=self.batch_size,
            norm=self.norm,
            attack_name=self.attack_name,
            model_name=self.model_name,
            optimizer=self.optimizer,
            normalize=normalize,
            path=self.exp_path
        )'''

    def save_data(self):
        _data = [
            f"Steps: {self.steps}\n",
            f"Batch size: {self.batch_size}\n",
            f"Norm: {self.norm}\n",
            f"Standard acc: {self.standard_accuracy}\n"
            f"Robust acc: {self.robust_accuracy}\n",
            f"Optimizer: {self.optimizer_name}\n",
            f"Scheduler: {self.scheduler_name}\n",
            f"Model: {self.model_name}\n"
        ]

        print("Saving experiment data...")
        data_path = os.path.join(self.exp_path, "data.txt")
        with open(data_path, "w+") as file:
            file.writelines(_data)

        # Save attack lists
        data_path = os.path.join(self.exp_path, "labels.pkl")
        with open(data_path, "wb") as file:
            pickle.dump(self.labels, file)

        for attack_list in self.attack.attack_data:
            data_path = os.path.join(self.exp_path, f"{attack_list}.pkl")

            with open(data_path, "wb") as file:
                pickle.dump(self.attack.attack_data[attack_list], file)
