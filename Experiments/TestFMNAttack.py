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

        self._optimizers = {
            'SGD': SGD,
            'Adam': Adam,
            'Adagrad': Adagrad,
            'Adadelta': Adadelta
        }

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
            self.attack.optimizer = self._optimizers[self.optimizer_name]
            self.attack.scheduler = self._schedulers[self.scheduler_name]

        self.standard_accuracy = None
        self.robust_accuracy = None

        self.best_adv = None

    """
best_result : Result(metrics={'distance': 0.04519006609916687, 'done': True, 'trial_id': '3a08bb24', 'experiment_tag': '18_dampening=0.1091,lr=0.9957,momentum=0.8017'}, error=None, log_dir=PosixPath('/home/lucas/ray_results/objective_2023-04-17_18-21-28/objective_3a08bb24_18_dampening=0.1091,lr=0.9957,momentum=0.8017_2023-04-17_18-28-00'))
, best config : {'lr': 0.995714859948475, 'momentum': 0.8017060210618113, 'dampening': 0.10910823490475516}
    """

    def run(self):
        config = {'lr': 5.349337273090887, 'eps': 4.4388051509281266e-08, 'amsgrad': False}
        distance, self.best_adv = self.attack.run(
            config
            # config = {'lr': 1.393421661847915, 'momentum': 0.9252981464345622, 'dampening': 0.016218714123266566, 'T_max': 37}
            # Configs = {'lr': 1.366176145774935, 'momentum': 0.9182007021137776, 'dampening': 0.011616722433639893, 'T_max': 21}
            # Configs = {'lr': 0.7907505200712125, 'momentum': 0.8646091064266745, 'dampening': 0.1736640884093873}
            # Configs={'lr': 1.3565079026428308, 'momentum': 0.89227479234969, 'dampening': 0.06903604081398133}
        )

        standard_acc = accuracy(self.model, self.samples, self.labels)
        model_robust_acc = accuracy(self.model, self.best_adv, self.labels)
        print("Standard Accuracy", standard_acc)
        print("[FMN] Robust accuracy: ", model_robust_acc)

        self.standard_accuracy = standard_acc
        self.robust_accuracy = model_robust_acc
        return self.best_adv

    def plot(self):
        pass

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
