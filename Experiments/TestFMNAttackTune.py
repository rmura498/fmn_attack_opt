import os.path
import pickle, math

import torch

from .TestAttack import TestAttack
from Attacks.FMN.FMNOptTuneSave import FMNOptTuneSave

from Utils.metrics import accuracy


class TestFMNAttackTune(TestAttack):
    def __init__(self,
                 model,
                 dataset,
                 model_name,
                 attack=FMNOptTuneSave,
                 norm='inf',
                 steps=10,
                 batch_size=10,
                 optimizer='SGD',
                 scheduler='CosineAnnealingLR',
                 create_exp_folder=True,
                 optimizer_config=None,
                 scheduler_config=None,
                 loss = 'LL',
                 device=torch.device('cpu'),
                 tuning_dataset_percent=0.5
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
            create_exp_folder,
            loss=loss,
            model_name=model_name,
        )
        self.device = device
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # load only the test part of the dataset (e.g. 50% was used for tuning, rest is for testing/val purposes)
        dataset_frac = list(range(math.floor(len(dataset) * tuning_dataset_percent)+1, len(dataset)))
        dataset_frac = torch.utils.data.Subset(dataset, dataset_frac)

        self.dl_test = torch.utils.data.DataLoader(dataset_frac,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)
        self.dl_test_iter = iter(self.dl_test)

        # self.samples, self.labels = next(iter(self.dl_test))
        self.loss = True if loss == 'LL' else False

        #self.samples.to(self.device)
        #self.labels.to(self.device)

        self.attacks = []
        for n in range(10):
            samples, labels = next(self.dl_test_iter)
            samples.to(self.device)
            labels.to(self.device)

            attack = self.attack(
                model=self.model,
                inputs=samples,
                labels=labels,
                norm=self.norm,
                steps=self.steps,
                optimizer=self.optimizer_name,
                scheduler=self.scheduler_name,
                optimizer_config=self.optimizer_config,
                scheduler_config=self.scheduler_config,
                logit_loss = self.loss,
                device=self.device
            )
            self.attacks.append(attack)
            del attack

        self.standard_accuracy = None
        self.robust_accuracy = None

        self.best_adv = None

    def run(self):
        for attack in self.attacks:
            _, _ = attack.run()

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

        attack_data = {
            'epsilon': [],
            'pred_labels': [],
            'distance': [],
            'inputs': [],
            'labels' : [],
            'best_adv': []
        }
        for attack in self.attacks:
            for data_key in attack.attack_data:
                attack_data[data_key].append(attack.attack_data[data_key])

        for attack_list in attack_data:
            data_path = os.path.join(self.exp_path, f"{attack_list}.pkl")
            torch.save(attack_data[attack_list], data_path)
