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
                 scheduler_config=None
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
            model_name=model_name
        )
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # load only the test part of the dataset (e.g. 50% was used for tuning, rest is for testing/val purposes)
        dataset_frac = list(range(math.floor(len(dataset) * tuning_dataset_percent)+1, len(dataset)))
        dataset_frac = torch.utils.data.Subset(dataset, dataset_frac)

        self.dl_test = torch.utils.data.DataLoader(dataset_frac,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        self.samples, self.labels = next(iter(self.dl_test))

        self.attack = self.attack(
            model=self.model,
            inputs=self.samples.clone(),
            labels=self.labels,
            norm=self.norm,
            steps=self.steps,
            optimizer=self.optimizer_name,
            scheduler=self.scheduler_name,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config
        )

        self.standard_accuracy = None
        self.robust_accuracy = None

        self.best_adv = None

    def run(self):
        distance, self.best_adv = self.attack.run()

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
