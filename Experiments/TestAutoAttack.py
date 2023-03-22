from .TestAttack import TestAttack
from autoattack import AutoAttack

import torch
import os

from Utils.metrics import accuracy


class TestAutoAttack(TestAttack):
    def __init__(self,
                 model,
                 dataset,
                 attack=['apgd-ce', 'apgd-dlr'],
                 norm=2,
                 steps=10,
                 batch_size=10,
                 optimizer=None,
                 scheduler=None):
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

        _norm_conv = {
            '1': 'L1',
            '2': 'L2',
            'inf': 'Linf'
        }
        self.norm = _norm_conv[str(self.norm)]

        self.standard_accuracy = None
        self.robust_accuracy = None

    def run(self):
        adversary = AutoAttack(self.model, norm=self.norm,
                               eps=8 / 255, version='custom',
                               attacks_to_run=self.attack, device='cpu')
        adversary.apgd.n_restarts = 1
        advs = adversary.run_standard_evaluation(self.samples, self.labels)

        standard_acc = accuracy(self.model, self.samples, self.labels)
        model_robust_acc = accuracy(self.model, advs, self.labels)
        print("[AA] Robust accuracy: ", model_robust_acc)

        self.standard_accuracy = standard_acc
        self.robust_accuracy = model_robust_acc

    def plot(self):
        pass

    def save_data(self):
        _data = [
            f"Steps: {self.steps}\n",
            f"Batch size: {self.batch_size}\n",
            f"Norm: {self.norm}\n",
            f"Standard acc: {self.standard_accuracy}\n"
            f"Robust acc: {self.robust_accuracy}\n",
        ]

        print("Saving experiment data...")
        data_file_path = os.path.join(self.exp_path, "data.txt")
        with open(data_file_path, "w+") as file:
            file.writelines(_data)
