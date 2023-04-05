from .TestAttack import TestAttack
from autoattack import AutoAttack

import torch
import os
import pickle

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
            scheduler,
            AA=True
        )

        self.dl_test = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.samples, self.labels = next(iter(self.dl_test))

        _norm_conv = {
            '1': 'L1',
            '2': 'L2',
            'inf': 'Linf'
        }
        self.norm = _norm_conv[str(self.norm)]

        self.best_adv = None
        self.standard_accuracy = None
        self.robust_accuracy = None

        self.attack_data = {
            'epsilon': [],
            'pred_labels': [],
            'distance': [],
            'inputs': [],
            'best_adv': []
        }

        self.attack_data['inputs'] = self.samples.clone()

    def run(self):
        adversary = AutoAttack(self.model, norm=self.norm,
                               eps=8 / 255, version='custom',
                               attacks_to_run=self.attack, device='cpu')
        adversary.apgd.n_restarts = 1
        self.best_adv = adversary.run_standard_evaluation(self.samples, self.labels, bs=self.batch_size)

        logits = self.model(self.best_adv)
        pred_labels = logits.argmax(dim=1)

        self.attack_data['best_adv'] = self.best_adv.clone()
        self.attack_data['pred_labels'] = pred_labels.clone()

        standard_acc = accuracy(self.model, self.samples, self.labels)
        model_robust_acc = accuracy(self.model, self.best_adv, self.labels)
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
            f"Model: {self.model_name}\n"
        ]

        print("Saving experiment data...")
        data_file_path = os.path.join(self.exp_path, "data.txt")
        with open(data_file_path, "w+") as file:
            file.writelines(_data)

        for attack_list in self.attack_data:
            data_path = os.path.join(self.exp_path, f"{attack_list}.pkl")

            with open(data_path, "wb") as file:
                pickle.dump(self.attack_data[attack_list], file)

