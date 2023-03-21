from .TestAttack import TestAttack
from autoattack import AutoAttack

import torch

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

    def run(self):
        adversary = AutoAttack(self.model, norm=self.norm,
                               eps=8 / 255, version='custom',
                               attacks_to_run=self.attack, device='cpu')
        adversary.apgd.n_restarts = 1
        advs = adversary.run_standard_evaluation(self.samples, self.labels)
        model_robust_acc = accuracy(self.model, advs, self.labels)
        print("[AA] Robust accuracy: ", model_robust_acc)

    def plot(self):
        pass
