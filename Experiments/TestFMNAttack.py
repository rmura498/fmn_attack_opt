from TestAttack import TestAttack
from Utils.metrics import accuracy
from Utils.plots import plot_loss_epsilon_over_steps
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch

class TestFMNAttack(TestAttack):
    def __init__(self,
                 model,
                 dataset,
                 attack,
                 batch_size=10,
                 optimizer=SGD,
                 scheduler=CosineAnnealingLR):
        super().__init__(model, dataset, attack, batch_size, optimizer, scheduler)
        self.dl_test = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.samples, self.labels = next(iter(self.dl_test))




    def run(self):
        # run the attack
        self.attack.run()
        pass

    def plot(self):
        plot_loss_epsilon_over_steps(self.model.loss_per_iter,
                                     self.model.epsilon_per_iter,
                                     self.model.steps, normalize=False)

    def accuracy(self):
        return accuracy(self.model, self.samples, self.labels)


