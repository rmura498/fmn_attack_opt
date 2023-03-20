from TestAttack import TestAttack
from Utils.metrics import accuracy
from Utils.plots import plot_loss_epsilon_over_steps
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

class TestFMNAttack(TestAttack):
    def __init__(self,
                 model,
                 dataset,
                 attack,
                 batch_size=10,
                 optimizer=SGD,
                 scheduler=CosineAnnealingLR):
        super().__init__(model, dataset, attack)

        #TODO: initialize samples and labels

        # self.samples
        # self.labels

    def run(self):
        # run the attack
        self.attack.run()
        pass

    def plot(self):
        plot_loss_epsilon_over_steps(self.model.loss_per_iter, self.model.epsilon_per_iter, self.model.steps, normalize=False)

    def accuracy(self):
        return accuracy(self.model, self.samples, self.labels)


