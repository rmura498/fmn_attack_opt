from TestAttack import TestAttack
from Utils.metrics import accuracy

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
        pass

    def accuracy(self):
        return accuracy(self.model, self.samples, self.labels)


