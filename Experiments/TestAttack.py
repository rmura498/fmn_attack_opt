from abc import ABC, abstractmethod


class TestAttack(ABC):
    def __init__(self,
                 model,
                 dataset,
                 attack,
                 norm,
                 steps,
                 batch_size,
                 optimizer=None,
                 scheduler=None):
        self.model = model
        self.dataset = dataset
        self.attack = attack
        self.norm = norm
        self.steps = steps
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass
