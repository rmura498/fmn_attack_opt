from abc import ABC, abstractmethod


class TestAttack(ABC):
    def __init__(self, model, dataset, attack, batch_size, optimizer, scheduler):
        self.model = model
        self.dataset = dataset
        self.attack = attack
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def accuracy(self):
        pass
