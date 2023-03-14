from abc import ABC, abstractmethod


class TestAttack(ABC):
    def __init__(self, model, dataset, attack):
        self.model = model
        self.dataset = dataset
        self.attack = attack

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass
