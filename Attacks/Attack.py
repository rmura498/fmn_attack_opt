from abc import ABC, abstractmethod


class Attack(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _init_attack(self):
        pass

    @abstractmethod
    def run(self, config):
        pass

