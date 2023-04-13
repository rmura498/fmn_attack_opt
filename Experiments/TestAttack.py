import os
from datetime import datetime
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
                 scheduler=None,
                 AA=False,
                 create_exp_folder=True):
        self.model = model
        self.dataset = dataset
        self.attack = attack
        self.norm = norm
        self.steps = steps
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.AA = AA
        self.exp_path = None
        self.create_exp_folder = create_exp_folder
        # Create experiment folder
        if self.create_exp_folder:
            if not AA:
                self.attack_name = self.attack.__name__
            else:
                self.attack_name = '_'.join(param for param in self.attack)
            self.model_name = self.model.__class__.__name__
            self.dataset_name = self.dataset.__class__.__name__

            time = datetime.now().strftime("%d%H%M")
            if not self.AA:
                experiment = f'Exp_{time}_{self.attack_name}_{self.model_name}_{self.dataset_name}'
            else:
                experiment = f'AA_Exp_{time}_{self.attack_name}_{self.model_name}_{self.dataset_name}'
            path = os.path.join("Experiments", experiment)
            if not os.path.exists(path):
                os.makedirs(path)

            self.exp_path = path

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save_data(self):
        pass
