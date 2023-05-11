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
                 create_exp_folder=True,
                 loss='LL',
                 model_name=''):
        self.model = model
        self.dataset = dataset
        self.attack = attack
        self.norm = norm
        self.steps = steps
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_path = None
        self.create_exp_folder = create_exp_folder
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.model_name = model_name

        # Create experiment folder
        if self.create_exp_folder:
            if self.model_name == '':
                self.model_name = self.model.__class__.__name__
            self.dataset_name = self.dataset.__class__.__name__

            # time = datetime.now().strftime("%d%H%M")
            experiment = f'{self.model_name}_{self.dataset_name}_{self.optimizer_name}_{self.scheduler_name}_{loss}'
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
