from TestAttack import TestAttack
import numpy as np
import matplotlib.pyplot as plt
import torch


class TestFMNAttack(TestAttack):
    def __init__(self, model, dataset, attack):
        super().__init__(model, dataset, attack)

    def run(self):
        # run the attack
        self.attack.run()
        pass

    def plot(self):
        # execute the plotting functions
        self._plot_loss_epsilon_over_iters()

    def _plot_loss_epsilon_over_iters(self):
        fig1, ax1 = plt.subplots()
        ax1.plot(
            torch.arange(0, self.model.steps),
            self.model.loss_per_iter,
            label="Loss"
        )
        ax1.plot(
            torch.arange(0, self.model.steps),
            self.model.epsilon_per_iter,
            label="Epsilon"
        )
        ax1.grid()
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss/Epsilon")
        fig1.legend()
        plt.show()
