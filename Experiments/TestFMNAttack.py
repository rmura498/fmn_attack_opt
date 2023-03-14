from TestAttack import TestAttack


class TestFMNAttack(TestAttack):
    def __init__(self, model, dataset, attack):
        super().__init__(model, dataset, attack)

    def run(self):
        # run the attack
        self.attack.run()
        pass

    def plot(self):
        # execute the plotting functions
        self._plot_loss_over_iters()
        self._plot_epsilon_over_iters()

    def _plot_loss_over_iters(self):
        # there should be a reference on the loss per iteration computed during the attack run
        pass

    def _plot_epsilon_over_iters(self):
        pass

