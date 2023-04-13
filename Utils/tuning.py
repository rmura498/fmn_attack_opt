import ray
from ray.air import session
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

from Experiments.TestFMNAttack import TestFMNAttack
from Attacks.FMN.FMNOpt import FMNOpt

from robustbench.utils import load_model
from Utils.datasets import load_dataset

space = {
    'lr': tune.uniform(1, 0)
}


def trainable(space):
    model = load_model(
        model_dir="./Models/pretrained",
        model_name='Gowal2021Improving_28_10_ddpm_100m',
        dataset='cifar10',
        norm='Linf'
    )
    dataset = load_dataset('cifar10')

    attack_params = {
        'batch_size': 50,
        'norm': float('inf'),
        'steps': 20,
        'optimizer': 'SGD'
    }

    for x in range(20):
        exp = TestFMNAttack(model,
                            dataset=dataset,
                            attack=FMNOpt,
                            steps=attack_params['steps'],
                            norm=attack_params['norm'],
                            batch_size=attack_params['batch_size'],
                            optimizer=attack_params['optimizer'],
                            alpha_init=space['lr'])
        best_loss = exp.run()

        session.report({"Best loss": best_loss})  # Send the score to Tune.


if __name__ == '__main__':

    tuner = tune.Tuner(trainable, param_space=space)
    tuner.fit()
