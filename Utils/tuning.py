import ray
from ray.air import session
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

from Experiments.TestFMNAttack import TestFMNAttack
from Attacks.FMN.FMNOpt import FMNOpt

from robustbench.utils import load_model
from Utils.datasets import load_dataset

ray.init()

space = {
    'lr': tune.uniform(1, 0)
}


model = load_model(
        model_dir="../Models/pretrained",
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


def trainable(space):

    exp = TestFMNAttack(model,
                        dataset=dataset,
                        attack=FMNOpt,
                        steps=attack_params['steps'],
                        norm=attack_params['norm'],
                        batch_size=attack_params['batch_size'],
                        optimizer=attack_params['optimizer'],
                        alpha_init=space['lr'],
                        create_exp_folder=False)
    best_loss = exp.run()

    session.report({"Best loss": best_loss})  # Send the score to Tune.

    del exp


tuner = tune.Tuner(trainable, param_space=space, tune_config=tune.TuneConfig(num_samples=10))
tuner.fit()
