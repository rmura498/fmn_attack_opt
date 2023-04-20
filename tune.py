import math
import argparse

import torch
from ray import air
from ray.air import session
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.flaml import CFO
from ray.tune.schedulers import ASHAScheduler
from optuna.samplers import TPESampler

from Attacks.FMN.FMNOptTune import FMNOptTune
from Models.load_data import load_data
from Configs.tuning_resources import TUNING_RES
# from Configs.search_spaces_optuna import OPTIMIZERS_SEARCH_OPTUNA, SCHEDULERS_SEARCH_OPTUNA
from Configs.search_spaces_tune import OPTIMIZERS_SEARCH_AX, SCHEDULERS_SEARCH_AX

parser = argparse.ArgumentParser(description='Retrieve tuning params')
parser.add_argument('-opt', '--optimizer',
                    default='SGD',
                    help='Provide the optimizer name (e.g. SGD, Adam)')
parser.add_argument('-sch', '--scheduler',
                    default='CosineAnnealingLR',
                    help='Provide the scheduler name (e.g. CosineAnnealingLR)')
parser.add_argument('-m_id', '--model_id',
                    default=0,
                    help='Provide the model ID')
parser.add_argument('-d_id', '--dataset_id',
                    default=0,
                    help='Provide the dataset ID (e.g. CIFAR10: 0)')
parser.add_argument('-b', '--batch',
                    default=50,
                    type=int,
                    help='Provide the batch size (e.g. 50, 100, 10000)')
parser.add_argument('-s', '--steps',
                    default=10,
                    type=int,
                    help='Provide the steps number (e.g. 50, 100, 10000)')
parser.add_argument('-n', '--norm',
                    default='inf',
                    help='Provide the norm (0, 1, 2, inf)')
parser.add_argument('-n_sample', '--num_samples',
                    default=5,
                    help='Provide the number of samples for the tuning')
parser.add_argument('-ep', '--epochs',
                    default=5,
                    help='Provide the number of epochs for the attack tuning \
                    (how many times the attack is run by the tuner)')
parser.add_argument('-dp', '--dataset_percent',
                    default=0.5,
                    help='Provide the percentage of test dataset to be used to tune the hyperparams (default: 0.5)')

args = parser.parse_args()


def objective(config, model, samples, labels, attack_params, epochs=5):
    for epoch in range(epochs):
        attack = FMNOptTune(
            model=model,
            inputs=samples.clone(),
            labels=labels.clone(),
            norm=attack_params['norm'],
            steps=attack_params['steps'],
            optimizer=attack_params['optimizer'],
            scheduler=attack_params['scheduler'],
            optimizer_config=config['optimizer_search'],
            scheduler_config=config['scheduler_search']
        )

        distance, _ = attack.run()
        session.report({"distance": distance})


if __name__ == '__main__':
    optimizer = args.optimizer
    scheduler = args.scheduler
    model_id = args.model_id
    dataset_id = args.dataset_id
    dataset_percent = args.dataset_percent

    attack_params = {
        'batch': int(args.batch),
        'steps': int(args.steps),
        'norm': args.norm,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    tune_config = {
        'num_samples': int(args.num_samples),
        'epochs': int(args.epochs)
    }

    # load model and dataset
    model, dataset = load_data(model_id, dataset_id, attack_params['norm'])

    dataset_frac = list(range(0, math.floor(len(dataset)*dataset_percent)))
    dataset_frac = torch.utils.data.Subset(dataset, dataset_frac)

    dl_test = torch.utils.data.DataLoader(dataset_frac,
                                          batch_size=attack_params['batch'],
                                          shuffle=False)
    samples, labels = next(iter(dl_test))

    # load search spaces
    optimizer_search = OPTIMIZERS_SEARCH_AX[optimizer]
    scheduler_search = SCHEDULERS_SEARCH_AX[scheduler]

    steps_keys = ['T_max', 'T_0', 'milestones']

    for key in steps_keys:
        if key in scheduler_search:
            scheduler_search[key] = scheduler_search[key](attack_params['steps'])
    search_space = {
        'optimizer_search': optimizer_search,
        'scheduler_search': scheduler_search
    }

    trainable_with_resources = tune.with_resources(
        tune.with_parameters(
            objective,
            model=model,
            samples=samples,
            labels=labels,
            attack_params=attack_params,
            epochs=tune_config['epochs']
        ),
        resources=TUNING_RES
    )

    scheduler = ASHAScheduler(mode='min', metric='distance', grace_period=2)
    # algo = OptunaSearch(space=search_space, mode='min', metric='distance', sampler=TPESampler())
    # algo = BayesOptSearch(metric="distance", mode="min")
    # algo = AxSearch(mode='min', metric='distance')
    # algo = TuneBOHB(metric="distance", mode="min")
    algo = CFO(metric='distance', mode='min')

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config['num_samples'],
            search_alg=algo,
            scheduler=scheduler
        ),
        run_config=air.RunConfig(local_dir="./TuningExp")
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric='distance', mode='min')
    best_config = best_result.config
    print(f"best_result : {best_result}\n, best config : {best_config}\n")



