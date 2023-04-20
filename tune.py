import os
import math
import argparse
import json
from datetime import datetime

import torch
from ray import air
from ray.air import session
from ray import tune
from ray.tune.search.flaml import CFO
from ray.tune.schedulers import ASHAScheduler

from Attacks.FMN.FMNOptTune import FMNOptTune
from Models.load_data import load_data
from Configs.tuning_resources import TUNING_RES
from Configs.search_spaces_tune import OPTIMIZERS_SEARCH_TUNE, SCHEDULERS_SEARCH_TUNE

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
parser.add_argument('-wp', '--working_path',
                    default='./TuningExp')

args = parser.parse_args()


def tune_attack(config, model, samples, labels, attack_params, epochs=5):
    for epoch in range(epochs):
        attack = FMNOptTune(
            model=model,
            inputs=samples.clone(),
            labels=labels.clone(),
            norm=attack_params['norm'],
            steps=attack_params['steps'],
            optimizer=attack_params['optimizer'],
            scheduler=attack_params['scheduler'],
            optimizer_config=config['opt_s'],
            scheduler_config=config['sch_s']
        )

        distance, _ = attack.run()
        session.report({"distance": distance})


if __name__ == '__main__':
    optimizer = args.optimizer
    scheduler = args.scheduler
    model_id = int(args.model_id)
    dataset_id = int(args.dataset_id)
    dataset_percent = float(args.dataset_percent)
    working_path = args.working_path

    # load search spaces
    optimizer_search = OPTIMIZERS_SEARCH_TUNE[optimizer]
    scheduler_search = SCHEDULERS_SEARCH_TUNE[scheduler]

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

    steps_keys = ['T_max', 'T_0', 'milestones']
    for key in steps_keys:
        if key in scheduler_search:
            scheduler_search[key] = scheduler_search[key](attack_params['steps'])
    search_space = {
        'opt_s': optimizer_search,
        'sch_s': scheduler_search
    }

    trainable_with_resources = tune.with_resources(
        tune.with_parameters(
            tune_attack,
            model=model,
            samples=samples,
            labels=labels,
            attack_params=attack_params,
            epochs=tune_config['epochs']
        ),
        resources=TUNING_RES
    )

    tune_scheduler = ASHAScheduler(mode='min', metric='distance', grace_period=2)
    algo = CFO(metric='distance', mode='min')

    # ./TuningExp/Modelname_dataset/...
    time = datetime.now().strftime("%d%H%M")
    tuning_exp_name = f"{optimizer}_{scheduler}_{time}"
    tuning_exp_path = os.path.join(working_path, tuning_exp_name)
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config['num_samples'],
            search_alg=algo,
            scheduler=tune_scheduler
        ),
        run_config=air.RunConfig(tuning_exp_name, local_dir=working_path)
    )

    results = tuner.fit()

    # Checking best result and best config
    best_result = results.get_best_result(metric='distance', mode='min')
    best_config = best_result.config
    print(f"best_distance : {best_result.metrics['distance']}\n, best config : {best_config}\n")

    best_result_json = {
        'distance': best_result.metrics['distance'],
        'best_config': best_result.config
    }

    best_result_json = json.dumps(best_result_json)

    filename = os.path.join(tuning_exp_path, "best_result.json")
    with open(filename, "w") as file:
        file.write(best_result_json)



