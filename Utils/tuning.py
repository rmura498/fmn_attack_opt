import ray
from ray.air import session
from ray import tune

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
from optuna.distributions import FloatDistribution

import torch
from robustbench.utils import load_model
from Utils.datasets import load_dataset
from Attacks.FMN.FMNOpt import FMNOpt

ray.init()

search_space = {
    'lr': FloatDistribution(0.5, 1, log=True),
    'momentum': FloatDistribution(0.8, 0.9),
    'dampening': FloatDistribution(0, 0.2)
}

model = load_model(
    model_dir="../Models/pretrained",
    model_name='Gowal2021Improving_R18_ddpm_100m',
    dataset='cifar10',
    norm='Linf'
)
dataset = load_dataset('cifar10')

attack_params = {
    'batch_size': 30,
    'norm': float('inf'),
    'steps': 20,
    'optimizer': 'SGD'
}
dl_test = torch.utils.data.DataLoader(dataset,
                                      batch_size=attack_params['batch_size'],
                                      shuffle=False)
samples, labels = next(iter(dl_test))


def objective(config, model, samples, labels):
    for epoch in range(10):
        attack = FMNOpt(
            model=model,
            inputs=samples.clone(),
            labels=labels.clone(),
            norm=attack_params['norm'],
            steps=attack_params['steps']
        )
        distance, _ = attack.run(config=config)
        session.report({"distance": distance})


trainable_with_resources = tune.with_resources(
    tune.with_parameters(
        objective,
        model=model,
        samples=samples,
        labels=labels
    ),
    resources={
        'cpu': 6,
        'cpu': 6
    }
)

mode = 'min'
metric = 'distance'

scheduler = ASHAScheduler(mode=mode, metric=metric)
optuna_search = OptunaSearch(space=search_space, mode=mode, metric=metric)

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=20,
        search_alg=optuna_search,
        scheduler=scheduler
    )
)

results = tuner.fit()

best_result = results.get_best_result(metric=metric, mode=mode)
best_config = best_result.config
print(f"best_result : {best_result}\n, best config : {best_config}\n")
