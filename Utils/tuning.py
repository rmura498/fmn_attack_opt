import ray
from ray.air import session
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

from Experiments.TestFMNAttack import TestFMNAttack
from optuna.distributions import FloatDistribution

import torch
from robustbench.utils import load_model
from Utils.datasets import load_dataset
from Attacks.FMN.FMNOpt import FMNOpt

ray.init()

search_space = {
    'lr': FloatDistribution(low=0.5, high=1),
    'momentum': FloatDistribution(0, 1),
    'weight_decay': FloatDistribution(0, 1),
    'dampening': FloatDistribution(0, 1)
}

model = load_model(
    model_dir="../Models/pretrained",
    model_name='Gowal2021Improving_28_10_ddpm_100m',
    dataset='cifar10',
    norm='Linf'
)
dataset = load_dataset('cifar10')

attack_params = {
    'batch_size': 5,
    'norm': float('inf'),
    'steps': 10,
    'optimizer': 'SGD'
}
dl_test = torch.utils.data.DataLoader(dataset,
                                      batch_size=attack_params['batch_size'],
                                      shuffle=False)


def objective(config, model, dl_test):
    for x in range(2):
        samples, labels = next(iter(dl_test))
        attack = FMNOpt(
            model=model,
            inputs=samples.clone(),
            labels=labels.clone(),
            norm=attack_params['norm'],
            steps=attack_params['steps']
        )
        score, best_loss = attack.run(config=config)
        session.report({"best_loss": best_loss})


trainable_with_resources = tune.with_resources(
    tune.with_parameters(
        objective,
        dl_test=dl_test,
        model=model
    ),
    resources={
        'cpu': 5,
        'cpu':5
    }
)

scheduler = ASHAScheduler(mode="min", metric="best_loss", grace_period=2)
scheduler_hyper = HyperBandScheduler(mode="min", metric="best_loss")
optuna_search = OptunaSearch(space=search_space, mode='min', metric='best_loss')
# bayes_search = BayesOptSearch(space=search_space, mode='min', metric='best_loss')

tuner = tune.Tuner(trainable_with_resources, param_space=search_space,
                   tune_config=tune.TuneConfig(num_samples=10, scheduler=scheduler,
                                               search_alg=optuna_search),
                   run_config=ray.air.RunConfig(verbose=3))
results = tuner.fit()
best_result = results.get_best_result(metric="best_loss", mode='min')
best_config = best_result.config
print(f"best_result : {best_result}\n, best config : {best_config}\n")
