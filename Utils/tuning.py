from fmn_attack_opt.Configs.tuning_resources import TUNING_RES
import ray
from ray.air import session
from ray import tune

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
from optuna.distributions import FloatDistribution
from optuna.samplers import TPESampler

import torch
from robustbench.utils import load_model
from Utils.datasets import load_dataset
from Attacks.FMN.FMNOpt import FMNOpt

# global device definition 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray.init(num_gpus=2)
adam = True

search_space_SGD = {
    'lr': FloatDistribution(0.5, 1, log=True),
    'momentum': FloatDistribution(0.81, 0.99),
    'dampening': FloatDistribution(0, 0.2),

}
search_space_ADAM = {
    'lr':FloatDistribution(0.5, 1, log=True)
    #'betas':FloatDistribution(0.9, 0.99)
    
}
if adam:
    search_space=search_space_ADAM
else:
    search_space=search_space_SGD

model = load_model(
    model_dir="../Models/pretrained",
    model_name='Gowal2021Improving_R18_ddpm_100m',
    dataset='cifar10',
    norm='Linf'
)
dataset = load_dataset('cifar10')

attack_params = {
    'batch_size': 50,
    'norm': float('inf'),
    'steps': 30,
    'optimizer': 'Adam'
}
dl_test = torch.utils.data.DataLoader(dataset,
                                      batch_size=attack_params['batch_size'],
                                      shuffle=False)
samples, labels = next(iter(dl_test))


def objective(config, model, samples, labels):
    for epoch in range(5):
        attack = FMNOpt(
            model=model,
            inputs=samples.clone(),
            labels=labels.clone(),
            norm=attack_params['norm'],
            steps=attack_params['steps'],
            optimizer=attack_params['optimizer']
        )
        distance, _ = attack.run(config=config)
        session.report({"distance": distance})


trainable_with_resources = tune.with_resources(
    tune.with_parameters(
        objective,
        model=torch.nn.DataParallel(model).to(device),
        samples=samples.to(device),
        labels=labels.to(device),
    ),
    resources=TUNING_RES
)

mode = 'min'
metric = 'distance'

scheduler = ASHAScheduler(mode=mode, metric=metric, grace_period=2)
optuna_search = OptunaSearch(space=search_space, mode=mode,
                             metric=metric, sampler=TPESampler())

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=10,
        search_alg=optuna_search,
        scheduler=scheduler
    )
)

results = tuner.fit()

best_result = results.get_best_result(metric=metric, mode=mode)
best_config = best_result.config
print(f"best_result : {best_result}\n, best config : {best_config}\n")
