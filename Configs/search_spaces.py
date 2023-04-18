import numpy as np

from ray import tune
from optuna.distributions import\
    FloatDistribution,\
    IntDistribution, \
    CategoricalDistribution,\
    LogUniformDistribution,\
    UniformDistribution


OPTIMIZERS_SEARCH = {
    'SGD': {
        'lr': LogUniformDistribution(0.5, 1),
        'momentum': FloatDistribution(0.81, 0.99),
        'dampening': FloatDistribution(0, 0.2),
        'Nesterov': CategoricalDistribution([False, True])
    },
    'Adam':
    {
        'lr': LogUniformDistribution(0.5, 1),
        'betas': FloatDistribution(0.81, 0.99),
        'eps': LogUniformDistribution(1e-8, 1e-7),
        'amsgrad': CategoricalDistribution([False, True])
    }
}

SCHEDULERS_SEARCH = {
    'CosineAnnealingLR':
        {
            'T_max': lambda steps: steps,
            'eta_min': 0,
            'last_epoch': -1
        },
    'CosineAnnealingWarmRestarts':
        {
            'T_0': lambda steps: steps/2,
            'T_mult': 1,
            'eta_min': 0,
            'last_epoch': -1
        },
    'MultiStepLR':
        {
            'milestones': lambda steps: tune.sample_from(lambda steps: [
                np.linspace(0, steps, 10),
                np.linspace(0, steps, 5),
                np.linspace(0, steps, 3)]),
            'gamma': UniformDistribution(0.1, 0.9)
        },
    'ReduceLROnPlateau':
        {
            'factor': UniformDistribution(0.1, 0.5),
            'patience': CategoricalDistribution([5, 10, 20]),
            'threshold': LogUniformDistribution(1e-5, 1e-3)
        }
}

