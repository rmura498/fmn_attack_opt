import numpy as np

from ray import tune
from optuna.distributions import\
    FloatDistribution,\
    IntDistribution, \
    CategoricalDistribution,\
    LogUniformDistribution,\
    UniformDistribution, \
    DiscreteUniformDistribution


OPTIMIZERS_SEARCH_OPTUNA = {
    'SGD': {
        'lr': FloatDistribution(0.5, 1, log=True),
        'momentum': FloatDistribution(0.81, 0.99),
        'dampening': FloatDistribution(0, 0.2)
    },
    'SGDNesterov': {
        'lr': FloatDistribution(0.5, 1, log=True),
        'momentum': FloatDistribution(0.81, 0.99),
        'dampening': CategoricalDistribution([0]),
        'nesterov': CategoricalDistribution([False, True])
    },
    'Adam':
    {
        'lr': FloatDistribution(0.5, 1, log=True),
        'eps': FloatDistribution(1e-8, 1e-7, log=True),
        'amsgrad': CategoricalDistribution([False, True])
    }
}

SCHEDULERS_SEARCH_OPTUNA = {
    'CosineAnnealingLR':
        {
            'T_max': lambda steps: CategoricalDistribution([steps]),
            'eta_min': CategoricalDistribution([0]),
            'last_epoch': CategoricalDistribution([-1])
        },
    'CosineAnnealingWarmRestarts':
        {
            'T_0': lambda steps: CategoricalDistribution([steps//2]),
            'T_mult': CategoricalDistribution([1]),
            'eta_min': CategoricalDistribution([0]),
            'last_epoch': CategoricalDistribution([-1])
        },
    'MultiStepLR':
        {
            'gamma': FloatDistribution(0.1, 0.9)
        },
    'ReduceLROnPlateau':
        {
            'factor': FloatDistribution(0.1, 0.5),
            'patience': CategoricalDistribution([5, 10, 20]),
            'threshold': FloatDistribution(1e-5, 1e-3, log=True)
        }
}

