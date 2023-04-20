import numpy as np

from ray import tune


OPTIMIZERS_SEARCH_AX = {
    'SGD': {
        'lr': tune.loguniform(1, 5),
        'momentum': tune.uniform(0.81, 0.99),
        'dampening': tune.uniform(0, 0.2)
    },
    'SGDNesterov': {
        'lr': tune.loguniform(0.1, 1),
        'momentum': tune.uniform(0.81, 0.99),
        'dampening': 0,
        'nesterov': tune.choice([False, True])
    },
    'Adam':
    {
        'lr': tune.loguniform(5, 10),
        'eps': tune.loguniform(1e-8, 1e-7),
        'amsgrad': tune.choice([False, True])
    }
}

SCHEDULERS_SEARCH_AX = {
    'CosineAnnealingLR':
        {
            'T_max': lambda steps: steps,
            'eta_min': 0,
            'last_epoch': -1
        },
    'CosineAnnealingWarmRestarts':
        {
            'T_0': lambda steps: [steps//2],
            'T_mult': 1,
            'eta_min': 0,
            'last_epoch': -1
        },
    'MultiStepLR':
        {
            'milestones': lambda steps: tune.grid_search(
                [np.linspace(0, steps, 10),
                 np.linspace(0, steps, 5),
                 np.linspace(0, steps, 3)]
            ),
            'gamma': tune.uniform(0.1, 0.9)
        },
    'ReduceLROnPlateau':
        {
            'factor': tune.uniform(0.1, 0.5),
            'patience': tune.choice([5, 10, 20]),
            'threshold': tune.loguniform(1e-5, 1e-3)
        }
}

