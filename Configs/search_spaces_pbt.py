import numpy as np

from ray import tune


OPTIMIZERS_SEARCH_PBT = {
    'SGD': [
        {
            'lr': tune.grid_search([1, 50]),
            'momentum': 0.9,
            'weight_decay': 0.01,
            'dampening': 0
        },
        {
            'lr': tune.loguniform(1, 100),
            'momentum': tune.uniform(0.81, 0.99),
            'weight_decay': tune.loguniform(0.01, 1),
            'dampening': tune.uniform(0, 0.2)
        }
    ],
    'SGDNesterov': [ # 0: param space for tuner (init config), 1: hyperparams perturbation (for PBT)
        {
            'lr': tune.grid_search([1, 50]), # number of trials
            'momentum': 0.9,
            'weight_decay': 0.01,
            'nesterov': True
        },
        {
            'lr': tune.loguniform(1, 100),
            'momentum': tune.uniform(0.81, 0.99),
            'weight_decay': tune.loguniform(0.01, 1),
        }
    ],
    'Adam': [
        {
            'lr': tune.loguniform(1, 100),
            'weight_decay': 0.01
        },
        {
            'lr': tune.loguniform(1, 100),
            'weight_decay': tune.loguniform(0.01, 1)
        }
    ],
    'AdamAmsgrad': [
        {
            'lr': tune.grid_search([10,50]),
            'amsgrad': True,
        },
        {
            'lr': tune.loguniform(10, 100),
        }
    ]
}

SCHEDULERS_SEARCH_PBT = {
    'CosineAnnealingLR': [
        {
            'T_max': lambda steps: steps,
            'eta_min': 0
        }
    ],
    'CosineAnnealingWarmRestarts': [
        {
            'T_0': lambda steps: steps//2,
            'T_mult': 1,
            'eta_min': 0
        }
    ],
    'MultiStepLR': [
        {
            'milestones': lambda steps: np.linspace(0, steps, 10),
            'gamma': tune.uniform(0.1, 0.9)
        },
        {
            'gamma': tune.uniform(0.1, 0.9)
        }
    ],
    'ReduceLROnPlateau': [
        {
            'factor': 0.1,
            'patience': tune.grid_search([5, 10, 20]),
        },
        {
            'factor': tune.uniform(0.1, 0.5),
            'patience': tune.uniform(1, 5),
        }
    ]
}

