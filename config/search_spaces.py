from optuna.distributions import FloatDistribution, IntDistribution, \
    CategoricalDistribution, LogUniformDistribution, UniformDistribution
from ray import tune
import numpy as np


SGD_search_space = {'SGD':
    {
        'lr': LogUniformDistribution(0.5, 1),
        'momentum': FloatDistribution(0.81, 0.99),
        'dampening': FloatDistribution(0, 0.2),
        'Nesterov': CategoricalDistribution([False, True])
    }
}

Adam_search_space = {'Adam':
    {
        'lr': LogUniformDistribution(0.5, 1),
        'betas': FloatDistribution(0.81, 0.99),
        'eps': LogUniformDistribution(1e-8, 1e-7),
        'amsgrad': CategoricalDistribution([False, True])
    }
}
MultistepLR_search_space = {'MultistepLR':
    {
        'milestones': tune.sample_from(lambda _: [np.linspace(0, _, 10),
                                                  np.linspace(0, _, 5),
                                                  np.linspace(0, _, 3)]),
        'gamma': UniformDistribution(0.1, 0.9)
    }
}

ReduceLROnPlateau_search_space = {'ReduceLROnPlateau':
    {
        'factor': UniformDistribution(0.1, 0.5),
        'patience': CategoricalDistribution([5, 10, 20]),
        'threshold': LogUniformDistribution(1e-5, 1e-3),

    }
}
