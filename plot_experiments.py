import os
import pickle

import torch

from Utils.plots import plot_epsilon_robust, plot_distance

experiments = {
            'Exp_171812_FMNOpt_DMPreActResNet_CIFAR10',
            'Exp_171917_FMNOpt_DMPreActResNet_CIFAR10',
            'Exp_172244_FMNOpt_DMPreActResNet_CIFAR10'
        }

exps_data = []
exps_params = []
AA_exps_data = []
AA_exps_params = []

for exp_path in experiments:
    exp_path = os.path.join("Experiments", exp_path)
    exp_data = {
        'epsilon': [],
        'pred_labels': [],
        'distance': [],
        'inputs': [],
        'best_adv': []
    }

    # load data.txt
    exp_params = {
        'optimizer': None,
        'scheduler': None,
        'norm': None,
        'batch size': None
    }
    data_path = os.path.join(exp_path, "data.txt")
    with open(data_path, 'r') as file:
        for line in file.readlines():
            _line = line.lower()
            if any((match := substring) in _line for substring in exp_params.keys()):
                exp_params[match] = line.split(":")[-1].strip()
                if match == 'norm':
                    if exp_params[match] in ['inf', 'Linf']:
                        exp_params[match] = float('inf')
                    else:
                        exp_params[match] = int(exp_params[match])

    for data in exp_data:
        data_path = os.path.join(exp_path, f"{data}.pkl")
        with open(data_path, 'rb') as file:
            data_load = pickle.load(file)
            exp_data[data] = data_load

    if 'AA' not in exp_path:
        exps_params.append(exp_params)
        exps_data.append(exp_data)
    else:
        AA_exps_data.append(exp_data)
        AA_exps_params.append(exp_params)
best_distances = []

for i, exp in enumerate(exps_data):
    best_adv = exp['best_adv']
    inputs = exp['inputs']
    distance = torch.linalg.norm((best_adv - inputs).data.flatten(1), dim=1, ord=exps_params[i]['norm'])
    best_distances.append(distance)

plot_epsilon_robust(
    exps_distances=[exp_data['distance']
                    for exp_data in exps_data],
    exps_names=experiments,
    exps_params=exps_params,
    best_distances=best_distances
)

plot_distance([exp_data['epsilon']
               for exp_data in exps_data],
              [exp_data['distance']
               for exp_data in exps_data],
              exps_names=experiments,
              exps_params=exps_params)