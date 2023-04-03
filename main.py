import os
import pickle

import torch
from torch import nn

from Models.SmallCNN import SmallCNN
from Models.downloadModel import download_model

from Utils.plots import plot_epsilon_robust, plot_distance
from Utils.datasets import load_dataset

from Experiments.TestFMNAttack import TestFMNAttack
from Experiments.TestAutoAttack import TestAutoAttack
from Attacks.FMN.FMNOpt import FMNOpt

from robustbench.utils import load_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot_experiments = True
    autoattack_test = False
    dataset = load_dataset('cifar10')

    if not plot_experiments:
        # TODO: move load model in Utils
        '''
        model = SmallCNN()
        model_params_path = download_model(model='mnist_regular')
        model.load_state_dict(torch.load(model_params_path, map_location=device))
        '''

        model = load_model(
            model_dir="./Models/pretrained",
            model_name='Wang2023Better_WRN-70-16',
            dataset='cifar10',
            norm='L2'
        )

        model.eval()

        exps = [
            {
                'batch_size': 5,
                'norm': 2,
                'steps': 10,
                'attack': [FMNOpt, ],
                'optimizer': 'SGD'
            }
        ]

        if autoattack_test:
            for exp_params in exps:
                AA = TestAutoAttack(model=model,
                                    dataset=dataset,
                                    batch_size=exp_params['batch_size'],
                                    norm=exp_params['norm'],
                                    steps=exp_params['steps'])
                AA.run()
                AA.save_data()

        for i, exp_params in enumerate(exps):
            print(f"\nRunning experiment #{i}")
            print(f"\t{exp_params}")

            for attack in exp_params['attack']:
                exp = TestFMNAttack(model,
                                    dataset=dataset,
                                    attack=attack,
                                    steps=exp_params['steps'],
                                    norm=exp_params['norm'],
                                    batch_size=exp_params['batch_size'],
                                    optimizer=exp_params['optimizer'])
                exp.run()
                exp.save_data()

    else:
        experiments = [
            'Exp_031738_FMNOpt_DMWideResNet_CIFAR10',
            'Exp_031757_FMNOpt_DMWideResNet_CIFAR10'
        ]

        exps_data = []
        exps_params = []
        for exp_path in experiments:
            exp_path = os.path.join("Experiments", exp_path)
            exp_data = {
                'epsilon': [],
                'labels': [],
                'pred_labels': [],
                'delta': []
            }

            # load data.txt
            exp_params = {
                'optimizer': None,
                'scheduler': None,
                'norm': None
            }
            data_path = os.path.join(exp_path, "data.txt")
            with open(data_path, 'r') as file:
                for line in file.readlines():
                    _line = line.lower()
                    if any((match := substring) in _line for substring in exp_params.keys()):
                        exp_params[match] = line.split(":")[-1].strip()

            exps_params.append(exp_params)

            for data in exp_data:
                data_path = os.path.join(exp_path, f"{data}.pkl")
                with open(data_path, 'rb') as file:
                    data_load = pickle.load(file)
                    exp_data[data] = data_load

            exps_data.append(exp_data)

        plot_epsilon_robust(
            exps_epsilon_per_iter=[exp_data['epsilon']
                                   for exp_data in exps_data],
            exps_names=experiments,
            exps_params=exps_params
        )

        plot_distance([exp_data['epsilon']
                       for exp_data in exps_data],
                      [exp_data['delta']
                       for exp_data in exps_data],
                      exps_names=experiments,
                      exps_params=exps_params)
