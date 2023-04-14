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

    plot_experiments = False
    autoattack_test = False
    dataset = load_dataset('cifar10')

    if not plot_experiments:
        # TODO: move load model in Utils
        '''
        model = SmallCNN()
        model_params_path = download_model(model='mnist_regular')
        model.load_state_dict(torch.load(model_params_path, map_location=device))
        '''
        #Pang2022Robustness_WRN70_16 - 11° ICML 2022
        #Pang2022Robustness_WRN28_10 - 18° ICML 2022
        #Wang2023Better_WRN-70-16 - 1° arxiv 2023
        #Wang2023Better_WRN-28-10 - 2° arxiv 2023
        #Gowal2021Improving_28_10_ddpm_100m - 10° Neurips 2021
        #Sehwag2021Proxy_ResNest152 - 13° ICLR 2022
        model = load_model(
            model_dir="./Models/pretrained",
            model_name='Gowal2021Improving_28_10_ddpm_100m',
            dataset='cifar10',
            norm='Linf'
        )
        # evitare potentially unreliable come modelli

        # 4, 10 neurips
        # 11 icml, 13 clear, 15

        model.eval()

        exps = [
            {
                'batch_size': 10,
                'norm': float('inf'),
                'steps': 15,
                'attack': [FMNOpt, ],
                'optimizer': 'SGD'
            }
        ]

        if autoattack_test:
            for exp_params in exps:
                AA = TestAutoAttack(model=model,
                                    dataset=dataset,
                                    attack=['apgd-ce', ],
                                    batch_size=exp_params['batch_size'],
                                    norm=exp_params['norm'])
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
                                    optimizer=exp_params['optimizer'],
                                    create_exp_folder=False)
                exp.run()
                exp.save_data()
                # TODO: save best adv

    else:
        experiments = [
            'Exp_141253_FMNOpt_DMWideResNet_CIFAR10',
            'Exp_141255_FMNOpt_DMWideResNet_CIFAR10',
            'Exp_141650_FMNOpt_DMWideResNet_CIFAR10'
        ]

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
