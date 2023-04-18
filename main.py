import torch

from Models.load_data import load_dataset

from Experiments.TestFMNAttack import TestFMNAttack
from Experiments.TestAutoAttack import TestAutoAttack
from Attacks.FMN.FMNOpt import FMNOpt

from robustbench.utils import load_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoattack_test = False
    dataset = load_dataset('cifar10')

    # TODO: move load model in Utils
    '''
    model = SmallCNN()
    model_params_path = download_model(model='mnist_regular')
    model.load_state_dict(torch.load(model_params_path, map_location=device))
    '''
    # Pang2022Robustness_WRN70_16 - 11° ICML 2022
    # Pang2022Robustness_WRN28_10 - 18° ICML 2022
    # Wang2023Better_WRN-70-16 - 1° arxiv 2023
    # Wang2023Better_WRN-28-10 - 2° arxiv 2023
    # Gowal2021Improving_28_10_ddpm_100m - 10° Neurips 2021
    # Gowal2021Improving_R18_ddpm_100m - 28° Neurips 2021 - 58.50
    # Sehwag2021Proxy_ResNest152 - 13° ICLR 2022

    model = load_model(
        model_dir="./Models/pretrained",
        model_name='Gowal2021Improving_R18_ddpm_100m',
        dataset='cifar10',
        norm='Linf'
    )

    model.eval()

    exps = [
        {
            'batch_size': 50,
            'norm': float('inf'),
            'steps': 30,
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
