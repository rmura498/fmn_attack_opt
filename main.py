import torch

from Models.load_data import load_dataset

from Experiments.TestAutoAttack import TestAutoAttack
from Experiments.TestFMNAttackTune import TestFMNAttackTune

from robustbench.utils import load_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoattack_test = False
    dataset = load_dataset('cifar10')

    model = load_model(
        model_dir="./Models/pretrained",
        model_name='Gowal2021Improving_R18_ddpm_100m',
        dataset='cifar10',
        norm='Linf'
    )

    model.eval()

    exps = [
        {
            'batch_size': 10,
            'norm': 'inf',
            'steps': 100,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        }
    ]

    optimizer_config = {
        'lr': 10,
        'betas': (0.9, 0.999),
        'amsgrad': False,
        'eps': 1e-7
    }

    scheduler_config = {
        'factor': 0.1,
        'patience': 10,
        'threshold': 0.1
    }

    for i, exp_params in enumerate(exps):
        print(f"\nRunning experiment #{i}")
        print(f"\t{exp_params}")

        exp = TestFMNAttackTune(model,
                                dataset=dataset,
                                steps=exp_params['steps'],
                                norm=exp_params['norm'],
                                batch_size=exp_params['batch_size'],
                                optimizer=exp_params['optimizer'],
                                scheduler=exp_params['scheduler'],
                                optimizer_config=optimizer_config,
                                scheduler_config=scheduler_config,
                                create_exp_folder=True)
        exp.run()
        exp.save_data()
