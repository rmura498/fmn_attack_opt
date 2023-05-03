import os
import torch

from Models.load_data import load_dataset

from Experiments.TestAutoAttack import TestAutoAttack
from Experiments.TestFMNAttackTune import TestFMNAttackTune

from robustbench.utils import load_model


if __name__ == '__main__':
    # disable json/csv/Tensorboard logger callbacks  
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

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
            'batch_size': 2,
            'norm': 'inf',
            'steps': 10,
            'optimizer': 'Adam',
            'scheduler': 'MultiStepLR'
        }
    ]

    optimizer_config = {'lr': 10}

    scheduler_config = {'milestones': [ 0.,  2.22222222,  4.44444444,  6.66666667,  8.88888889,
       11.11111111, 13.33333333, 15.55555556, 17.77777778, 20.], 'gamma': 0.2428195052231441}

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
