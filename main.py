import torch
import torchvision
from torch import nn

from Models.SmallCNN import SmallCNN
from Models.downloadModel import download_model

from Experiments.TestFMNAttack import TestFMNAttack
from Experiments.TestAutoAttack import TestAutoAttack
from Attacks.FMN.FMNOpt import FMNOpt

from robustbench.utils import load_model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    mnist_test = torchvision.datasets.MNIST('./data',
                                            train=False,
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())
    model = SmallCNN()
    model_params_path = download_model(model='mnist_regular')
    model.load_state_dict(torch.load(model_params_path, map_location=device))
    model.eval()
    '''


    cifar10_test = torchvision.datasets.CIFAR10('./data',
                                                train=False,
                                                download=True,
                                                transform=torchvision.transforms.ToTensor())
    
    model = load_model(
        model_dir="./Models/pretrained",
        model_name='Wang2023Better_WRN-70-16',
        dataset='cifar10',
        norm='L2'
    )
    model.eval()

    dataset = cifar10_test

    exps = [
        {
            'batch_size': 10,
            'norm': 2,
            'steps': 30,
            'attack': [FMNOpt, ],
            'optimizer': 'SGD',
            'epsilon': 8/255
        }
    ]
    autoattack_test = False

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
                                optimizer=exp_params['optimizer'],
                                epsilon_init=exp_params['epsilon'])
            exp.run()
            exp.plot()
            exp.save_data()
