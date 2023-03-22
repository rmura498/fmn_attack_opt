import torch
import torchvision
from Experiments.TestAutoAttack import TestAutoAttack
from torch import nn

from Models.SmallCNN import SmallCNN
from Models.downloadModel import download_model

from Experiments.TestFMNAttack import TestFMNAttack
from Attacks.FMN.FMNBase import FMNBase
from Attacks.FMN.FMNOpt import FMNOpt

from robustbench.utils import load_model
from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR100_STD, CIFAR100_MEAN, DMWideResNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    model_params_path = download_model(model='Wang2023Better_WRN-70-16')
    model = DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=nn.SiLU,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD)
    model.load_state_dict(torch.load(model_params_path, map_location=device))
    model.eval()
    '''
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
    # model.eval()

    '''
    mnist_test = torchvision.datasets.MNIST('./data',
                                            train=False,
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())
    '''

    # ----------------------------------------------------------------------------------------
    cifar10_test = torchvision.datasets.CIFAR10('./data',
                                                train=False,
                                                download=True,
                                                transform=torchvision.transforms.ToTensor())

    # model_params_path = download_model(model='Wang2023Better_WRN-70-16')
    model = load_model(
        model_dir="./Models/pretrained",
        model_name='Wang2023Better_WRN-70-16',
        dataset='cifar10',
        norm='Linf'
    )
    # model.load_state_dict(torch.load(model_params_path, map_location=device))
    model.eval()

    exps = [
        {
            'batch_size': 10,
            'norm': float('inf'),
            'steps': 10,
            'attack': [FMNOpt, ]
        }
    ]
    autoattack_test = False

    if autoattack_test:
        for exp_params in exps:
            AA = TestAutoAttack(model=model,
                                dataset=cifar10_test,
                                batch_size=exp_params['batch_size'],
                                norm=exp_params['norm'],
                                steps=exp_params['steps'])
            AA.run()

    for i, exp_params in enumerate(exps):
        print(f"\nRunning experiment #{i}")
        print(f"\t{exp_params}")

        for attack in exp_params['attack']:
            exp = TestFMNAttack(model,
                                cifar10_test,
                                attack=attack,
                                steps=exp_params['steps'],
                                norm=exp_params['norm'],
                                batch_size=exp_params['batch_size'])
            exp.run()
            exp.plot()
