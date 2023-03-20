import torch
import torchvision

from Models.SmallCNN import SmallCNN
from Models.downloadModel import download_model

from Experiments.TestFMNAttack import TestFMNAttack
from Attacks.FMN.FMNBase import FMNBase
from Attacks.FMN.FMNOpt import FMNOpt


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params_path = download_model()

    model = SmallCNN()
    model.load_state_dict(torch.load(model_params_path, map_location=device))
    model.eval()
    mnist_test = torchvision.datasets.MNIST('./data',
                                            train=False,
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())

    exps = [
        {
            'batch_size': 10,
            'norm': 0,
            'steps': 10,
            'attack': [FMNBase, FMNOpt]
        },
        {
            'batch_size': 10,
            'norm': 0,
            'steps': 20,
            'attack': [FMNBase, FMNOpt]
        },
        {
            'batch_size': 10,
            'norm': 0,
            'steps': 50,
            'attack': [FMNBase, FMNOpt]
        },
        {
            'batch_size': 10,
            'norm': 1,
            'steps': 10,
            'attack': [FMNBase, FMNOpt]
        },
        {
            'batch_size': 10,
            'norm': 1,
            'steps': 50,
            'attack': [FMNBase, FMNOpt]
        }
    ]
    for i, exp_params in enumerate(exps):
        print(f"\nRunning experiment #{i}--")
        print(f"\t{exp_params}")

        for attack in exp_params['attack']:
            exp = TestFMNAttack(model,
                                mnist_test,
                                attack=attack,
                                steps=exp_params['steps'],
                                norm=exp_params['norm'],
                                batch_size=exp_params['batch_size'])
            exp.run()
            exp.plot()


