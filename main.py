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

    batch_size = 10
    norm = 1
    steps = 50
    attack = FMNOpt

    mnist_test = torchvision.datasets.MNIST('./data',
                                            train=False,
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())

    exp = TestFMNAttack(model,
                        mnist_test,
                        attack=attack,
                        steps=steps,
                        norm=norm,
                        batch_size=batch_size)
    exp.run()
    exp.plot()


