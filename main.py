import torch
import torchvision

from Utils.metrics import accuracy
from Models.SmallCNN import SmallCNN
from Models.downloadModel import download_model
from Attacks.FMN.FMNBase import FMNBase

from Attacks.FMN import original_fmn


if __name__=='__main__':
    #TODO:  create a class to perform the experiments
    #        - allows to perform a single experiment in a run (batch or single samples)
    #        - allows to keep track of the experiment results (model accuracy, attack params)

    # exp = TestAttack(model, dataset, batch (true, false), attack)
    # exp.run()
    # exp.plotEpsilonLoss() - plots epsilon and loss over the iteration


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params_path = download_model()

    model = SmallCNN()

    model.load_state_dict(torch.load(model_params_path, map_location=device))
    model.eval()

    BATCH_SIZE = 10
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
    dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)


    samples, labels = next(iter(dl_test))

    acc = accuracy(model, samples, labels)
    print("standard accuracy: ", acc)

    x_adv = samples.clone()
    # attack = FmnOpt(model, x_adv, labels, norm=1)
    attack = FMNBase(model, x_adv, labels, norm=0)
    advs = attack.run()

    #advs = original_fmn.fmn(model, x_adv, labels, norm=0)

    acc = accuracy(model, advs, labels)
    print("Robust accuracy", acc)