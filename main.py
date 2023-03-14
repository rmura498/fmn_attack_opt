import torch
import torchvision

from Utils.metrics import accuracy
from Models.SmallCNN import SmallCNN
from Models.DownloadModel import download_model
from FMN.FMN_attack_opt import fmn


if __name__=='__main__':
    model_params_path = download_model()

    model = SmallCNN()
    model.load_state_dict(torch.load(model_params_path, ))
    model.eval()

    BATCH_SIZE = 10
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
    dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)


    samples, labels = next(iter(dl_test))

    acc = accuracy(model, samples, labels)

    print("standard accuracy: ", acc)

    x_adv = samples.clone()

    advs = fmn(model, x_adv, labels, norm=0)
    acc = accuracy(model, advs, labels)
    print("Robust accuracy", acc)