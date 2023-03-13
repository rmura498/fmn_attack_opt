import torch
import torchvision
from robustbench.utils import download_gdrive
import os
from Models.SmallCNN import SmallCNN
from FMN.FMNattack import fmn

# download model
PRETRAINED_FOLDER = 'pretrained'
# create folder for storing models
if not os.path.exists(PRETRAINED_FOLDER):
    os.mkdir(PRETRAINED_FOLDER)
MODEL_ID_REGULAR = '12HLUrWgMPF_ApVSsWO4_UHsG9sxdb1VJ'
filepath = os.path.join(PRETRAINED_FOLDER, f'mnist_regular.pth')
if not os.path.exists(filepath):
    # utility function to handle google drive data
    download_gdrive(MODEL_ID_REGULAR, filepath)

model = SmallCNN()
model.load_state_dict(torch.load(os.path.join(PRETRAINED_FOLDER,
                                              'mnist_regular.pth'), ))
model.eval()

BATCH_SIZE = 10
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)


#-----------------------------------------------------------------------------------------#


def accuracy(model, samples, labels):
    preds = model(samples)
    acc = (preds.argmax(dim=1) == labels).float().mean()
    return acc.item()


samples, labels = next(iter(dl_test))

acc = accuracy(model, samples, labels)

print("standard accuracy: ", acc)

x_adv = samples.clone()

advs = fmn(model, x_adv, labels, norm=0)
acc = accuracy(model, advs, labels)
print("Robust accuracy", acc)