from robustbench.utils import load_model
from Models.SmallCNN import SmallCNN
from Models.downloadModel import download_model
import torch


def load_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    model = SmallCNN()
    model_params_path = download_model(model='mnist_regular')
    model.load_state_dict(torch.load(model_params_path, map_location=device))
    '''

    model = load_model(
        model_dir="./Models/pretrained",
        model_name='Wang2023Better_WRN-70-16',
        dataset='cifar10',
        norm='L2'
    )

    model.eval()
