import os
from robustbench.utils import download_gdrive

PRETRAINED_FOLDER = 'pretrained'
MODELS = {
    'mnist_regular' : '12HLUrWgMPF_ApVSsWO4_UHsG9sxdb1VJ'
}


def download_model(model='mnist_regular'):
    model_id = None

    if model not in MODELS.keys():
        return None

    if not os.path.exists(PRETRAINED_FOLDER):
        os.mkdir(PRETRAINED_FOLDER)
    filepath = os.path.join(PRETRAINED_FOLDER, f'{model}.pth')
    
    if not os.path.exists(filepath):
        download_gdrive(model_id, filepath)

    return filepath

    