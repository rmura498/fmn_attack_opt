import os
from robustbench.utils import download_gdrive

PRETRAINED_FOLDER = 'Models'
MODELS = {
    'mnist_regular' : '12HLUrWgMPF_ApVSsWO4_UHsG9sxdb1VJ',
    'Wang2023Better_WRN-70-16':'1-yYcT73GP13c0y9HrgtpyB3NAfkGKgjY'
}



def download_model(model='mnist_regular'):
    model_id = None

    if model not in MODELS.keys():
        return None

    model_id = MODELS[model]

    if not os.path.exists(PRETRAINED_FOLDER):
        os.mkdir(PRETRAINED_FOLDER)
    filepath = os.path.join(PRETRAINED_FOLDER, f'{model}.pth')

    if not os.path.exists(filepath):
        download_gdrive(model_id, filepath)

    return filepath