import torchvision
from robustbench.utils import load_model as rb_load_model

MODELS = [
    'Gowal2021Improving_R18_ddpm_100m',
    'Wang2023Better_WRN-70-16'
]

DATASETS = [
    'cifar10',
    'cifar100',
    'mnist'
]

MODEL_NORMS = [0, 1, 2, 'inf']


def load_dataset(dataset_name='cifar10'):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('./data',
                                             train=False,
                                             download=True,
                                             transform=torchvision.transforms.ToTensor())
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('./data',
                                               train=False,
                                               download=True,
                                               transform=torchvision.transforms.ToTensor())

    return dataset


def load_model(model_name, dataset_name, norm='inf'):
    if norm not in MODEL_NORMS:
        norm = 'inf'

    norm_name = f'L{norm}'

    try:
        model = rb_load_model(
            model_dir="./Models/pretrained",
            model_name=model_name,
            dataset=dataset_name,
            norm=norm_name
        )
    except KeyError:
        model = rb_load_model(
            model_dir="./Models/pretrained",
            model_name=MODELS[0],
            dataset='cifar10',
            norm='Linf'
        )

    return model


def load_data(model_id=0, dataset_id=0, norm='inf'):
    """
    Load model and dataset (default: Gowal2021Improving_R18_ddpm_100m, CIFAR10)
    """

    model_id = 0 if model_id > len(MODELS) else model_id
    dataset_id = 0 if dataset_id > len(DATASETS) else dataset_id

    model_name = MODELS[model_id]
    dataset_name = DATASETS[dataset_id]

    model = load_model(model_name, dataset_name, norm)
    dataset = load_dataset(dataset_name)

    return model, dataset
