import torchvision


def load_dataset(dataset_name='mnist'):
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