import argparse

from Models.load_data import load_data
from Configs.tuning_resources import TUNING_RES
from Configs.search_spaces import OPTIMIZERS_SEARCH, SCHEDULERS_SEARCH

parser = argparse.ArgumentParser(description='Retrieve tuning params')
parser.add_argument('-opt', '--optimizer',
                    default='SGD',
                    help='Provide the optimizer name (e.g. SGD, Adam)')
parser.add_argument('-scd', '--scheduler',
                    default='CosineAnnealingLR',
                    help='Provide the scheduler name (e.g. CosineAnnealingLR)')
parser.add_argument('-m_id', '--model_id',
                    default=0,
                    help='Provide the model ID')
parser.add_argument('-d_id', '--dataset_id',
                    default=0,
                    help='Provide the dataset ID (e.g. CIFAR10: 0)')
parser.add_argument('-b', '--batch',
                    default=50,
                    type=int,
                    help='Provide the batch size (e.g. 50, 100, 10000)')
parser.add_argument('-s', '--steps',
                    default=50,
                    type=int,
                    help='Provide the steps number (e.g. 50, 100, 10000)')
parser.add_argument('-n', '--norm',
                    default='inf',
                    help='Provide the norm (0, 1, 2, inf)')
parser.add_argument('-n_sample', '--num_samples',
                    default=5,
                    help='Provide the number of samples for the tuning')
parser.add_argument('-ep', '--epochs',
                    default=5,
                    help='Provide the number of epochs for the attack tuning \
                    (how many times the attack is run by the tuner)')

args = parser.parse_args()

if __name__ == '__main__':
    optimizer = args.optimizer
    scheduler = args.scheduler
    model_id = args.model_id
    dataset_id = args.dataset_id

    attack_params = {
        'batch': args.batch,
        'steps': args.steps,
        'norm': args.norm
    }

    tune_config = {
        'num_samples': args.num_samples,
        'epochs': args.epochs
    }

    # load model and dataset
    #model, dataset = load_data(model_id, dataset_id, attack_params['norm'])

    # load search spaces
    optimizer_search = OPTIMIZERS_SEARCH[optimizer]
    scheduler_search = SCHEDULERS_SEARCH[scheduler]

    steps_keys = ['T_max', 'T_0', 'milestones']

    for key in steps_keys:
        if key in scheduler_search:
            scheduler_search[key] = scheduler_search[key](attack_params['steps'])

    search_space =


