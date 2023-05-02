import argparse, pickle
import torch
import numpy as np
from Experiments.TestFMNAttackTune import TestFMNAttackTune

from Models.load_data import load_model, load_dataset

# global device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Retrieve attack params')
parser.add_argument('-b', '--batch',
                    default=32,
                    help='Provide the batch size')
parser.add_argument('-s', '--steps',
                    default=100,
                    help='Provide the step size')
parser.add_argument('-fc', '--fmn_config',
                    default='./Configs/ModelsBestConfigs/Gowal2021Improving_28_10_ddpm_100m_cifar10_Adam_CosineAnnealingLR_CE.pkl',
                    help='Provide the path of the .pkl file which contains the best \
                    config for a given optimizer, scheduler and loss')
parser.add_argument('-dp', '--dataset_percent',
                    default=0.5,
                    help='Provide the dataset percentage which was used to tune the attack hyperparams')

args = parser.parse_args()


def splitting_pkl_name(filename):
    splits = filename.split('_')
    splits.remove('cifar10')
    optimizer, scheduler, loss = splits[-3:]
    loss = loss.split('.')[0]

    model = '_'.join(splits[:-3])
    return model, optimizer, scheduler, loss


if __name__ == '__main__':

    # load arguments
    batch = int(args.batch)
    steps = int(args.steps)
    fmn_config_path = str(args.fmn_config)
    tuning_dataset_percent = float(args.dataset_percent)

    pkl_filename = fmn_config_path.split('/')[-1]
    model_name, optimizer, scheduler, loss = splitting_pkl_name(pkl_filename)

    model = load_model(model_name, 'cifar10')
    dataset = load_dataset('cifar10')
    model.eval()

    # load fmn pkl config file
    try:
        with open(fmn_config_path, 'rb') as file:
            fmn_config = pickle.load(file)
    except Exception as e:
        print("Cannot load the configuration:")
        print(fmn_config_path)
        exit(1)

    optimizer_config = fmn_config['best_config']['opt_s']
    scheduler_config = fmn_config['best_config']['sch_s']

    if scheduler == 'MultiStepLR':
        milestones = len(scheduler_config['milestones'])
        scheduler_config['milestones'] = np.linspace(0, steps, milestones)

    if scheduler == 'CosineAnnealingLR':
        scheduler_config['T_max'] = steps

    if scheduler == 'CosineAnnealingWarmRestarts':
        scheduler_config['T_0'] = steps//2

    exp = TestFMNAttackTune(model,
                            dataset=dataset,
                            steps=steps,
                            batch_size=batch,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss=loss,
                            optimizer_config=optimizer_config,
                            scheduler_config=scheduler_config,
                            create_exp_folder=True,
                            tuning_dataset_percent=tuning_dataset_percent,
                            model_name=model_name)
    exp.run()
    exp.save_data()



