import os, argparse
import numpy as np

import torch

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Retrieve experiment data')
parser.add_argument('-ep', '--exp_name',
                    required=True,
                    help='Provide the experiment folder name (e.g. Model_Dataset_Opt_Sch_Loss) which \
                         can be retrieved inside the ./Experiments folder')

parser.add_argument('-AA', '--autoattack_robust',
                    default=0.60,
                    help='Provide the autoattack robust value for 8/255')

args = parser.parse_args()


def load_data_txt(exp_path):
    exp_params = {
        'optimizer': None,
        'scheduler': None,
        'norm': None,
        'batch size': None,
        'steps': None
    }
    data_path = os.path.join(exp_path, "data.txt")
    with open(data_path, 'r') as file:
        for line in file.readlines():
            _line = line.lower()
            if any((match := substring) in _line for substring in exp_params.keys()):
                exp_params[match] = line.split(":")[-1].strip()
                if match == 'norm':
                    if exp_params[match] in ['inf', 'Linf']:
                        exp_params[match] = float('inf')
                    else:
                        exp_params[match] = int(exp_params[match])
    return exp_params

if __name__ == '__main__':
    exp_name = args.exp_name
    exp_path = os.path.join("Experiments", exp_name)

    AA_robust_val = float(args.autoattack_robust)

    # load merged_distances and best_distances for the experiment to plot
    distances = torch.load(os.path.join(exp_path, 'sorted_distances.pkl'))
    robust = torch.load(os.path.join(exp_path, 'sorted_robust.pkl'))

    exp_params = load_data_txt(exp_path)

    fig, ax = plt.subplots()

    # single experiment
    steps = exp_params['steps']
    batch_size = exp_params['batch size']

    ax.plot(distances,
            robust,
            label='robust')
    ax.plot(8 / 255, AA_robust_val, 'x')
    ax.axvline(8 / 255, c='g', linewidth=1)
    ax.grid()

    dpi = fig.dpi
    rect_height_inch = ax.bbox.height / dpi
    fontsize = rect_height_inch * 4

    ax.set_title(
        f"Steps: {steps}, batch: {batch_size}, norm: {exp_params['norm']},\n\
        Optimizer: {exp_params['optimizer']}, Scheduler: {exp_params['scheduler']}",
        fontsize=fontsize)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Robust")
    plt.tight_layout()
    plt.show()
