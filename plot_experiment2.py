import os, argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scienceplots

matplotlib.use("TkAgg")
plt.style.use(['science', 'ieee'])


parser = argparse.ArgumentParser(description='Retrieve experiment data')
parser.add_argument('-e', '--experiments',
                    default='./Experiments/ModelsBestExp',
                    help='Provide the experiments path (e.g. ./Experiments/ModelsBestExp)')

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
    '''
    /ModelsBestExp/
        M0/
            Baseline/
                distance.pkl
                sorted_distances.pkl
                ...
            Tuned/
                distance.pkl
                sorted_distances.pkl
                ...
            AA_robust.txt
        ...
        M8/
            Baseline/
            Tuned/
            AA_robust.txt
    '''

    # ModelsBestExp path
    exps_path = args.experiments

    dirs = os.listdir(exps_path)
    dirs.remove('AA_robust.txt')
    dirs.sort()

    # load AA robust values
    with open(os.path.join(exps_path, 'AA_robust.txt'), 'r') as file:
        AA_robust_values = file.readlines()
        AA_robust_values = [float(val) for val in AA_robust_values]

    num_cols = 3
    num_rows = 3
    base_figsize = 3
    legend_color = (0.961, 0.961, 0.961, 0.8)


    fig = plt.figure(figsize=(base_figsize * num_cols, 3 * num_rows))
    for i, exp in enumerate(dirs):
        print(f"Plotting exp: {exp}...")
        exp_path = os.path.join(exps_path, exp)
        baseline_path = os.path.join(exp_path, 'Baseline')
        tuned_path = os.path.join(exp_path, 'Tuned')

        # load baseline pkl
        b_distances = torch.load(os.path.join(baseline_path, 'sorted_distances.pkl'))
        b_robust = torch.load(os.path.join(tuned_path, 'sorted_robust.pkl'))

        # load tuned pkl
        t_distances = torch.load(os.path.join(tuned_path, 'sorted_distances.pkl'))
        t_robust = torch.load(os.path.join(tuned_path, 'sorted_robust.pkl'))

        exp_params = load_data_txt(tuned_path)
        steps = exp_params['steps']
        batch_size = exp_params['batch size']

        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.plot(b_distances,
                b_robust, label='Baseline')
        ax.plot(t_distances,
                t_robust, label='Tuned')
        ax.plot(8 / 255, AA_robust_values[i], 'x', label='AA')
        ax.grid()

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4

        ax.set_title(f"{exp}", fontsize=fontsize)

        ax.legend(loc=0, prop={'size': 8})
        if i == 7:
            ax.set_xlabel(r"$||\boldsymbol{\delta}||_\infty$")
        if i % 3 == 0:
            ax.set_ylabel("R. Acc.")
        ax.set_xlim([0, 0.2])

    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    print("Saving fig...")
    plt.savefig("example.pdf", bbox_inches='tight', format='pdf')
    plt.show()
