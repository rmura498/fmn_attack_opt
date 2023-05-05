import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from Utils.compute_robust import compute_robust
matplotlib.use("TkAgg")
plt.style.use(['science', 'ieee'])




def plot_epsilon_robust(exps_distances=[],
                        exps_names=[],
                        exps_params=[],
                        best_distances=[]):
    if len(exps_distances) == 0:
        print("Error: Not enough distances per experiment!")
        return

    num_cols = 3
    num_rows = 3
    base_figsize = 3

    # number of experiments
    n_exps = len(exps_distances)
    plot_grid_size = n_exps//2 + 1



    for i, exp_distances in enumerate(exps_distances):
        # single experiment
        steps = len(exp_distances)
        batch_size = len(exp_distances[0])

        distances, robust_per_iter = compute_robust(exp_distances, best_distances[i])

        distances = np.array(distances)
        distances.sort()
        robust_per_iter.sort(reverse=True)

        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.plot(distances,
                robust_per_iter)
        ax.plot(8/255, 0.661, 'x', label='AA')
        ax.grid()

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4
        """ ax.set_title(f"Steps: {steps}, batch: {batch_size}, norm: {exps_params[i]['norm']},"
                     f"\nOptimizer: {exps_params[i]['optimizer']}, Scheduler: {exps_params[i]['scheduler']}",
                     fontsize=fontsize)"""
        ax.legend(loc=0, prop={'size': 8})
        ax.set_xlabel(r"$||\boldsymbol{\delta}||_\infty$")
        ax.set_ylabel("R. Acc.")


    plt.xlim([0, 0.2])
    fig.savefig("example.pdf")