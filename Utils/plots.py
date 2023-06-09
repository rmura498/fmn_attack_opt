import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from Utils.compute_robust import compute_robust
#matplotlib.use("TkAgg")
plt.style.use(['science', 'ieee'])


def plot_distance(exps_epsilon_per_iter=[],
                  exps_distance_per_iter=[],
                  exps_names=[],
                  exps_params=[]):




    if len(exps_epsilon_per_iter) == 0:
        print("Error: Not enough epsilon values per iter!")
        return

    # number of experiments
    if (len(exps_epsilon_per_iter)) != (len(exps_distance_per_iter)):
        print("Error: epsilon size is different from distance size!")
        return

    n_exps = len(exps_epsilon_per_iter)
    plot_grid_size = n_exps // 2 + 1
    fig = plt.figure()

    for i in range(n_exps):
        exp_epsilons = exps_epsilon_per_iter[i]
        exp_distances = exps_distance_per_iter[i]

        steps = len(exp_epsilons)
        batch_size = len(exp_epsilons[0])

        # single experiment
        epsilons = []
        distances = []

        for epsilon in exp_epsilons:
            epsilons.append(torch.linalg.norm(epsilon, ord=exps_params[i]['norm']).item())
        for distance in exp_distances:
            distances.append(torch.linalg.norm(distance, ord=exps_params[i]['norm']).item())

        ax = fig.add_subplot(plot_grid_size, plot_grid_size, i + 1)
        ax.plot(epsilons,
                label='epsilon')

        ax.plot(distances,
                label='distances')

        ax.legend(loc=0, prop={'size': 8})

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4

        ax.set_title(f"Steps: {steps}, batch: {batch_size}, norm: {exps_params[i]['norm']},\nOptimizier: {exps_params[i]['optimizer']}, Scheduler: {exps_params[i]['scheduler']}",
                     fontsize=fontsize)

        ax.set_xlabel("Steps")
        ax.set_ylabel("Epsilon/Distance (x-x0)")
        ax.grid()


    plt.show()


def plot_epsilon_robust(exps_distances=[],
                        exps_names=[],
                        exps_params=[],
                        best_distances=[]):
    if len(exps_distances) == 0:
        print("Error: Not enough distances per experiment!")
        return

    # number of experiments
    n_exps = len(exps_distances)
    plot_grid_size = n_exps//2 + 1

    fig = plt.figure(figsize=(3,3))

    for i, exp_distances in enumerate(exps_distances):
        # single experiment
        steps = len(exp_distances)
        batch_size = len(exp_distances[0])

        distances, robust_per_iter = compute_robust(exp_distances, best_distances[i])

        distances = np.array(distances)
        distances.sort()
        robust_per_iter.sort(reverse=True)

        ax = fig.add_subplot(plot_grid_size, plot_grid_size, i + 1)
        ax.plot(distances,
                robust_per_iter)
        ax.plot(8/255, 0.661, 'x', label='AA')
        ax.grid()

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4
        ax.set_title(f"Steps: {steps}, batch: {batch_size}, norm: {exps_params[i]['norm']},"
                     f"\nOptimizer: {exps_params[i]['optimizer']}, Scheduler: {exps_params[i]['scheduler']}",
                     fontsize=fontsize)
        ax.legend(loc=0, prop={'size': 8})
        ax.set_xlabel(r"$||\boldsymbol{\delta}||_\infty$")
        ax.set_ylabel("R. Acc.")


    plt.xlim([0, 0.2])
    fig.savefig("example.pdf")

'''
def plot_2D_attack(clf, target, labels, n_classes):
    """
    Args:
        clf: the classifier
        target: True if the attack is targeted
        labels: true labels
        n_classes: total number of classes
    """
    if target is not False:
        target_classes = (labels + 1) % n_classes * target
        criterion = fb.criteria.TargetedMisclassification(target_classes)
    else:
        criterion = fb.criteria.Misclassification(labels)
        target_classes = labels

    image_path = "../../../images/"  # change the path here
    fig = CFigure(width=5, height=5)

    n_grid_pts = 20

    # Convenience function for plotting the decision function of a classifier
    fig.sp.plot_decision_regions(clf, n_grid_points=200, plot_background=False)
    # We tried to implement the loss_fmn_fn() function
    # this function should be changed to let the fig.sp.plot_fun generate the landscape
    fig.sp.plot_fun(func=loss_fmn_fn(),
                    multipoint=False,
                    colorbar=False,
                    cmap='coolwarm',
                    n_grid_points=n_grid_pts,
                    func_args=[target_classes, fmodel, target])
    fig.sp.grid(grid_on=False)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1)
    fig.sp.xlim([0, 1])
    fig.sp.ylim([0, 1])
    plt.savefig(os.path.join(image_path, 'bg.png'),
                bbox_inches=0,
                format='png')
    plt.show()
    fig.close()
    
'''
