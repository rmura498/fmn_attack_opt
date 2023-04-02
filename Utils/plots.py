import os.path
import numpy as np
import matplotlib.pyplot as plt

from .metrics import loss_fmn_fn


def plot_epsilon_robust(exps_epsilon_per_iter=[], exps_names=[]):
    if len(exps_epsilon_per_iter) == 0:
        return

    # number of experiments
    n_exps = len(exps_epsilon_per_iter)

    fig = plt.figure()

    for i, exp_epsilons in enumerate(exps_epsilon_per_iter):
        # single experiment
        steps = len(exp_epsilons)
        batch_size = len(exp_epsilons[0])

        epsilons = np.array([])
        robust_per_iter = []
        for j, epsilon in enumerate(exp_epsilons):
            # checking, for each step, the epsilon tensor
            epsilons = np.concatenate((epsilons, epsilon.numpy()), axis=None)
            robust_per_iter += [
                (np.count_nonzero(eps > exp_epsilons[-1])/batch_size)
                for eps in epsilon
            ]

        epsilons = np.array(epsilons)
        epsilons /= 100
        epsilons.sort()
        epsilons = epsilons[::-1]

        robust_per_iter.sort()
        robust_per_iter = robust_per_iter[::-1]

        ax = fig.add_subplot(n_exps, n_exps, i+1)
        x_values = np.linspace(1, steps*batch_size, steps*batch_size)

        ax.plot(x_values,
                robust_per_iter,
                label='robust')
        ax.plot(x_values,
                epsilons,
                label='epsilon')
        ax.legend()

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4

        if len(exps_names) > 0:
            ax.set_title(exps_names[i], fontsize=fontsize)

        # TODO: set labels
        # ax.set_xlabel("Steps*Batch")
        # ax.set_ylabel("Robust/Epsilon")
    # fig.legend()
    plt.show()


    # TODO: save the plot
    # plot_name = f'plot_{steps}_{norm}'
    # fig1.savefig(os.path.join(path, f"{plot_name}.png"))


def plot_2D_attack(clf, target, labels, n_classes):
    if target is not False:
        target_classes = (labels + 1) % n_classes * target
        criterion = fb.criteria.TargetedMisclassification(target_classes)
    else:
        criterion = fb.criteria.Misclassification(labels)
        target_classes = labels

    image_path = "../../../images/"  # sarebbe da cambiare
    fig = CFigure(width=5, height=5)

    n_grid_pts = 20

    # Convenience function for plotting the decision function of a classifier
    fig.sp.plot_decision_regions(clf, n_grid_points=200, plot_background=False)
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
