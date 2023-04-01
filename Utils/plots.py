import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt

from .metrics import loss_fmn_fn


def plot_epsilon_robust(epsilon_per_iter, steps, batch_size):
    if len(epsilon_per_iter) == 0:
        return

    final_eps = []
    robust_per_iter = []

    for exp_epsilons in epsilon_per_iter:
        for epsilon in exp_epsilons:
            for eps in epsilon:
                print('epsilon',epsilon, "\n")
                robust = np.count_nonzero(eps > exp_epsilons[-1]) / len(epsilon)
                print(robust, "\n")
                eps=eps.numpy()
                final_eps.append(eps/100)
                robust_per_iter.append(robust)
    final_eps=np.array(final_eps)
    final_eps=np.sort(final_eps)[::-1]
    robust_per_iter = np.sort(robust_per_iter)[::-1]

    fig, axs = plt.subplots()
    axs.plot(
             np.linspace(1, steps*batch_size, steps*batch_size),

             robust_per_iter,
             #final_eps,
             label='robust'
             )
    axs.plot(
             np.linspace(1,steps*batch_size, steps*batch_size),
             final_eps,
             label='epsilon'
             )
    fig.legend()
    plt.show()


def plot_loss_epsilon_over_steps(loss=None,
                                 epsilon=None,
                                 distance_to_boundary=None,
                                 steps=10,
                                 batch_size=10,
                                 norm=None,
                                 attack_name='attack',
                                 model_name='model',
                                 optimizer='-',
                                 path="."):
    fig1, ax1 = plt.subplots()
    if loss is not None:
        ax1.plot(
            torch.arange(0, steps),
            loss,
            label="Loss"
        )
    if epsilon is not None:
        ax1.plot(
            torch.arange(0, steps),
            epsilon,
            label="Epsilon"
        )
    if norm != 0 and distance_to_boundary is not None:
        ax1.plot(
            torch.arange(0, steps),
            distance_to_boundary,
            label="Distance to boundary"
        )
    ax1.grid()
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss/Epsilon/Distance to boundary")
    ax1.title.set_text(f"Attack: {attack_name}, Model: {model_name}, Optimizer: {optimizer}")
    fig1.legend()
    fig1.suptitle(f"Steps: {steps}, Batch: {batch_size}, Norm: {norm}")

    plt.show()

    plot_name = f'plot_{steps}_{norm}'
    fig1.savefig(os.path.join(path, f"{plot_name}.png"))


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
