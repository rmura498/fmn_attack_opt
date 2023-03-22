import os.path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize_01(data):
    min_val = np.min(data)
    max_val = np.max(data)

    if min_val == max_val:
        return np.divide(data, 100)
    else:
        return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_loss_epsilon_over_steps(loss=None,
                                 epsilon=None,
                                 distance_to_boundary=None,
                                 steps=10,
                                 norm=None,
                                 attack_name='attack',
                                 model_name='model',
                                 normalize=True,
                                 translate_loss=True,
                                 translate_distance=True,
                                 path="."):

    if normalize:
        if loss is not None:
            loss = normalize_01(loss)
            if translate_loss:
                loss = loss - loss.mean()
        if epsilon is not None:
            epsilon = normalize_01(epsilon)

        if norm != 0 and distance_to_boundary is not None:
            distance_to_boundary = normalize_01(distance_to_boundary)
            if translate_distance:
                distance_to_boundary = distance_to_boundary - distance_to_boundary.mean()

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
    ax1.title.set_text(f"Attack: {attack_name}, Model: {model_name}")
    fig1.legend()
    fig1.suptitle(f"Steps: {steps}, Norm: {norm}")

    plt.show()

    '''
    if path is None:
        time = datetime.now().strftime("%d%H%M")
        path = os.path.join("Experiments", "Plots")
        experiment = f'Exp_{time}_{attack_name}_{model_name}'
        path = os.path.join(path, experiment)
        if not os.path.exists(path):
            os.makedirs(path)
    '''

    plot_name = f'plot_{steps}_{norm}'
    fig1.savefig(os.path.join(path, f"{plot_name}.png"))