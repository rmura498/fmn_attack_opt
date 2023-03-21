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


def plot_loss_epsilon_over_steps(loss,
                                 epsilon,
                                 distance_to_boundary=None,
                                 steps=10,
                                 norm=None,
                                 attack_name='attack',
                                 model_name='model',
                                 normalize=True):

    if normalize:
        loss = normalize_01(loss)
        loss = loss - loss.mean()
        epsilon = normalize_01(epsilon)

        if norm != 0:
            distance_to_boundary = normalize_01(distance_to_boundary)

    fig1, ax1 = plt.subplots()
    ax1.plot(
        torch.arange(0, steps),
        loss,
        label="Loss"
    )
    ax1.plot(
        torch.arange(0, steps),
        epsilon,
        label="Epsilon"
    )
    if norm != 0:
        ax1.plot(
            torch.arange(0, steps),
            distance_to_boundary,
            label="Distance to boundary"
        )
    ax1.grid()
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss/Epsilon/Distance to boundary")
    ax1.title.set_text(f"Attack: {attack_name}, Model: {model_name}, Norm: {norm}")
    fig1.legend()

    plt.show()