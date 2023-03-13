import numpy as np
import matplotlib as plt
import torch


def normalize_01(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_loss_epsilon_over_steps(loss, epsilon, steps, normalize=True):
    if normalize:
        loss = normalize_01(loss)
        epsilon = normalize_01(epsilon)

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
    ax1.grid()
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss/Epsilon")
    fig1.legend()

    plt.show()