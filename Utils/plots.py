import torch
import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def normalize_01(data):
    min_val = np.min(data)
    max_val = np.max(data)

    if min_val == max_val:
        return np.divide(data, data)
    else:
        return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_loss_epsilon_over_steps(loss,
                                 epsilon,
                                 steps,
                                 attack_name='attack',
                                 model_name='model',
                                 normalize=True):
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
    ax1.title.set_text(f"Attack: {attack_name}, Model: {model_name}")
    fig1.legend()

    plt.show()