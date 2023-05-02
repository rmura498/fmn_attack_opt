import numpy as np
import torch


def compute_best_distance(best_adv, inputs):
    best_distance = torch.linalg.norm((best_adv - inputs).data.flatten(1), dim=1, ord=float('inf'))
    return best_distance


def compute_robust(exp_distances, best_distance):
    batch_size = len(best_distance)

    distances = np.array([])
    robust_per_iter = []
    for distance in exp_distances:
        # checking, for each step, the epsilon tensor
        distances = np.concatenate((distances, distance.numpy()), axis=None)
        robust_per_iter += [
            (np.count_nonzero(dist > best_distance) / batch_size)
            for dist in distance
        ]

    return distances, robust_per_iter
