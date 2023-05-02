import numpy as np


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
