import pandas as pd
import numpy as np
import os, pickle, math
import torch

from Utils.compute_robust import compute_robust, compute_best_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


if __name__ == '__main__':
    models_confs = {}

    filenames = next(os.walk("Experiments"))[1]
    filenames = [file for file in filenames if '__pycache__' not in file]

    for filename in filenames:
        splits = filename.split('_')
        splits.remove('CIFAR10')
        optimizer, scheduler, loss = splits[-3:]
        model = '_'.join(splits[:-3])

        try:
            with open(os.path.join("Experiments", filename, 'distance.pkl'), 'rb') as file:
                # distance = pickle.load(file)
                distance = torch.load(file, map_location=device)
            with open(os.path.join("Experiments", filename, 'best_adv.pkl'), 'rb') as file:
                best_adv = torch.load(file, map_location=device)
            with open(os.path.join("Experiments", filename, 'inputs.pkl'), 'rb') as file:
                inputs = torch.load(file, map_location=device)
        except FileNotFoundError:
            continue

        best_distance = compute_best_distance(best_adv, inputs)
        distances_flat, robust_per_iter = compute_robust(distance, best_distance)

        distances_flat.sort()
        robust_per_iter.sort(reverse=True)
        idx = find_nearest(distances_flat, 8/255)
        std_robust = robust_per_iter[idx]

        model_conf_key = f'{optimizer}_{scheduler}_{loss}'
        if model not in models_confs:
            models_confs[model] = {}

        models_confs[model][model_conf_key] = std_robust

    models_confs_df = pd.DataFrame(models_confs)

    with open("tuning_comparison_latex.txt", "w+") as file:
        latex_string = models_confs_df.to_latex(index=True, float_format="{:.4f}".format)
        file.writelines(latex_string)
