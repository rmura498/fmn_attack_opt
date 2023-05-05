import pandas as pd
import numpy as np
import os, math
import torch

from Utils.compute_robust import compute_robust, compute_best_distance
from Configs.model_dataset import MODEL_DATASET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


opt_conversion = {
    "Adam": 'Adam',
    "AdamAmsgrad": 'AdamA',
    "SGD": 'SGD',
    "SGDNesterov":'SGDN'
}

sch_conversion = {
    "CosineAnnealingLR": 'CALR',
    "CosineAnnealingWarmRestarts": 'CAWR',
    "MultiStepLR": 'MSLR',
    "ReduceLROnPlateau":'RLROP'
}


if __name__ == '__main__':
    models_ids = {}
    for model_id in MODEL_DATASET:
        models_ids[MODEL_DATASET[model_id]['model_name']] = f'M{model_id}'

    df_columns = ['Optimizer', 'Scheduler', 'Loss']
    for model_id in models_ids:
        df_columns.append(models_ids[model_id])

    BIG_DF = pd.DataFrame(columns=df_columns)

    filenames = next(os.walk("Experiments"))[1]
    filenames = [file for file in filenames if '__pycache__' not in file]

    for filename in filenames:
        splits = filename.split('_')
        splits.remove('CIFAR10')
        optimizer, scheduler, loss = splits[-3:]
        model = '_'.join(splits[:-3])

        try:
            with open(os.path.join("Experiments", filename, 'distance.pkl'), 'rb') as file:
                #distance = pickle.load(file)
                distance = torch.load(file, map_location=device)
            with open(os.path.join("Experiments", filename, 'best_adv.pkl'), 'rb') as file:
                #best_adv = pickle.load(file)
                best_adv = torch.load(file, map_location=device)
            with open(os.path.join("Experiments", filename, 'inputs.pkl'), 'rb') as file:
                #inputs = pickle.load(file)
                inputs = torch.load(file, map_location=device)
        except FileNotFoundError:
            continue

        # best_distance = compute_best_distance(best_adv, inputs)
        # distances_flat, robust_per_iter = compute_robust(distance, best_distance)

        '''
        distances_flat.sort()
        robust_per_iter.sort(reverse=True)
        idx = find_nearest(distances_flat, 8/255)
        '''

        distances = compute_best_distance(best_adv, inputs)
        acc_distances = np.linspace(0, 0.2, 500)
        robust = np.array([(distances > a).mean() for a in acc_distances])

        idx = find_nearest(distances, 8 / 255)

        std_robust = robust[idx]

        model_id = models_ids[model]

        tune_conf = BIG_DF.loc[(BIG_DF['Optimizer'] == opt_conversion[f'{optimizer}']) &
                                (BIG_DF['Scheduler'] == sch_conversion[f'{scheduler}']) &
                                (BIG_DF['Loss'] == loss)]
        robust_values = np.full(len(models_ids), None)
        robust_values[int(model_id[-1])] = std_robust

        if tune_conf.empty:
            values = [opt_conversion[f'{optimizer}'], sch_conversion[f'{scheduler}'], loss]

            values.extend(robust_values)
            new_series = pd.Series(dict(zip(df_columns, values)))

            BIG_DF = pd.concat([BIG_DF, new_series.to_frame().T], ignore_index=True)
        else:
            BIG_DF.loc[(BIG_DF['Optimizer'] == opt_conversion[f'{optimizer}']) &
                       (BIG_DF['Scheduler'] == sch_conversion[f'{scheduler}']) &
                       (BIG_DF['Loss'] == loss), [f'{model_id}']] = std_robust

    BIG_DF['Mean'] = BIG_DF.iloc[:, 3:11].mean(axis=1)
    BIG_DF.sort_values(inplace=True, by=['Optimizer', 'Scheduler', 'Loss'], ascending=['Optimizer', 'Scheduler', 'Loss'])

    print(BIG_DF)

    with open("tuning_comparison.csv", "w+") as file:
        file.writelines(BIG_DF.to_csv(index=False, float_format='%.2f'))

    with open("tuning_comparison_latex.txt", "w+") as file:
        latex_string = BIG_DF.to_latex()
        file.writelines(latex_string)
