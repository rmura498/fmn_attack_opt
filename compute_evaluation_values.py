import os, argparse

import torch
import numpy as np

from Utils.compute_robust import compute_best_distance, compute_robust

parser = argparse.ArgumentParser(description='Retrieve experiment data')
parser.add_argument('-ep', '--exp_name',
                    required=True,
                    help='Provide the experiment folder name (e.g. Model_Dataset_Opt_Sch_Loss) which \
                         can be retrieved inside the ./Experiments folder')

args = parser.parse_args()


if __name__ == '__main__':
    exp_name = args.exp_name
    exp_path = os.path.join("Experiments", exp_name)

    exp_data = {
        'distance': [],
        'inputs': [],
        'best_adv': []
    }

    for data in exp_data:
        data_path = os.path.join(exp_path, f"{data}.pkl")
        data_load = torch.load(data_path, map_location=torch.device('cpu'))
        exp_data[data] = data_load

    inputs = exp_data['inputs']
    best_adv = exp_data['best_adv']

    batch_number = len(inputs)
    batch_size = len(best_adv[0])

    distances = np.empty((batch_number, batch_size))
    for b in range(batch_number):
        distance = compute_best_distance(best_adv[b], inputs[b])
        distances[b, :] = distance

    distances = distances.ravel()

    acc_distances = np.linspace(0, 0.2, 500)
    robust = np.array([(distances>a).mean() for a in acc_distances])

    '''
    (distances > acc_distances[np.newaxis, :].T).mean(axis=1)
    '''


    '''
    best_distances = distances.mean(axis=0)

    distances = [x for l in exp_data['distance'] for x in l]
    acc_distances, robust = compute_robust(distances, best_distances)

    acc_distances = np.array(acc_distances)
    '''

    # acc_distances.sort()
    # robust.sort(reverse=True)

    # save merged distances and best distances (mean over best dists) as pkl
    torch.save(acc_distances, os.path.join(exp_path, 'sorted_distances.pkl'))
    torch.save(robust, os.path.join(exp_path, 'sorted_robust.pkl'))
    # torch.save(best_distances, os.path.join(exp_path, 'best_distances.pkl'))