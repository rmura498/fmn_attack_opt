import os, argparse

from Configs.model_dataset import MODEL_DATASET

OPTIMIZERS = ["SGD","SGDNesterov","Adam",'AdamAmsgrad']

SCHEDULERS = [
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "MultiStepLR",
    "ReduceLROnPlateau"
]

parser = argparse.ArgumentParser(description='Retrieve tuning params')
parser.add_argument('-b', '--batch',
                    default=50,
                    help='Provide the batch size')
parser.add_argument('-s', '--steps',
                    default=100,
                    help='Provide the step size')
parser.add_argument('-n_s', '--num_samples',
                    default=1,
                    help='Provide the number of trials to execute')
parser.add_argument('-ep', '--epochs',
                    default=1,
                    help='Provide the epochs (how many times the attack is executed per trial)')
parser.add_argument('-dp', '--dataset_percent',
                    default=0.5,
                    help='Provide the dataset percentage to be used to tune the hyperparams')

args = parser.parse_args()

if __name__ == '__main__':
    batch = str(args.batch)
    steps = str(args.steps)
    num_s = str(args.num_samples)
    epochs = str(args.epochs)
    dt_percent = str(args.dataset_percent)

    tuning_cmds = []
    for model_id in MODEL_DATASET:
        model = MODEL_DATASET[model_id]
        model_name = model['model_name']
        datasets = model['datasets']

        for dataset_id, dataset_name in enumerate(datasets):
            tuning_exp_wp = os.path.join("TuningExp", f"{model_name}_{dataset_name}")

            for opt in OPTIMIZERS:
                for sch in SCHEDULERS:
                    for i in range(2):
                        tuning_cmd = f'python tune_pbt.py --model_id {model_id} --dataset_id {dataset_id} --optimizer {opt} --scheduler {sch} --batch {batch} --steps {steps} --num_samples {num_s} --epochs {epochs} --dataset_percent {dt_percent} --working_path {tuning_exp_wp} --loss {i}\n'
                        tuning_cmds.append(tuning_cmd)

    with open("tuning_pbt_cmd.txt", 'w') as f:
        f.writelines(tuning_cmds)

