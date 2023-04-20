import os

from Configs.model_dataset import MODEL_DATASET

OPTIMIZERS = ["SGD","SGDNesterov","Adam"]

SCHEDULERS = [
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "MultiStepLR",
    "ReduceLROnPlateau"
]

if __name__ == '__main__':
    batch = 100
    steps = 100
    num_s = 20
    epochs = 5
    dt_percent=50

    tuning_cmds = []
    for model_id in MODEL_DATASET:
        model = MODEL_DATASET[model_id]
        model_name = model['model_name']
        datasets = model['datasets']

        for dataset_id, dataset_name in enumerate(datasets):
            tuning_exp_wp = os.path.join("TuningExp", f"{model_name}_{dataset_name}")

            for opt in OPTIMIZERS:
                for sch in SCHEDULERS:
                    tuning_cmd = f'python tune.py --model_id {model_id} --dataset_id {dataset_id} --optimizer {opt}\'' \
                                 f'--scheduler {sch} --batch {batch} --steps {steps} --num_samples {num_s} --epochs {epochs}\'' \
                                 f'--dataset_percent {dt_percent} --working_path {tuning_exp_wp}'
                    tuning_cmds.append(tuning_cmd)

    print(tuning_cmds)
    #os.cmd...