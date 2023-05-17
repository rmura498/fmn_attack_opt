import os
import pickle

import pandas as pd

MODEL_DATASET = {
    0: {
        'model_name': 'Wang2023Better_WRN-70-16',
        'datasets': ['cifar10']
        },
    1: {'model_name': 'Wang2023Better_WRN-28-10',
        'datasets': ['cifar10']
        },
    2: {'model_name': 'Gowal2021Improving_70_16_ddpm_100m',
        'datasets': ['cifar10']
        },
    3: {'model_name': 'Rebuffi2021Fixing_106_16_cutmix_ddpm',
        'datasets': ['cifar10']
        },
    4: {'model_name': 'Gowal2021Improving_28_10_ddpm_100m',
        'datasets': ['cifar10']
        },
    5: {'model_name': 'Pang2022Robustness_WRN70_16',
        'datasets': ['cifar10']
        },
    6: {'model_name': 'Sehwag2021Proxy_ResNest152',
        'datasets': ['cifar10']
        },
    7: {'model_name': 'Pang2022Robustness_WRN28_10',
        'datasets': ['cifar10']
        },
    8: {'model_name': 'Gowal2021Improving_R18_ddpm_100m',
        'datasets': ['cifar10']
        }

}

models_ids = {}
for model_id in MODEL_DATASET:
    models_ids[MODEL_DATASET[model_id]['model_name']] = f'M{model_id}'

configs = os.listdir("../Configs/ModelsBestConfigs/final_configs")
configs.remove('read_final_confs.py')

best_conf = {}
for conf in configs:
    with open(conf, 'rb') as file:
        fmn_config = pickle.load(file)
        conf_name = conf.split('.')[0]
        splits = conf_name.split('_')
        splits.remove('cifar10')
        model_name = '_'.join(splits[:-3])
        model_name = models_ids[model_name]
        best_conf[model_name] = {}

        best_conf[model_name]['opt'] = fmn_config['best_config']['opt_s']
        best_conf[model_name]['sch'] = fmn_config['best_config']['sch_s']

best_conf_df = pd.DataFrame.from_dict(best_conf, orient='index')
best_conf_df.sort_index(inplace=True)
print(best_conf_df)
best_conf_df.to_csv("comparing_final_res.csv")