import pickle
import torch


try:
    with open("./Configs/ModelsBestConfigs/final_configs/Wang2023Better_WRN-70-16_cifar10_Adam_ReduceLROnPlateau_LL.pkl", 'rb') as file:
        fmn_config = pickle.load(file)
except Exception as e:
    print("Cannot load the configuration:")
    exit(1)


optimizer_config = fmn_config['best_config']['opt_s']
scheduler_config = fmn_config['best_config']['sch_s']
print(optimizer_config)
print(scheduler_config)