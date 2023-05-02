import os, argparse



parser = argparse.ArgumentParser(description='Retrieve tuning params')
parser.add_argument('-b', '--batch',
                    default=32,
                    help='Provide the batch size')
parser.add_argument('-s', '--steps',
                    default=100,
                    help='Provide the step size')
parser.add_argument('-dp', '--dataset_percent',
                    default=0.5,
                    help='Provide the dataset percentage to be used to tune the hyperparams')

args = parser.parse_args()






if __name__ == '__main__':
    batch = str(args.batch)
    steps = str(args.steps)
    dt_percent = str(args.dataset_percent)

    tuning_cmds = []
    filenames = os.listdir("./Configs/ModelsBestConfigs")
    for filename in filenames:
        print(filename)
        tuning_cmd = f'python run_attack.py --batch {batch} --steps {steps} --dataset_percent {dt_percent} --fmn_config {filename}\n'
        tuning_cmds.append(tuning_cmd)
    with open("run_attack_cmd.txt", 'w') as f:
        f.writelines(tuning_cmds)