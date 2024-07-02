import numpy as np
import os
import pandas as pd

root_dir = ' /home/path-to-dataset/logs/ucf101_kinetics_pretrained/'

all_dfs = []
dataset_name = 'ucf101'
for include in ['KD_one_teacher']:
    splits = []
    seeds = []
    ic = []
    accs = []
    id = []
    dataset = []
    # if dataset_name == 'hmdb51' and include == 'prog_kd':
    #     log_name = 'logs'
    # else:
    #     log_name = 'logs2'
    log_dir = os.path.join(root_dir, include)
    log_files = os.listdir(log_dir)
    for log_file in log_files:
        if include not in log_file:
            continue

        if dataset_name in log_file:
            with open(os.path.join(log_dir, log_file), "r") as f:
                file_content = f.readlines()

            file_name_split = log_file.split('_')
            dataset.extend([file_name_split[0]]*3)
            splits.extend([file_name_split[3]]*3)
            seeds.extend([file_name_split[4]]*3)

            for results in file_content[-3:]:
                ic_name = results.split()[4].strip()
                ic.append(ic_name)
                id.extend(['.'.join(file_name_split[3:5] + [ic_name])])
                acc = float(results.split()[-1].strip())
                accs.append(acc)
           
        else:
            continue

    print(accs)
    # df = pd.DataFrame(data={'dataset':dataset, 'splits':splits, 'seeds': seeds, 'id': id, 'ic':ic, f'acc_{include}': accs})
    # all_dfs.append(df)