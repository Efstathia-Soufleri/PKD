# Progressive KD and WISE

Using the official implementation of paper [TEAM-Net](https://github.com/villawang/TEAM-Net) as a back bone for early exits with training code for Progressive KD and Cross Entropy as the baseline.

## Environment
This code is tested and validated with py3.6 **ONLY** due to the dependency on [CoViAR](https://github.com/chaoyuaw/pytorch-coviar/) code. To replicate our environment please use the `environment.yml` file provided by running
```
conda env update -n py3.6_prog_kd --file environment.yml
```

For `team_net_wise_all_acc.py` we need newer pytorch features, so please use `py39env.yml` environment for running this script.

## Replicating Experiments on TEAM-NET

We use the `./multirunner` directory to launch all of our experiments the exact scripts are provided in `./multirunner`.
We documentation here the steps for running our experiments on single or multiple hosts.

1. Create a `hosts.txt` file and specify the hostnames, make sure to **add and empty line at the end of the file**
2. Generate `all_jobs.sh` by running  
```
sh create_all_jobs.sh > all_jobs.sh
```
**add and empty line at the end of the file**

3. For a single node run use:
```
sh all_jobs.sh
```
4. For running on multiple hosts, run the command below
```
sh multirunner.sh hosts.txt all_jobs.sh
```
Example `hosts.txt` and `all_jobs.sh` are provided to run training on some example hosts is provided. Make sure to set the working directory `wd`, python environment `environment_name` in the `multirunner.sh` and the `data_root` in `create_all_jobs.sh`

Implementation
---
Our implementations are described below

| File     | Description         |
|----------|---------------------|
| `train_team_ce.py` | Trains IC attached to TEAM-Net using cross entropy |
| `train_prog_kd.py` | Trains IC attached to TEAM-Net using curriculum progressive KD |
| `train_prog_kd_anti.py` | Trains IC attached to TEAM-Net using anti-curriculum progressive KD |
| `train_i_frame.py` | Trains IC attached to TEAM-Net using I-Frame final classifier as teacher |
| `save_logits.py` | For running wise it is more efficient to save the logits and then compute $\beta$ so this saves the logits from models. |
| `wise_team_net_logit.py` | Runs WISE for the seed and split specified |
| `team_net_wise_all_acc.py` | Applies WISE and threshold to get the accuracy of the early exits on TEAM-Net backbone (this requires py3.9 environment) |
| `trade_off_all.ipynb` | Plots the trade off of our method and different methods |
| `train_team_net.py` | Use these to train the backbones for UCF-101 and HMDB-51|
| `./MACs/main.py`| Calculate the MAC operations for different methods  |

Calculate fair MAC operations for various techniques
---
Run the `main.py` file after setting the path for the logits
```
python main.py
```
Running training code
---
1. Download pretrained model from [here (will be updated for final publication)](https://example.com/). To pretrain models please see the pretraining section.
2. Create `checkpoints` in the root directory with the follwing structure

```
TEAM_NET_Backbone
  +-- checkpoints
  |     +-- hmdb51
  |     |     
  |     +-- ucf101
  ...
  +-- dataset_team
  ...
```
3. Run the command below for testing things are setup correctly
```
python train_team_ce.py --lr 0.02 --is_train --checkpoint_dir ./checkpoints --split split1 --random_seed 1 --batch-size 128 --arch resnet50 --data-name ucf101 --data-root /home/path-to-dataset/ucf101 --train-list /home/path-to-dataset/ucf101/ucf101_split1_train.txt --test-list /home/path-to-dataset/ucf101/ucf101_split1_test.txt --lr-steps 5 10 15 --epochs 30 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers=4
```
4. For the exact commands to re-create our experiments you can run look at and run `./multirunner/all_jobs.sh`

Pretraining
---
Use the following command to pretrain the models
1. Pretrain on Kinetics
```
python pretrain.py \
  --lr 0.01 \
  --is_train \
  --batch-size 32 \
  --arch resnet50 \
  --data-name kinetic400 \
  --data-root <path to mpeg4_videos> \
  --train-list <path to train annotations> \
  --test-list <path to val annotations> \
  --lr-steps 30 40 45 --epochs 50 --wd 5e-4 \
  --gpus 0 --num_segments 8 --eval-freq 1 --dropout 0.5    
```
2. Use these to train the backbones for UCF-101 and HMDB-51
```
python train_team_net.py --lr 0.02 \
  --is_shift\
  --batch-size 16 \
  --arch resnet50 \
  --data-name ucf101 \
  --data-root <path to mpeg4_videos> \
  --train-list <path to train annotations> \
  --test-list <path to val annotations> \
  --lr-steps 5 10 15 --epochs 30 \
  --num_segments 8 --dropout 0.5 --wd 1e-4 \
  --eval-freq 1 --gpus 0 --workers 20

python train_team_net.py --lr 0.02 \
  --is_train --split $split \
  --data-name hmdb51 \
  --batch-size 16 \
  --arch resnet50 \
  --data-root <path to mpeg4_videos> \
  --train-list <path to train annotations> \
  --test-list <path to val annotations> \
  --lr-steps 10 15 20 --epochs 30 \
  --num_segments 8 --dropout 0.5 --wd 5e-4 \
  --eval-freq 1 --gpus 0

```

 