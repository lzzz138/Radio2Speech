#!/bin/bash
  
config=config/train.yaml
list_train=lrs2_radar_csv/train.csv
list_val=lrs2_radar_csv/val.csv

tensorboard_dir=tensorboard
save_ckpt=ckpt

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train_Radio2Speech.py \
  --config=$config \
  --list_train=$list_train \
  --list_val=$list_val \
  --tensorboard_dir=$tensorboard_dir \
  --save_ckpt=$save_ckpt