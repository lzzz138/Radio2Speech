#!/bin/bash
  
config=config/train.yaml
list_train=lrs2_radar_csv/train.csv
list_val=lrs2_radar_csv/val.csv

tensorboard_dir=tensorboard/run_fdam2
save_ckpt=ckpt

# --master_port 29502
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2  train_FdamUnet.py \
  --config=$config \
  --list_train=$list_train \
  --list_val=$list_val \
  --tensorboard_dir=$tensorboard_dir \
  --save_ckpt=$save_ckpt