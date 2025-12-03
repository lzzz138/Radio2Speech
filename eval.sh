#!/bin/bash

# LJSpeech (TransUnet)
config=config/eval.yaml
dataset_name=LS2
vocoder_ckpt=train_nodev_ljspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl
vocoder_config=train_nodev_ljspeech_parallel_wavegan.v1/config.yml
list_val=lrs2_radar_csv/test.csv
audio_path=/home/lzq/data/lrs2_radar/audio_raw
load_best_model=ckpt/net_best.pth
save_wave_path=save_waves

CUDA_VISIBLE_DEVICES=1 python vitunet_vocoder_eval.py \
 --config=$config \
 --vocoder_ckpt=$vocoder_ckpt \
 --vocoder_config=$vocoder_config \
 --dataset_name=$dataset_name \
 --list_val=$list_val \
 --audio_path=$audio_path \
 --load_best_model=$load_best_model


 # TIMIT