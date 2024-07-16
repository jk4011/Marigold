#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1

export BASE_DATA_DIR=/root/wlsgur4011/marigold_data  # directory of training data
export BASE_CKPT_DIR=/data2/wlsgur4011/Marigold/checkpoint  # directory of pretrained checkpoint

# command 1
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config config/train_marigold.yaml \
    --three_modality \
    --no_wandb
    # --resume_run ./output/train_marigold/checkpoint/latest \

# ... 



















set +x; duration=SECONDS; RED='\033[0;31m'; Yellow='\033[1;33m'; Green='\033[0;32m'; NC='\033[0m'; echo -e "RED$((duration / 3600))hNC Yellow$((duration / 60 % 60))mNC Green$((duration % 60))sNC elapsed."