#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

mkdir checkpoint
cd checkpoint
wget https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt