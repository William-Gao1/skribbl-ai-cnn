#!/bin/bash

#SBATCH --gpus=1
#SBATCH --mem=90G

source ../.venv/bin/activate

export LD_LIBRARY_PATH=~/src/skribbl-ai-model/.venv/lib/python3.11/site-packages/tensorrt_libs

python train.py
python convert_to_lite.py

