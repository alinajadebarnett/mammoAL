#!/bin/bash
#SBATCH  --gres=gpu:p100:1 -p compsci-gpu --time=10-00:00:00

source /usr/xtmp/vs196/mammoproj/Env/trainenv6/bin/activate
echo "start running"
nvidia-smi

python getResults.py