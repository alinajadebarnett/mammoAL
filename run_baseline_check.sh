#!/bin/bash
#SBATCH  --gres=gpu:p100:1 -p compsci-gpu --time=10-00:00:00

source /usr/xtmp/vs196/mammoproj/Env/trainenv6/bin/activate
echo "start running"
nvidia-smi

python baseline_check.py --unet --task_id 6015 --run_id 1_08_test20 --output_dir /usr/xtmp/vs196/ReRunOutputs/AllOracleRuns  --random_seed 44 --query_method "best"