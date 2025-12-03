#!/bin/bash

#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_samples_analysis
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/goflowv2 || exit

mamba activate goflow

SAMPLE_PATH="/home/leonard.galustian/projects/goflowv2/logs/test_save_all_samples/multiruns/2025-12-02_08-54-04/0/test_samples/samples_all.pkl"

python -m goflow.test_samples_analysis $SAMPLE_PATH rdb7
