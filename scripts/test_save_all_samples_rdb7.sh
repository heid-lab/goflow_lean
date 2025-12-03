#!/bin/bash

#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_save_all_samples
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/goflowv2 || exit

mamba activate goflow

MODEL_PATH="/home/leonard.galustian/projects/goflowv2/logs/train_rdb7/multiruns/2025-12-02_07-54-39/0/checkpoints/epoch_316.ckpt"

python -m goflow.flow_train -m model.num_samples=25 model.num_steps=25 model.representation.n_atom_rdkit_feats=36 model.save_all_samples=True task_name=test_save_all_samples train=False data=rdb7 custom_model_weight_path=$MODEL_PATH
