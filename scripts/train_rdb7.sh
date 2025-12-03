#!/bin/bash

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name=train_rdb7
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/goflowv2/ || exit

mamba activate goflow

python -m goflow.flow_train -m seed=1 model.num_steps=25 model.representation.n_atom_rdkit_feats=36 task_name=train_rdb7 data=rdb7 test=False
