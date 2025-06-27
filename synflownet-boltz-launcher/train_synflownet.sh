#!/bin/bash

# Purpose: Script to launch a synflownet training
# Usage: sbatch train_sfn.sh

#SBATCH --job-name=sfn_trainer
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a_%N.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=20
#SBATCH --qos=high
#SBATCH --mem=32GB
#SBATCH --partition def

conda activate synflownet-env
echo "Using environment={$CONDA_DEFAULT_ENV}"
python scripts/train_synflownet.py --config $1
