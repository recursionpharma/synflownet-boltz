#!/bin/bash

# Purpose: Script to launch an array of reward workers serving synflownet training
# Usage: sbatch boltz_reward_workers.sh

#SBATCH --job-name=boltz_reward_wrk
#SBATCH --array=1-50
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a_%N.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=20
#SBATCH --qos=normal
#SBATCH --mem=32GB
#SBATCH --partition def

conda activate boltz-env
echo "Using environment={$CONDA_DEFAULT_ENV}"
python scripts/boltz_reward_worker.py --config $1
