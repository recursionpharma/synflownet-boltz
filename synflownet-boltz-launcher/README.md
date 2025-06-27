# SynFlowNet - Boltz Launcher

This pipeline assumes the existence of two separate python environments:

1. `synflownet-env`: python environment with SynFlowNet installed.
2. `boltz-env`: python environment with Boltz-2 installed.

See installation instructions in the main [README.md](../README.md#installation).

## Training Paradigm Overview

This repository is focused on training SynFlowNet models while using Boltz-2 affinity predictions as a reward function. Compared to commmonly used reward functions such as binding affinity predictors represented by small neural networks or molecular descriptors implemented in [RDKit](https://www.rdkit.org/), which can run in a few milliseconds, Boltz-2 is relatively expensive (~20s per molecule). To avoid impractical latency in the SynFlowNet training, one typically requires a large number of Boltz-2 workers to approximately keep up with the data generation rate of the SynFlowNet model.

> [!IMPORTANT] 
> **The training paradigm presented here leverages asymmetric computing ressources between SynFlowNet and Boltz-2. We recommend using around 50 Boltz-2 workers for 1 SynFlowNet training instance.**

SynFlowNet is a [GFlowNet model](https://arxiv.org/abs/2111.09266) for synthesizable molecular generation. It is thus readily compatible with *off-policy* learning. To accomodate the computational cost of running Boltz-2 as a reward function, we leverage the off-policy training capabilities of SynFlowNet through an *asynchronous* training paradigm:
- **Reward Queue**: At each training step, the SynFlowNet forward policy generates a batch of trajectories leading to terminal states representing synthetically feasible molecules. This batch of trajectories is immediately pushed to a *Reward Queue*. A flock of Boltz-2 workers can asynchronously pull batches from the Reward Queue to compute the binding affinity of the generated molecules (terminal states).
- **Replay Buffer**: Once a Boltz-2 worker is done computing the binding affinity for a given batch, the annotated (rewarded) samples are pushed back to a *Replay Buffer* where they can be used to update the SynFlowNet policy. At each training step, immediately after generating a new batch of trajectories, the SynFlowNet model thus samples an annotated batch from the Replay Buffer to compute the Trajectory Balance Loss and update its parameters. Samples used for training are obtained from the Replay Buffer both randomly and based on recentness.
- **Reward Cache**: To avoid recomputing the binding affinity for the same molecule multiple times, we cache Boltz-2 predictions in memory.

All asynchronous data structures are implemented as SQL databases to allow for safe concurrent access. See [synflownet/data/async_sql_databases.py](../synflownet/data/async_sql_databases.py) for more details.

## Training a SynFlowNet model

The code currently assumes the use of `synflownet-boltz-launcher` as the working directory. We provide two exemple scripts for launching a training run and reward workers on a SLURM cluster, which could be executed as follows:

```
cd synflownet-boltz-launcher

conda activate synflownet-env
sbatch train_synflownet.sh configs/TYK2_config.yaml

conda deactivate

conda activate boltz-env
sbatch boltz_reward_workers.sh configs/TYK2_config.yaml
```

These launching scripts contain examples for the required compute ressources and simply execute the following python scripts (respectively):

- [scripts/train_synflownet.py](../synflownet-boltz-launcher/scripts/train_synflownet.py) for training a SynFlowNet model in this asynchronous paradigm.
- [scripts/boltz_reward_worker.py](../synflownet-boltz-launcher/scripts/boltz_reward_worker.py) for running a Boltz-2 reward worker.

## Monitoring the experiment

We use `wandb` to monitor the training process. The notebook [monitoring/data_loading_example.ipynb](../synflownet-boltz-launcher/monitoring/data_loading_example.ipynb) provides an example of how to load the generated data.
