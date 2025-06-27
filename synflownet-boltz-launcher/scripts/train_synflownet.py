import argparse
import datetime

import torch
import wandb
import yaml
from loguru import logger

from synflownet.config import Config
from synflownet.tasks.async_reward_task import AsyncRewardTrainer

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the SynFlowNet-Boltz config file, see `configs/TYK2_config.yaml` for an example.",
    )
    args = parser.parse_args()

    # Load SynFlowNet-Boltz config
    with open(args.config) as f:
        sfn_boltz_cfg = yaml.safe_load(f)

    # Set up SynFlowNet trainingconfig
    cfg = Config.empty()
    cfg.log_dir = f"./logs/run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.replay.reward_queue_path = sfn_boltz_cfg["reward_queue_path"]
    cfg.replay.persistent_replay_path = sfn_boltz_cfg["persistent_replay_path"]
    cfg.replay.reward_queue_max_size = sfn_boltz_cfg["reward_queue_max_size"]
    cfg.replay.persistent_replay_max_size = sfn_boltz_cfg["persistent_replay_max_size"]
    cfg.replay.persistent_replay_last_to_random_ratio = 0.5
    cfg.replay.halt_training_when_queue_is_full = False
    cfg.cond.temperature.sample_dist = "constant"  # "uniform"
    cfg.cond.temperature.dist_params = [36.0]
    cfg.algo.train_random_action_prob = 0.2
    cfg.replay.warmup = 500
    cfg.print_every = 10

    # Set up logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(cfg.log_dir + "/logger.txt", level="INFO")
    logger.info(f"Loaded SynFlowNet-Boltz config from {args.config}:\n{yaml.dump(sfn_boltz_cfg)}")

    # Set up wandb
    name = cfg.log_dir.split("/")[-1]
    wandb.init(project="creatr-sfn", name=name)

    # Trainer
    logger.info("Starting training")
    trainer = AsyncRewardTrainer(cfg)
    trainer.run()
