import copy
import os
import pathlib

import git
import torch
from loguru import logger
from torch import Tensor

from synflownet.algo.soft_q_learning import SoftQLearning
from synflownet.algo.trajectory_balance import TrajectoryBalance
from synflownet.data.replay_buffer import ReplayBuffer
from synflownet.models.graph_transformer import GraphTransformerSynGFN
from synflownet.utils.misc import optimizer_to

from .config import Config
from .trainer import GFNTrainer


def model_grad_norm(model):
    x = 0
    for i in model.parameters():
        if i.grad is not None:
            x += (i.grad * i.grad).sum()
    return torch.sqrt(x)


class StandardOnlineTrainer(GFNTrainer):
    def setup_model(self):
        self.model = GraphTransformerSynGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "TB":
            algo = TrajectoryBalance
        elif algo == "SQL":
            algo = SoftQLearning
        else:
            raise ValueError(algo)
        self.algo = algo(self.env, self.ctx, self.cfg, self.sampler)

    def setup_data(self):
        self.training_data = []
        self.test_data = []

    def _opt(self, params, lr=None, momentum=None):
        if lr is None:
            lr = self.cfg.opt.learning_rate
        if momentum is None:
            momentum = self.cfg.opt.momentum
        if self.cfg.opt.opt == "adam":
            return torch.optim.Adam(
                params,
                lr,
                (momentum, 0.999),
                weight_decay=self.cfg.opt.weight_decay,
                eps=self.cfg.opt.adam_eps,
            )

        raise NotImplementedError(f"{self.cfg.opt.opt} is not implemented")

    def setup(self):
        super().setup()
        self.offline_ratio = 0
        if self.cfg.replay.use and not self.cfg.replay.buffer_is_async:
            self.replay_buffer = ReplayBuffer(self.cfg)
        else:
            self.replay_buffer = None
        self.sampling_hooks.append(AvgRewardHook())
        self.valid_sampling_hooks.append(AvgRewardHook())

        # Separate Z parameters from non-Z to allow for LR decay on the former
        if hasattr(self.model, "_logZ"):
            Z_params = list(self.model._logZ.parameters())
            non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        else:
            Z_params = []
            non_Z_params = list(self.model.parameters())
        self.opt = self._opt(non_Z_params)
        self.opt_Z = self._opt(Z_params, self.cfg.algo.tb.Z_learning_rate, 0.9)
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2 ** (-steps / self.cfg.algo.tb.Z_lr_decay))

        self.sampling_tau = self.cfg.algo.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        self.clip_grad_callback = {
            "value": lambda params: torch.nn.utils.clip_grad_value_(params, self.cfg.opt.clip_grad_param),
            "norm": lambda params: [torch.nn.utils.clip_grad_norm_(p, self.cfg.opt.clip_grad_param) for p in params],
            "total_norm": lambda params: torch.nn.utils.clip_grad_norm_(params, self.cfg.opt.clip_grad_param),
            "none": lambda x: None,
        }[self.cfg.opt.clip_grad_type]

        # saving hyperparameters
        try:
            self.cfg.git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        except git.InvalidGitRepositoryError:
            self.cfg.git_hash = "unknown"  # May not have been installed through git

        logger.debug(f"\nHyperparameters\n----\n{self.cfg.to_str()}")
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        with open(pathlib.Path(self.cfg.log_dir) / "config.yaml", "w", encoding="utf8") as f:
            f.write(self.cfg.to_str())

    def step(self, loss: Tensor):
        loss.backward()
        with torch.no_grad():
            g0 = model_grad_norm(self.model)
            self.clip_grad_callback(self.model.parameters())
            g1 = model_grad_norm(self.model)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
        return {"grad_norm": g0, "grad_norm_clip": g1}

    def _get_state(self, it):
        state = super()._get_state(it)
        state["optimizer_state_dict"] = self.opt.state_dict()
        state["optimizer_Z_state_dict"] = self.opt_Z.state_dict()
        state["lr_scheduler"] = self.lr_sched.state_dict()
        state["lr_scheduler_Z"] = self.lr_sched_Z.state_dict()
        return state

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.sampling_model.to(device)
        self.opt = optimizer_to(self.opt, device)
        self.opt_Z = optimizer_to(self.opt_Z, device)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **trainer_kwargs):
        state = torch.load(checkpoint_path)
        config = Config.empty().merge(state["cfg"])
        trainer = cls(config, **trainer_kwargs)
        trainer.model.load_state_dict(state["models_state_dict"])
        if "sampling_model_state_dict" in state:
            trainer.sampling_model.load_state_dict(state["sampling_model_state_dict"])
        else:
            if trainer.sampling_tau > 0:
                trainer.sampling_model = copy.deepcopy(trainer.model)
            else:
                trainer.sampling_model = trainer.model
        if "optimizer_state_dict" in state:
            trainer.opt.load_state_dict(state["optimizer_state_dict"])
        if "optimizer_Z_state_dict" in state:
            trainer.opt_Z.load_state_dict(state["optimizer_Z_state_dict"])
        if "lr_scheduler" in state:
            trainer.lr_sched.load_state_dict(state["lr_scheduler"])
        if "lr_scheduler_Z" in state:
            trainer.lr_sched_Z.load_state_dict(state["lr_scheduler_Z"])
        trainer.cfg.start_at_step = state["step"]
        logger.info(f"Loaded checkpoint at {checkpoint_path}. Starting at step {trainer.cfg.start_at_step}")
        trainer.to(trainer.device)
        return trainer


class AvgRewardHook:
    def __call__(self, trajs, rewards, obj_props, extra_info):
        return {"sampled_reward_avg": rewards.mean().item()}
