import gc
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
import wandb
from loguru import logger
from rdkit import RDLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from synflownet import GFNAlgorithm, GFNTask
from synflownet.algo.config import Backward
from synflownet.algo.graph_sampling import Sampler
from synflownet.data.data_source import DataSource
from synflownet.data.replay_buffer import ReplayBuffer
from synflownet.envs.graph_building_env import GraphActionCategorical
from synflownet.envs.synthesis_building_env import (
    ReactionTemplateEnv,
    ReactionTemplateEnvContext,
)
from synflownet.utils.misc import set_main_process_device, set_worker_rng_seed
from synflownet.utils.multiprocessing_proxy import BufferUnpickler, mp_object_wrapper
from synflownet.utils.sqlite_log import SQLiteLogHook

from .config import Config


class Closable(Protocol):
    def close(self):
        pass


class GFNTrainer:
    def __init__(self, config: Config):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        config: Config
            The hyperparameters for the trainer.
        """
        self.to_terminate: list[Closable] = []
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        self.replay_buffer: Optional[ReplayBuffer]
        self.env: ReactionTemplateEnv
        self.ctx: ReactionTemplateEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm
        self.sampler: Sampler

        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method, typically what is defined by a task
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources with the following precedence:
        #   config classes < default_hps < constructor (i.e. the constructor overrides the default_hps, and so on)
        default_cfg: Config = Config()
        self.set_default_hps(default_cfg)
        assert isinstance(default_cfg, Config) and isinstance(
            config, Config
        )  # make sure the config is a Config object, and not the Config class itself
        self.cfg: Config = default_cfg.merge(config)

        self.device = torch.device(self.cfg.device)
        set_main_process_device(self.device)
        # Print the loss every `self.print_every` iterations
        self.print_every = self.cfg.print_every
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: list[Callable] = []
        self.valid_sampling_hooks: list[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup()

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup_env_context(self):
        raise NotImplementedError()

    def setup_task(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_algo(self):
        raise NotImplementedError()

    def setup_sampler(self):
        raise NotImplementedError()

    def setup_data(self):
        pass

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def setup(self):
        RDLogger.DisableLog("rdApp.*")
        set_worker_rng_seed(self.cfg.seed)
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_sampler()
        self.setup_algo()
        self.setup_model()

    def _wrap_for_mp(self, obj):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if self.cfg.num_workers > 0 and obj is not None:
            wrapper = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=self.cfg.pickle_mp_messages,
                sb_size=self.cfg.mp_buffer_size,
            )
            self.to_terminate.append(wrapper.terminate)
            return wrapper.placeholder
        else:
            return obj

    def build_callbacks(self):
        return {}

    def _make_data_loader(self, src):
        return torch.utils.data.DataLoader(
            src,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_training_data_loader(self) -> DataLoader:
        # Since the model may be used by a worker in a different process, we need to wrap it.
        # See implementation_notes.md for more details.
        model = self._wrap_for_mp(self.sampling_model)
        replay_buffer = self._wrap_for_mp(self.replay_buffer)

        if self.cfg.replay.use:
            # None is fine for either value, it will be replaced by num_from_policy, but 0 is not
            assert self.cfg.replay.num_new_samples != 0, "Replay is enabled but no new samples are being added to it"

        n_drawn = self.cfg.algo.num_from_policy
        # n_replayed = self.cfg.replay.num_from_replay or n_drawn if self.cfg.replay.use else 0
        n_replayed = self.cfg.replay.num_from_replay if self.cfg.replay.use else 0
        n_new_replay_samples = self.cfg.replay.num_new_samples or n_drawn if self.cfg.replay.use else None
        n_from_dataset = self.cfg.algo.num_from_dataset
        num_from_buffer_for_pb = self.cfg.algo.num_from_buffer_for_pb
        backward_policy = self.cfg.algo.tb.backward_policy

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, replay_buffer=replay_buffer)
        if n_from_dataset:
            src.do_sample_dataset(self.training_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            src.do_sample_model(model, n_drawn, n_new_replay_samples)
        if n_replayed and replay_buffer is not None:
            src.do_sample_replay(n_replayed)
        if n_replayed and hasattr(src, "reward_queue"):
            src.do_sample_async_replay(model, n_new_replay_samples, n_replayed)
        if num_from_buffer_for_pb and backward_policy == Backward.REINFORCE:
            assert self.cfg.replay.use, "REINFORCE for the backward policy requires a replay buffer"
            src.do_sample_backward(num_from_buffer_for_pb, backwards_model=model)
        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_validation_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        n_drawn = self.cfg.algo.valid_num_from_policy
        n_from_dataset = self.cfg.algo.valid_num_from_dataset

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        if n_from_dataset:
            src.do_dataset_in_order(self.test_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            assert self.cfg.num_validation_gen_steps is not None
            src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_validation_gen_steps * n_drawn)

        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "valid"), self.ctx))
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_final_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)

        n_drawn = self.cfg.algo.num_from_policy
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        assert self.cfg.num_final_gen_steps is not None
        src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_final_gen_steps * n_drawn)

        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "final"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def _maybe_resolve_shared_buffer(self, batch: Union[Batch, tuple, list], dl: DataLoader) -> Batch:
        if dl.dataset.mp_buffer_size and isinstance(batch, (tuple, list)):
            batch, wid = batch
            batch = BufferUnpickler(dl.dataset.result_buffer[wid], batch, self.device).load()
        elif isinstance(batch, Batch):
            batch = batch.to(self.device)
        return batch

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> dict[str, Any]:
        tick = time.time()
        self.model.train()
        try:
            loss, info = self.algo.compute_batch_losses(self.model, batch)
            if not torch.isfinite(loss):
                raise ValueError("loss is not finite")
            step_info = self.step(loss)
            self.algo.step()  # This also isn't used anywhere?
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError("parameters are not finite")
        except ValueError as e:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            torch.save(
                [self.model.state_dict(), batch, loss, info],
                open(self.cfg.log_dir + "/dump.pkl", "wb"),
            )
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["train_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> dict[str, Any]:
        tick = time.time()
        self.model.eval()
        loss, info = self.algo.compute_batch_losses(self.model, batch)

        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["eval_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self, use_tqdm: bool = False):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader() if valid_freq is not None else None
        final_dl = self.build_final_data_loader() if self.cfg.num_final_gen_steps else None
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        start_time = time.time()
        for it, batch in tqdm(
            zip(range(start, 1 + num_training_steps), cycle(train_dl)),
            total=num_training_steps,
            disable=not (use_tqdm),
        ):
            if batch is None:
                logger.warning(f"iteration {it} : no batch returned from data source. Continuing ...")
                continue

            # the memory fragmentation or allocation keeps growing, how often should we clean up?
            # is changing the allocation strategy helpful?
            if it % 1024 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            batch = self._maybe_resolve_shared_buffer(batch, train_dl)

            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(f"iteration {it} : warming up replay buffer {len(self.replay_buffer):,}/{self.replay_buffer.warmup:,}")
                continue
            if (
                hasattr(train_dl.dataset, "persistent_replay_buffer")
                and train_dl.dataset.persistent_replay_buffer.get_db_size(fast=True) < self.cfg.replay.warmup
            ):
                logger.info(
                    f"iteration {it} : warming up persistent replay buffer "
                    f"{train_dl.dataset.persistent_replay_buffer.get_db_size(fast=True):,}/{self.cfg.replay.warmup:,}"
                )
                continue
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            info["time_spent"] = time.time() - start_time
            if hasattr(train_dl.dataset, "persistent_replay_buffer"):
                info["persistent_replay_buffer_size"] = train_dl.dataset.persistent_replay_buffer.get_db_size(fast=True)
                info["reward_queue_size"] = train_dl.dataset.reward_queue.get_db_size(fast=False)
            start_time = time.time()
            self.log(info, it, "train")
            if it % self.print_every == 0:
                if self.cfg.verbose:
                    logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
                else:
                    verbose_info = {k: v for k, v in info.items() if k in ["loss", "sampled_reward_avg"]}
                    logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in verbose_info.items()))

            if valid_freq is not None and it % valid_freq == 0:
                valid_step_outputs = []
                for batch in valid_dl:
                    batch = self._maybe_resolve_shared_buffer(batch, valid_dl)
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, "valid")
                    if self.cfg.verbose:
                        logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
                    else:
                        verbose_info = {k: v for k, v in info.items() if k in ["loss", "sampled_reward_avg"]}
                        logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in verbose_info.items()))
                    valid_step_outputs.append({"batch": batch, "info": info})
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        # c.on_validation_end(end_metrics)
                        end_metrics.update(c.on_validation_end(valid_step_outputs))
                self.log(end_metrics, it, "valid_end")
            if ckpt_freq is not None and it % ckpt_freq == 0:
                self._save_state(it)
        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        final_info = {}
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(num_training_steps + 1, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                batch = self._maybe_resolve_shared_buffer(batch, final_dl)
                if hasattr(batch, "extra_info"):
                    for k, v in batch.extra_info.items():
                        if k not in final_info:
                            final_info[k] = []
                        if hasattr(v, "item"):
                            v = v.item()
                        final_info[k].append(v)
                if it % self.print_every == 0:
                    logger.info(f"Generating objs {it - num_training_steps}/{num_final_gen_steps}")
            final_info = {k: np.mean(v) for k, v in final_info.items()}

            logger.info("Final generation steps completed - " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))
            self.log(final_info, num_training_steps, "final")

        # for pypy and other GC having implementations, we need to manually clean up
        del train_dl
        del valid_dl
        if self.cfg.num_final_gen_steps:
            del final_dl

    def terminate(self):
        logger = logging.getLogger("logger")
        for handler in logger.handlers:
            handler.close()

        for hook in self.sampling_hooks:
            if hasattr(hook, "terminate") and hook.terminate not in self.to_terminate:
                hook.terminate()

        for terminate in self.to_terminate:
            terminate()

    def _get_state(self, it):
        state = {
            "models_state_dict": self.model.state_dict(),
            "cfg": self.cfg,
            "step": it,
        }
        if self.sampling_model is not self.model:
            state["sampling_model_state_dict"] = self.sampling_model.state_dict()
        return state

    def _save_state(self, it):
        fn = pathlib.Path(self.cfg.log_dir) / "model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(
                self._get_state(it),
                fd,
            )
        if self.cfg.store_all_checkpoints:
            shutil.copy(fn, pathlib.Path(self.cfg.log_dir) / f"model_state_{it}.pt")

    def log(self, info, index, stage):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{stage}_{k}", v, index)
        if wandb.run is not None:
            wandb.log({f"{stage}/{k}": v for k, v in info.items()}, step=index)

    def __del__(self):
        self.terminate()


def cycle(it):
    while True:
        yield from it
