import pickle
import time
from collections.abc import Generator
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import IterableDataset

from synflownet import GFNAlgorithm, GFNTask
from synflownet.config import Config
from synflownet.data.async_sql_databases import PersistentReplayBuffer, RewardQueue
from synflownet.data.replay_buffer import ReplayBuffer, detach_and_cpu
from synflownet.envs.graph_building_env import GraphBuildingEnvContext
from synflownet.utils.misc import get_worker_rng
from synflownet.utils.multiprocessing_proxy import BufferPickler, SharedPinnedBuffer


def cycle_call(it):
    while True:
        yield from it()


class DataSource(IterableDataset):
    def __init__(
        self,
        cfg: Config,
        ctx: GraphBuildingEnvContext,
        algo: GFNAlgorithm,
        task: GFNTask,
        replay_buffer: Optional[ReplayBuffer] = None,
        is_algo_eval: bool = False,
        start_at_step: int = 0,
    ):
        """A DataSource mixes multiple iterators into one. These are created with do_* methods."""
        self.iterators: list[Generator] = []
        self.cfg = cfg
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.replay_buffer = replay_buffer
        self.is_algo_eval = is_algo_eval
        self.sampling_hooks: list[Callable] = []
        self.active = True

        self.global_step_count = torch.zeros(1, dtype=torch.int64) + start_at_step
        self.global_step_count.share_memory_()
        self.global_step_count_lock = torch.multiprocessing.Lock()
        self.current_iter = start_at_step
        self.setup_mp_buffers()

        if self.cfg.replay.buffer_is_async:
            assert self.cfg.replay.reward_queue_path is not None
            assert self.cfg.replay.persistent_replay_path is not None
            assert self.cfg.replay.hindsight_ratio == 0, "Rewards cannot be re-computed in hindsight with an async replay buffer"

            self.reward_queue = RewardQueue(
                db_path=self.cfg.replay.reward_queue_path,
                max_size=self.cfg.replay.reward_queue_max_size,
            )
            self.persistent_replay_buffer = PersistentReplayBuffer(
                db_path=self.cfg.replay.persistent_replay_path,
                max_size=self.cfg.replay.persistent_replay_max_size,
            )

    def add_sampling_hook(self, hook: Callable):
        """Add a hook that is called when sampling new trajectories.

        The hook should take a list of trajectories as input.
        The hook will not be called on trajectories that are sampled from the replay buffer or dataset.
        """
        self.sampling_hooks.append(hook)
        return self

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        self.rng = get_worker_rng()
        its = [i() for i in self.iterators]
        while True:
            if hasattr(self.algo, "set_is_eval"):
                self.algo.set_is_eval(self.is_algo_eval)
            with self.global_step_count_lock:
                self.current_iter = self.global_step_count.item()
                self.global_step_count += 1
            iterator_outputs = [next(i, None) for i in its]
            if any(i is None for i in iterator_outputs):
                if not all(i is None for i in iterator_outputs):
                    logger.warning("Some iterators are done, but not all. You may be mixing incompatible iterators.")
                    iterator_outputs = [i for i in iterator_outputs if i is not None]
                else:
                    break
            traj_lists, batch_infos = zip(*iterator_outputs)
            trajs = sum(traj_lists, [])
            # Merge all the dicts into one
            batch_info = {}
            for d in batch_infos:
                batch_info.update(d)
            yield self.create_batch(trajs, batch_info)

    def do_sample_model(self, model, num_from_policy, num_new_replay_samples=None):
        assert not self.cfg.replay.buffer_is_async, "Async Replay training is not compatible with pure online training"
        if num_new_replay_samples is not None:
            assert self.replay_buffer is not None, "num_new_replay_samples specified without a replay buffer"
        if num_new_replay_samples is None:
            assert self.replay_buffer is None, "num_new_replay_samples not specified with a replay buffer"

        num_new_replay_samples = num_new_replay_samples or 0
        num_samples = max(num_from_policy, num_new_replay_samples)

        def iterator():
            while self.active:
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                # take note whether the trajectories come from own samples or from graphs
                trajs = self.algo.create_training_data_from_own_samples(model, num_samples, cond_info["encoding"], p)
                for i in range(len(trajs)):
                    trajs[i]["from_p_b"] = torch.tensor([0.0])
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                if self.ctx.bbs_costs:
                    self.compute_bb_costs(trajs)
                self.compute_bck_rewards(trajs)
                self.send_to_replay(trajs[:num_new_replay_samples])  # This is a no-op if there is no replay buffer
                batch_info = self.call_sampling_hooks(trajs)
                yield (trajs[:num_from_policy], batch_info)

        self.iterators.append(iterator)
        return self

    def do_sample_async_replay(self, model, num_new_replay_samples: int, num_from_replay: int):
        """Sample from model and push to reward queue, then sample from persistent replay buffer.

        Args:
            model: The model to sample from
            num_new_replay_samples (int): Number of new samples to generate
            num_from_replay (int): Number of samples to retrieve from replay
        """
        assert self.reward_queue is not None, "Reward queue not initialized"
        assert self.persistent_replay_buffer is not None, "Persistent replay buffer not initialized"

        def iterator():
            while self.active:
                # Check if we should halt training when queue is full
                if self.cfg.replay.halt_training_when_queue_is_full:
                    # Check queue size before generating more
                    queue_size = self.reward_queue.get_db_size()
                    if queue_size >= self.reward_queue.max_size:
                        logger.info(f"RewardQueue full ({queue_size} pending), halting training and waiting on reward computation...")
                        time.sleep(5)
                        continue  # Skip this iteration entirely - don't yield any batch

                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_new_replay_samples, t)

                # Generate new trajectories
                trajs = self.algo.create_training_data_from_own_samples(model, num_new_replay_samples, cond_info["encoding"], p)

                for i in range(len(trajs)):
                    trajs[i]["from_p_b"] = torch.tensor([0.0])
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs

                # Serialize the trajectories for insertion into the database
                smiles = [self.ctx.object_to_log_repr(traj["result"]) for traj in trajs]
                trajs = [pickle.dumps(traj) for traj in trajs]

                # Push new trajectories to reward queue
                try:
                    self.reward_queue.push(smiles, trajs, allow_fifo=not self.cfg.replay.halt_training_when_queue_is_full)
                except ValueError as e:
                    logger.warning(f"Failed to push to reward queue: {e}")
                    if self.cfg.replay.halt_training_when_queue_is_full:
                        continue  # Skip this iteration if push failed and we're in halt mode

                # Sample rewarded trajectories from persistent buffer
                try:
                    num_samples_as_randomly_sampled = int(num_from_replay * self.cfg.replay.persistent_replay_last_to_random_ratio)
                    num_samples_as_last_added = num_from_replay - num_samples_as_randomly_sampled

                    # Sample randomly from the persistent buffer
                    entries = self.persistent_replay_buffer.sample(num_samples_as_randomly_sampled)
                    if len(entries) == 0:
                        raise ValueError("No samples found in persistent replay buffer.")
                    idxs, smiles, trajs, rewards, timestamps, infos = entries

                    # Get the last n entries from the persistent buffer
                    entries = self.persistent_replay_buffer.get_last_n_entries(num_samples_as_last_added)
                    if len(entries) == 0:
                        raise ValueError("No samples found in persistent replay buffer.")
                    (
                        idxs_last,
                        smiles_last,
                        trajs_last,
                        rewards_last,
                        timestamps_last,
                        infos_last,
                    ) = entries

                    # Merge the two lists of trajectories
                    smiles = smiles + smiles_last
                    trajs = trajs + trajs_last
                    rewards = rewards + rewards_last
                    infos = infos + infos_last

                    # Deserialize trajectories
                    trajs = [pickle.loads(traj) for traj in trajs]
                    rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
                    assert rewards.shape == (len(trajs), 1)

                    # Compute object properties as in self.compute_properties but skip the reward computation
                    valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
                    for i in range(len(trajs)):
                        trajs[i]["obj_props"] = rewards[i]
                        trajs[i]["is_online"] = True
                    for i in valid_idcs:
                        trajs[i]["is_valid"] = True

                    # Finish adjusting the traj properties and reward
                    self.compute_log_rewards(trajs)
                    if self.ctx.bbs_costs:
                        self.compute_bb_costs(trajs)
                    self.compute_bck_rewards(trajs)
                    batch_info = self.call_sampling_hooks(trajs)
                    yield (trajs, batch_info)

                except ValueError as e:
                    logger.warning(f"Failed to obtain a labeled batch: {e}")
                    continue  # Skip this iteration if we can't get samples from persistent buffer

        self.iterators.append(iterator)
        return self

    def do_sample_model_n_times(self, model, num_samples_per_batch, num_total):
        total = torch.zeros(1, dtype=torch.int64)
        total.share_memory_()
        total_lock = torch.multiprocessing.Lock()
        total_barrier = torch.multiprocessing.Barrier(max(1, self.cfg.num_workers))

        def iterator():
            while self.active:
                with total_lock:
                    n_so_far = total.item()
                    n_this_time = min(num_total - n_so_far, num_samples_per_batch)
                    total[:] += n_this_time
                    if n_this_time == 0:
                        break
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(n_this_time, t)
                trajs = self.algo.create_training_data_from_own_samples(model, n_this_time, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                if self.ctx.bbs_costs:
                    self.compute_bb_costs(trajs)
                self.compute_bck_rewards(trajs)
                batch_info = self.call_sampling_hooks(trajs)
                yield trajs, batch_info
            total_barrier.wait()  # Wait for all workers to finish before resetting the counter
            total[:] = 0

        self.iterators.append(iterator)
        return self

    def do_sample_replay(self, num_samples):
        def iterator():
            while self.active:
                trajs, *_ = self.replay_buffer.sample(num_samples)
                self.relabel_in_hindsight(trajs)  # This is a no-op if the hindsight ratio is 0
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def do_dataset_in_order(self, data, num_samples, backwards_model):
        def iterator():
            for idcs in self.iterate_indices(len(data), num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                objs, props = map(list, zip(*[data[i] for i in idcs])) if len(idcs) else ([], [])
                trajs = self.algo.create_training_data_from_graphs(objs, backwards_model, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.set_traj_props(trajs, props)
                self.compute_log_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def do_conditionals_dataset_in_order(self, data, num_samples, model):
        def iterator():
            for idcs in self.iterate_indices(len(data), num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.encode_conditional_information(torch.stack([data[i] for i in idcs]))
                trajs = self.algo.create_training_data_from_own_samples(model, len(idcs), cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs)  # This is a no-op if there is no replay buffer
                # If we're using a dataset of preferences, the user/hooks may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
                batch_info = self.call_sampling_hooks(trajs)
                yield trajs, batch_info

        self.iterators.append(iterator)
        return self

    def do_sample_dataset(self, data, num_samples, backwards_model):
        def iterator():
            while self.active:
                idcs = self.sample_idcs(len(data), num_samples)
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                objs, props = map(list, zip(*[data[i] for i in idcs])) if len(idcs) else ([], [])
                trajs = self.algo.create_training_data_from_graphs(objs, backwards_model, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.set_traj_props(trajs, props)
                self.compute_log_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def do_sample_backward(self, num_samples, backwards_model):
        def iterator():
            while self.active:
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                # take note whether the trajectories come from own samples or from graphs
                data, *_ = self.replay_buffer.sample(num_samples)
                # only take valid graphs
                valid_idcs = torch.tensor([i for i in range(len(data)) if data[i].get("is_valid", True)]).long()
                graphs = [data[i]["result"] for i in valid_idcs]
                cond_info = self.task.sample_conditional_information(len(graphs), t)
                trajs = self.algo.create_training_data_from_graphs(graphs, backwards_model, cond_info["encoding"], p)
                for i in range(len(trajs)):
                    trajs[i]["from_p_b"] = torch.tensor([1.0])
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                if self.ctx.bbs_costs:
                    self.compute_bb_costs(trajs)
                self.compute_bck_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def call_sampling_hooks(self, trajs):
        batch_info = {}
        obj_props = torch.stack([t["obj_props"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        rewards = torch.exp(log_rewards / (cond_info.get("beta", 1)))
        for hook in self.sampling_hooks:
            batch_info.update(hook(trajs, rewards, obj_props, cond_info))
        return batch_info

    def create_batch(self, trajs, batch_info):
        trajs = detach_and_cpu(trajs)
        ci = torch.stack([t["cond_info"]["encoding"] for t in trajs])
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        batch = self.algo.construct_batch(trajs, ci, log_rewards)
        batch.num_online = sum(t.get("is_online", 0) for t in trajs)
        batch.num_offline = len(trajs) - batch.num_online
        batch.extra_info = batch_info
        if "preferences" in trajs[0]["cond_info"].keys():
            batch.preferences = torch.stack([t["cond_info"]["preferences"] for t in trajs])
        if "focus_dir" in trajs[0]["cond_info"].keys():
            batch.focus_dir = torch.stack([t["cond_info"]["focus_dir"] for t in trajs])

        if self.ctx.has_n() and self.cfg.algo.tb.do_predict_n:
            log_ns = [self.ctx.traj_log_n(i["traj"]) for i in trajs]
            batch.log_n = torch.tensor([i[-1] for i in log_ns], dtype=torch.float32)
            batch.log_ns = torch.tensor(sum(log_ns, start=[]), dtype=torch.float32)
        batch.obj_props = torch.stack([t["obj_props"] for t in trajs])
        return self._maybe_put_in_mp_buffer(batch)

    def compute_properties(self, trajs, mark_as_online=False):
        """Sets trajs' obj_props and is_valid keys by querying the task."""
        valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
        objs = [self.ctx.graph_to_obj(trajs[i]["result"]) for i in valid_idcs]
        traj_lens = torch.tensor([len(trajs[i]["traj"]) for i in valid_idcs]).long()
        obj_props, m_is_valid = self.task.compute_obj_properties(objs, traj_lens=traj_lens)
        assert obj_props.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
        # The task may decide some of the objs are invalid, we have to again filter those
        valid_idcs = valid_idcs[m_is_valid]
        all_fr = torch.zeros((len(trajs), obj_props.shape[1]))
        all_fr[valid_idcs] = obj_props
        for i in range(len(trajs)):
            trajs[i]["obj_props"] = all_fr[i]
            trajs[i]["is_online"] = mark_as_online
        # Override the is_valid key in case the task made some objs invalid
        for i in valid_idcs:
            trajs[i]["is_valid"] = True

    def compute_log_rewards(self, trajs):
        """Sets trajs' log_reward key by querying the task."""
        obj_props = torch.stack([t["obj_props"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        log_rewards = self.task.cond_info_to_logreward(cond_info, obj_props)
        min_r = torch.as_tensor(self.cfg.algo.illegal_action_logreward).float()
        for i in range(len(trajs)):
            trajs[i]["log_reward"] = log_rewards[i] if trajs[i].get("is_valid", True) else min_r

    def compute_bb_costs(self, trajs):
        assert not self.cfg.algo.synthesis_cost_as_bck_reward or (
            self.ctx.bbs_costs and isinstance(self.ctx.bbs_costs, dict)
        ), "`cfg.algo.synthesis_cost_as_bck_reward` requires `bbs_costs` to be a non-empty dictionary."
        for i in range(len(trajs)):
            bb_costs = torch.tensor([self.ctx.bbs_costs.get(bb, 0.0) for bb in trajs[i]["bbs"]])
            trajs[i]["bbs_cost"] = torch.sum(bb_costs)

    def compute_bck_rewards(self, trajs):
        """Sets trajs' bck_reward key by querying the task."""
        # all states in the traj get reward 0, except the last one, which get reward=1 if it is valid and -1 otherwise
        min_r = torch.as_tensor(self.cfg.algo.illegal_bck_traj_reward).float()
        for i in range(len(trajs)):
            # synthetic_cost = torch.tensor(len(trajs[i]["traj"])).float() + torch.sum(trajs[i]["bbs_costs"])
            if self.cfg.algo.synthesis_cost_as_bck_reward:
                trajs[i]["bck_reward"] = (
                    (1 - trajs[i]["bbs_cost"]) ** self.cfg.algo.bck_reward_exponent if trajs[i].get("ends_in_s_0", True) else min_r
                )
            else:
                trajs[i]["bck_reward"] = torch.ones(1) if trajs[i].get("ends_in_s_0", True) else min_r  # just reward making it back to s_0

    def send_to_replay(self, trajs):
        if self.replay_buffer is not None:
            for t in trajs:
                self.replay_buffer.push(
                    t,
                    t["log_reward"],
                    t["obj_props"],
                    t["cond_info"],
                    t["is_valid"],
                    unique_obj=self.ctx.get_unique_obj(t["result"]),
                    priority=t.get("priority", t["log_reward"].item()),
                )

    def set_traj_cond_info(self, trajs, cond_info):
        for i in range(len(trajs)):
            trajs[i]["cond_info"] = {k: cond_info[k][i] for k in cond_info}

    def set_traj_props(self, trajs, props):
        for i in range(len(trajs)):
            trajs[i]["obj_props"] = props[i]

    def relabel_in_hindsight(self, trajs):
        if self.cfg.replay.hindsight_ratio == 0:
            return
        assert hasattr(
            self.task, "relabel_condinfo_and_logrewards"
        ), "Hindsight requires the task to implement relabel_condinfo_and_logrewards"
        # samples indexes of trajectories without repeats
        hindsight_idxs = torch.randperm(len(trajs))[: int(len(trajs) * self.cfg.replay.hindsight_ratio)]
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        obj_props = torch.stack([t["obj_props"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        cond_info, log_rewards = self.task.relabel_condinfo_and_logrewards(cond_info, log_rewards, obj_props, hindsight_idxs)
        self.set_traj_cond_info(trajs, cond_info)
        for i in range(len(trajs)):
            trajs[i]["log_reward"] = log_rewards[i]

    def sample_idcs(self, n, num_samples):
        return self.rng.choice(n, num_samples, replace=False)

    def iterate_indices(self, n, num_samples):
        worker_info = torch.utils.data.get_worker_info()
        if n == 0:
            # Should we be raising an error here? warning?
            yield np.arange(0, 0)
            return

        if worker_info is None:  # no multi-processing
            start, end, wid = 0, n, -1
        else:  # split the data into chunks (per-worker)
            nw = worker_info.num_workers
            wid = worker_info.id
            start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))

        if end - start <= num_samples:
            yield np.arange(start, end)
            return
        for i in range(start, end - num_samples, num_samples):
            yield np.arange(i, i + num_samples)
        if i + num_samples < end:
            yield np.arange(i + num_samples, end)

    def setup_mp_buffers(self):
        if self.cfg.num_workers > 0:
            self.mp_buffer_size = self.cfg.mp_buffer_size
            if self.mp_buffer_size:
                self.result_buffer = [SharedPinnedBuffer(self.mp_buffer_size) for _ in range(self.cfg.num_workers)]
        else:
            self.mp_buffer_size = None

    def _maybe_put_in_mp_buffer(self, batch):
        if self.mp_buffer_size:
            return (
                BufferPickler(self.result_buffer[self._wid]).dumps(batch),
                self._wid,
            )
        else:
            return batch
