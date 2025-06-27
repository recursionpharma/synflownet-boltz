import argparse
import datetime
import socket
import threading
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import wandb
from loguru import logger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

import synflownet
from synflownet import GFNTask, LogScalar, ObjectProperties
from synflownet.algo.reaction_sampling import SynthesisSampler
from synflownet.config import Config
from synflownet.data.scripts.precompute_bb_masks import get_precomputed_rxn_bb_masks
from synflownet.envs.synthesis_building_env import (
    ReactionTemplateEnv,
    ReactionTemplateEnvContext,
)
from synflownet.online_trainer import StandardOnlineTrainer
from synflownet.utils.conditioning import TemperatureConditional
from synflownet.utils.transforms import to_logreward


class AsyncRewardTask(GFNTask):
    def __init__(
        self,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.cfg = cfg
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        if final:
            cfg = self.cfg
            cfg.cond.temperature.sample_dist = "constant"
            cfg.cond.temperature.dist_params = [max(cfg.cond.temperature.dist_params)]
            self.temperature_conditional = TemperatureConditional(cfg)
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_obj_properties(self, mols: list[RDMol], traj_lens: Tensor, **kwargs) -> tuple[ObjectProperties, Tensor]:
        raise NotImplementedError("Reward computation is handled by the reward workers")

    def setup_sampling_space(self, cfg: Config):
        synflownet_path = Path(synflownet.__file__).parent
        bb_path = synflownet_path / "data" / "building_blocks"
        templates_path = synflownet_path / "data" / "templates"

        if cfg.task.denovo_task.sampling_space == "hartenfeller":
            cfg.task.denovo_task.building_blocks_path = bb_path / "enamine_bbs_hartenfeller_matching_sanitized.txt"
            cfg.task.denovo_task.templates_path = templates_path / "hartenfeller.txt"
        elif cfg.task.denovo_task.sampling_space == "enamine":
            cfg.task.denovo_task.building_blocks_path = bb_path / "enamine_bbs_real_matching.txt"
            cfg.task.denovo_task.templates_path = templates_path / "real.txt"
        elif cfg.task.denovo_task.sampling_space == "enamine_conditionMasked":
            cfg.task.denovo_task.building_blocks_path = bb_path / "enamine_bbs_real_matching_conditionMasked.txt"
            cfg.task.denovo_task.templates_path = templates_path / "real.txt"
        else:
            raise ValueError(f"Invalid sampling space: {cfg.task.denovo_task.sampling_space}")

        return cfg


class AsyncRewardTrainer(StandardOnlineTrainer):
    task: AsyncRewardTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.verbose = False

        # Accelerators
        cfg.pickle_mp_messages = False
        cfg.num_workers = 0
        # If True, a fwd action is allowed only if in reverse it produces
        # the exact same reaction (identical reactants and products)
        cfg.algo.strict_forward_policy = False
        # If True, bimolecular bck actions are masked if they don't produce at least one bb
        cfg.algo.strict_bck_masking = False
        cfg.algo.tb.do_correct_idempotent = False
        cfg.mp_buffer_size = 64 * 4096**2

        # Model and Optimizer
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 2_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4
        cfg.model.graph_transformer.continuous_action_embs = True
        cfg.model.graph_transformer.fingerprint_type = "morgan_1024"
        cfg.task.denovo_task.sampling_space = "hartenfeller"

        # Training
        cfg.num_training_steps = 100_000
        cfg.print_every = 100

        # Conditioning
        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [32.0]

        # Algorithm
        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.99
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.05
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = False
        cfg.algo.tb.backward_policy = "MaxLikelihood"
        cfg.algo.num_from_buffer_for_pb = 0  # not used for Uniform or MaxLikelihood backward policies

        # MDP design
        cfg.data_root = Path(synflownet.__file__).parent
        cfg.task.denovo_task.building_blocks_path = cfg.data_root / "data" / "building_blocks" / "enamine_bb_270k_HB_matching_sanitized.txt"
        cfg.task.denovo_task.templates_path = cfg.data_root / "data" / "templates" / "hartenfeller.txt"
        cfg.algo.max_len = 3

        # Because the reward computation is done externally
        # we don't have any special treatment for validation samples
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 0
        cfg.num_validation_gen_steps = 0
        cfg.validate_every = None
        cfg.num_final_gen_steps = 0

        # This task assumes the use of an expensive external reward function
        # The reward here is not computed on Synflownet's side, but rather by
        # asynchronous reward workers which run in separate processes and
        # interact with Synflownet via a RewardQueue and a PersistentReplayBuffer.
        cfg.replay.use = True
        cfg.replay.capacity = 0
        cfg.replay.buffer_is_async = True
        cfg.replay.warmup = 200
        cfg.replay.num_from_replay = 64
        cfg.replay.num_new_samples = 64
        cfg.replay.persistent_replay_last_to_random_ratio = 0.5
        cfg.replay.halt_training_when_queue_is_full = False  # Default to FIFO behavior
        cfg.algo.num_from_policy = 0  # so the model doesn't train using on-policy samples

    def setup_task(self):
        self.task = AsyncRewardTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        # Setup sampling space config arguments
        self.cfg = self.task.setup_sampling_space(self.cfg)

        # Load building blocks
        with open(Path(self.cfg.task.denovo_task.building_blocks_path)) as file:
            building_blocks = file.read().splitlines()

        # Load templates
        with open(Path(self.cfg.task.denovo_task.templates_path)) as file:
            reaction_templates = file.read().splitlines()

        # Compute building block masks (if not already precomputed)
        precomputed_bb_masks = get_precomputed_rxn_bb_masks(
            bb_path=Path(self.cfg.task.denovo_task.building_blocks_path),
            template_path=Path(self.cfg.task.denovo_task.templates_path),
        )

        # Instantiate ctx and env
        self.ctx = ReactionTemplateEnvContext(
            num_cond_dim=self.task.num_cond_dim,
            building_blocks=building_blocks,
            reaction_templates=reaction_templates,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
            strict_bck_masking=self.cfg.algo.strict_bck_masking,
            device=self.device,
            add_hs=Path(self.cfg.task.denovo_task.templates_path).stem == "real",
            building_blocks_costs=None,
        )
        self.env = ReactionTemplateEnv(ctx=self.ctx)

    def setup_sampler(self):
        self.sampler = SynthesisSampler(
            cfg=self.cfg,
            ctx=self.ctx,
            env=self.env,
            max_len=self.cfg.algo.max_len,
            correct_idempotent=self.cfg.algo.tb.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.algo.tb.do_parameterize_p_b,
        )

    def build_callbacks(self):
        from rdkit.Chem.Scaffolds import MurckoScaffold

        graph_to_obj = self.ctx.graph_to_obj
        reward_fn = lambda rdmols, traj_lens: self.task.compute_obj_properties(rdmols, traj_lens)[0]  # noqa: E731

        class UniqueMurckoScaffoldsCallback:
            def __init__(self, reward_thresh):
                self._reward_thresh = reward_thresh

            def on_validation_end(self, step_outputs):
                mols = []
                rewards = []
                for out in step_outputs:
                    batch = out["batch"]
                    final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
                    final_graphs = [batch.nx_graphs[i] for i in final_graph_idx]
                    batch_mols = [graph_to_obj(g) for g in final_graphs]
                    mols.extend(batch_mols)
                    rewards.append(reward_fn(batch_mols, batch.traj_lens))

                murcko_scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
                rewards = torch.cat(rewards)
                assert len(murcko_scaffolds) == len(rewards)

                scaffolds_above_thresh = [smi for smi, r in zip(murcko_scaffolds, rewards) if r > self._reward_thresh]
                unique_scaffolds = set(scaffolds_above_thresh)

                return {f"unique_murcko_r_gt_{self._reward_thresh}": len(unique_scaffolds)}

        return {"murcko_scaffolds": UniqueMurckoScaffoldsCallback(reward_thresh=0.5)}


def main(resume_training: Optional[str] = None):
    if resume_training is not None:
        assert Path(resume_training).exists(), "Path to the experiment log_dir to resume does not exist."
        checkpoint_path = Path(resume_training) / "model_state.pt"
        assert checkpoint_path.exists(), "Checkpoint file not found."
        logger.add(resume_training + "/logger.txt")
        logger.info(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        trainer = AsyncRewardTrainer.load_from_checkpoint(checkpoint_path)

        name = resume_training.split("/")[-1]
        wandb.init(project="synflownet", name=name, id=name, resume="must")

    else:
        logger.info("Creating a new config and starting a new run")

        # Initialise config -------
        cfg = Config.empty()
        cfg.log_dir = f"./logs/debug_run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.print_every = 1
        cfg.task.denovo_task.sampling_space == "hartenfeller"
        # ... asynchronous replay buffer
        DBS_PATH = Path("./tmp_dbs")
        cfg.replay.reward_queue_max_size = 10_000
        cfg.replay.persistent_replay_max_size = 100_000
        cfg.replay.reward_queue_path = DBS_PATH / "reward_queue.db"
        cfg.replay.persistent_replay_path = DBS_PATH / "persistent_replay.db"

        cfg.mp_buffer_size = 64 * 4096**2
        cfg.num_workers = 0
        cfg.replay.persistent_replay_last_to_random_ratio = 0.5

        REWARD_IN_SAME_PROCESS = True
        if REWARD_IN_SAME_PROCESS:
            from synflownet.tasks.script_simulate_reward_worker import (
                main as reward_worker,
            )

            threading.Thread(target=reward_worker, args=(DBS_PATH,), daemon=True).start()

        # -------

        logger.remove()  # Remove default logger
        logger.add(lambda msg: print(msg, end=""), level="INFO")
        logger.add(cfg.log_dir + "/logger.txt", level="INFO")
        trainer = AsyncRewardTrainer(cfg)
        name = cfg.log_dir.split("/")[-1]
        wandb.init(project="sfn-boltz", name=name, id=name)

    trainer.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--resume_training",
        type=str,
        required=False,
        default=None,
        help="Path to the experiment log_dir to resume",
    )
    args = p.parse_args()
    main(resume_training=args.resume_training)
