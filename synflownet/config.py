import os
from dataclasses import dataclass, field, is_dataclass
from typing import Optional

from omegaconf import MISSING

from synflownet.algo.config import AlgoConfig
from synflownet.data.config import ReplayConfig
from synflownet.models.config import ModelConfig
from synflownet.tasks.config import TasksConfig
from synflownet.utils.config import ConditionalsConfig
from synflownet.utils.misc import StrictDataClass


@dataclass
class OptimizerConfig(StrictDataClass):
    """Generic configuration for optimizers

    Attributes
    ----------
    opt : str
        The optimizer to use (either "adam" or "sgd")
    learning_rate : float
        The learning rate
    lr_decay : float
        The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))
    weight_decay : float
        The L2 weight decay
    momentum : float
        The momentum parameter value
    clip_grad_type : str
        The type of gradient clipping to use (either "norm" or "value")
    clip_grad_param : float
        The parameter for gradient clipping
    adam_eps : float
        The epsilon parameter for Adam
    """

    opt: str = "adam"
    learning_rate: float = 1e-4
    lr_decay: float = 20_000
    weight_decay: float = 1e-8
    momentum: float = 0.9
    clip_grad_type: str = "norm"
    clip_grad_param: float = 10.0
    adam_eps: float = 1e-8


@dataclass
class WandBConfig(StrictDataClass):
    """Configuration for Weights & Biases

    Attributes
    ----------
    project : str
        The project name
    entity : str
        The entity name
    tags : List[str]
        The tags to use
    """

    project: str = "synflownet"
    entity: Optional[str] = None
    tags: list = field(default_factory=lambda: ["synflownet"])


@dataclass
class Config(StrictDataClass):
    """Base configuration for training

    Attributes
    ----------
    desc : str
        A description of the experiment
    verbose : bool
        Whether to print debug information
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    device : str
        The device to use for training (either "cpu" or "cuda[:<device_id>]")
    seed : int
        The random seed
    validate_every : int
        The number of training steps after which to validate the model
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    store_all_checkpoints : bool
        Whether to store all checkpoints or only the last one
    print_every : int
        The number of training steps after which to print the training loss
    start_at_step : int
        The training step to start at (default: 0)
    num_final_gen_steps : Optional[int]
        After training, the number of steps to generate graphs for
    num_training_steps : int
        The number of training steps
    num_workers : int
        The number of workers to use for creating minibatches (0 = no multiprocessing)
    hostname : Optional[str]
        The hostname of the machine on which the experiment is run
    pickle_mp_messages : bool
        Whether to pickle messages sent between processes (only relevant if num_workers > 0)
    git_hash : Optional[str]
        The git hash of the current commit
    """

    desc: str = "noDesc"
    verbose: bool = True
    log_dir: str = MISSING
    resume: Optional[str] = None
    device: str = "cuda"
    data_root: str = "./data"
    repo_root: str = os.path.dirname(__file__)
    seed: int = 0
    validate_every: Optional[int] = 1000
    checkpoint_every: Optional[int] = None
    store_all_checkpoints: bool = False
    print_every: int = 100
    start_at_step: int = 0
    num_final_gen_steps: Optional[int] = None
    num_validation_gen_steps: Optional[int] = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    hostname: Optional[str] = None
    pickle_mp_messages: bool = False
    mp_buffer_size: int = 32 * 4096**2
    git_hash: Optional[str] = None
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    task: TasksConfig = field(default_factory=TasksConfig)
    cond: ConditionalsConfig = field(default_factory=ConditionalsConfig)
    reward: str = "seh_reaction"
    wandb: WandBConfig = field(default_factory=WandBConfig)


def override_config(cfg, overrides):
    """
    Override a dataclass instance with a dictionary of overrides,
    including nested dataclasses.

    This is meant to be used on the user side (tasks) to provide
    some configuration using the Config class while overwritting
    only the fields that have been set by the user.
    """
    for key, value in overrides.items():
        if is_dataclass(getattr(cfg, key)):
            override_config(getattr(cfg, key), value)
        else:
            setattr(cfg, key, value)

    return cfg
