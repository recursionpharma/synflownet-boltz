from dataclasses import dataclass, field
from typing import Optional

from synflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False


@dataclass
class ReactionTaskConfig(StrictDataClass):
    templates_filename: str = "templates.txt"
    building_blocks_filename: str = "pharmaron_bbs.txt"
    precomputed_bb_masks_filename: str = "precomputed_bb_masks_pharmaron_bbs.pkl" 
    reverse_templates_filename: Optional[str] = None
    reward: Optional[str] = None
    building_blocks_costs: Optional[str] = None
    sanitize_building_blocks: bool = False


@dataclass
class DenovoTaskConfig(StrictDataClass):
    sampling_space: str = "hartenfeller"
    building_blocks_path: str = "./building_blocks.txt"
    templates_path: str = "./templates.txt"


@dataclass
class TasksConfig(StrictDataClass):
    reactions_task: ReactionTaskConfig = field(default_factory=ReactionTaskConfig)
    denovo_task: DenovoTaskConfig = field(default_factory=DenovoTaskConfig)
