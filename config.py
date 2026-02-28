from typing import Optional
from dataclasses import dataclass
from simple_parsing import field

@dataclass
class InitParams:
    initial_gaussians: Optional[int] = None
    coarse_res: int = 10

@dataclass
class SimulationParams:
    map_size: float = 20.0
    grid_res: int = 20
    num_beams: int = 30
    num_blobs: int = 5
    no_gauss_filter: bool = field(default=False, action="store_true")

@dataclass
class TrainParams:
    pos_lr: float = 0.01
    scale_lr: float = 0.005
    rotation_lr: float = 0.001
    concentration_lr: float = 0.01
    lr_decay: float = 0.5
    lr_decay_step: int = 200
    iterations: int = 3000
    target_loss: float = 1e-5
    l1_reg: float = 0.1
    no_live_vis: bool = field(default=False, action="store_true")

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.0005
    scale_threshold: float = 2.5
    prune_threshold: float = 0.005
    densify_from: int = 200
    densify_until: int = 1500
    densify_interval: int = 100

@dataclass
class Config:
    init: InitParams
    sim: SimulationParams
    train: TrainParams
    densify: DensificationParams
