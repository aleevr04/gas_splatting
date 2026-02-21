from dataclasses import dataclass

@dataclass
class InitParams:
    initial_gaussians: int = 5
    coarse_res: int = 10

@dataclass
class SimulationParams:
    map_size: float = 20.0
    grid_res: int = 20
    num_beams: int = 30
    num_blobs: int = 5
    gauss_filter: bool = True

@dataclass
class TrainParams:
    pos_lr: float = 0.01
    scale_lr: float = 0.005
    rotation_lr: float = 0.001
    concentration_lr: float = 0.01
    lr_decay: float = 0.5
    lr_decay_step: int = 200
    iterations: int = 1500
    target_loss: float = 1e-5

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.0005
    prune_threshold: float = 0.05
    densify_from: int = 200
    densify_until: int = 800
    densify_interval: int = 200