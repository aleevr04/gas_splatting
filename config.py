import torch

from typing import Optional, Tuple
from dataclasses import dataclass
from simple_parsing import field

@dataclass
class InitParams:
    initial_gaussians: Optional[int] = None
    coarse_proportion: float = 0.1

@dataclass
class SimulationParams:
    seed: Optional[int] = None # Random seed for GT generation
    gt_file: Optional[str] = None # csv file containing gas distribution data

    map_size: Tuple[float, float] = (20.0, 20.0) # (map_width, map_height). Ignored when a csv file is provided
    cell_size: float = 1.0

    num_beams: int = 30
    num_blobs: int = 5
    no_gauss_filter: bool = field(default=False, action="store_true")
    
    noise: bool = field(default=False, action="store_true") # Add noise to simulated measurements
    snr_db: int = 30 # Signal-to-noise ratio (dB)

@dataclass
class TrainParams:
    pos_lr: float = 0.008
    scale_lr: float = 0.003
    rotation_lr: float = 0.001
    concentration_lr: float = 0.005

    iterations: int = 1500
    target_loss: float = 1e-5

    do_eval: bool = field(default=False, action="store_true")
    eval_interval: int = 25

    live_vis: bool = field(default=False, action="store_true")

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.005
    scale_threshold: float = 0.05
    prune_threshold: float = 0.005
    densify_from: int = 100
    densify_until: int = 750
    densify_interval: int = 50
    long_axis_split: bool = field(default=False, action="store_true")

@dataclass
class Config:
    init: InitParams
    sim: SimulationParams
    train: TrainParams
    densify: DensificationParams

    # "cuda" if available, "cpu" otherwise. Can be overwritten
    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> torch.device:
        if self.device_type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        return torch.device(self.device_type)
