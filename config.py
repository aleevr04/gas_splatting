import torch

from typing import Optional, Tuple
from dataclasses import dataclass
from simple_parsing import field

@dataclass
class EnvironmentParams:
    map_size: Tuple[float, float] = (20.0, 20.0) # Map size (map_width, map_height) in meters. Ignored when a csv file is provided
    cell_size: float = 1.0 # Cell size in meters

@dataclass
class SimulationParams:
    seed: Optional[int] = None # Random seed for GT generation
    gt_file: Optional[str] = None # csv file containing gas distribution data 
    num_beams: int = 50 # Total number of TDLAS beams
    
    noise: bool = field(default=False, action="store_true") # Add noise to simulated measurements
    snr_db: int = 30 # Signal-to-noise ratio (dB)

@dataclass
class TrainParams:
    pos_lr: float = 0.008 # Position learning rate
    scale_lr: float = 0.003 # Scale learning rate
    rotation_lr: float = 0.001 # Rotation learning rate
    concentration_lr: float = 0.005 # Concentration learning rate

    iterations: int = 1500 # Max number of iterations

    obstacle_lambda: float = 0.1 # Weight for the obstacle penalty term in the loss function

    early_stopping_patience: int = 100      # How many iterations to wait for an improvement
    early_stopping_min_delta: float = 1e-3  # Minimum improvement required to reset the patience counter
    ema_alpha: float = 0.6                  # Smoothing factor (Lower = smoother, more memory of past loss)

    do_eval: bool = field(default=False, action="store_true") # Evaluate model during training using ground truth
    eval_interval: int = 25 # Model evaluation interval

    live_vis: bool = field(default=False, action="store_true") # Visualize training progress in real-time

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.001 # Threshold for gradient-based densification
    scale_threshold: float = 0.05 # Threshold used by original densification method. It decides whether the Gaussian should be splitted or cloned
    prune_threshold: float = 0.005 # Threshold for pruning Gaussians with low concentration
    densify_from: int = 100 # Densification will start at this iteration
    densify_until: int = 750 # Densification will stop at this iteration
    densify_interval: int = 50 # Iteration interval for densification
    original_dens: bool = field(default=False, action="store_true") # Use the original densification method instead of the proposed one based on Long-Axis Split approach

@dataclass
class Config:
    env: EnvironmentParams
    sim: SimulationParams
    train: TrainParams
    densify: DensificationParams

    # Global flag to silence console output
    quiet: bool = field(default=False, action="store_true")

    # "cuda" if available, "cpu" otherwise. Can be overwritten
    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> torch.device:
        if self.device_type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        return torch.device(self.device_type)

@dataclass
class ExperimentConfig(Config):
    num_seeds: int = 30 # Number of random seeds for the experiment