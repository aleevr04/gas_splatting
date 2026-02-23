import argparse
from dataclasses import dataclass, fields

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
    no_gauss_filter: bool = False

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
    l1_reg: float = 0.1

@dataclass
class DensificationParams:
    gradient_threshold: float = 0.0004
    scale_threshold: float = 2.5
    prune_threshold: float = 0.05
    densify_from: int = 200
    densify_until: int = 800
    densify_interval: int = 200

def parse_args_into_dataclasses(*dataclass_types):
    parser = argparse.ArgumentParser(description="Gas Splatting parameters")
    
    # Add one argument group for each dataclass
    for dc in dataclass_types:
        group = parser.add_argument_group(dc.__name__)
        for f in fields(dc):
            if f.type == bool:
                group.add_argument(f"--{f.name}", action="store_true", default=f.default, help=f"Default: {f.default}")
            else:
                group.add_argument(f"--{f.name}", type=f.type, default=f.default, help=f"Default: {f.default}")
            
    # Parse arguments
    args = parser.parse_args()
    
    # Return dataclasses instances with updated information
    results = []
    for dc in dataclass_types:
        kwargs = {f.name: getattr(args, f.name) for f in fields(dc)}
        results.append(dc(**kwargs))
        
    return results