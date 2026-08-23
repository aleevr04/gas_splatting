import torch
from dataclasses import dataclass, field
from typing import Any, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import MeasurementBatch, extract_candidate_positions

@dataclass
class InitializationData:
    """Data class to hold the initialization results"""
    pos: torch.Tensor
    concentration: torch.Tensor
    std: torch.Tensor
    # Optional dictionary to store additional information about the initialization process
    info: Dict[str, Any] = field(default_factory=dict) 

def injection_initialization(batch: MeasurementBatch, cfg: Config):
    """
    Initializes Gaussians based on high-concentration beams using Continuous Spatial NMS.
    """
    # Dynamic scale
    map_w, map_h = cfg.env.map_size
    init_std_val = min(map_w, map_h) * 0.1
    
    final_pos = extract_candidate_positions(
        beams=batch.beams,
        importance_scores=batch.measurements,
        min_dist=init_std_val,
        device=cfg.device
    )
    
    num_final = final_pos.shape[0]
    
    if num_final == 0:
        return InitializationData(
            pos=torch.empty((0, 2), device=cfg.device),
            concentration=torch.empty((0,), device=cfg.device),
            std=torch.empty((0,), device=cfg.device)
        )
        
    std = torch.full((num_final,), init_std_val, dtype=torch.float32, device=cfg.device)
    concentration = torch.full((num_final,), 0.1, dtype=torch.float32, device=cfg.device)
    
    return InitializationData(pos=final_pos, concentration=concentration, std=std)

def setup_gs_model(batch: MeasurementBatch, cfg: Config) -> tuple[GasSplattingModel, InitializationData]:
    """
    Initializes Gas Splatting model.

    Args:
        batch (MeasurementBatch): Batch containing beams geometry and their measurements.
        cfg (Config): Configuration object with model and other parameters.

    Returns:
        tuple[GasSplattingModel, InitializationData]: Initialized Gas Splatting model and the initialization data.
    """

    init_data = injection_initialization(batch, cfg)
        
    initial_gaussians = init_data.pos.shape[0]

    if initial_gaussians > 0:
        model = GasSplattingModel(initial_gaussians, cfg).to(cfg.device)
        model.initialize_gaussians(
            init_data.pos.to(cfg.device), 
            init_data.concentration.to(cfg.device), 
            init_data.std.to(cfg.device)
        )
    else:
        model = GasSplattingModel(1, cfg).to(cfg.device)
        init_data.pos = model.get_pos().detach().cpu()

    return model, init_data