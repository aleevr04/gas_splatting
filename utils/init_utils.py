import torch
from dataclasses import dataclass, field
from typing import Any, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import MeasurementBatch

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
    Initializes Gaussians by sampling random points along the beams with the highest measurements.
    """
    measurements = batch.measurements
    beams = batch.beams
    
    # Select the top 20% most critical beams (ignoring empty environment noise)
    threshold = torch.quantile(measurements, 0.8)
    
    important_mask = measurements >= threshold
    target_beams = beams[important_mask]
    
    if target_beams.shape[0] == 0:
        target_beams = beams # Fallback
        
    # Sample one random point along each important beam
    p0 = target_beams[:, 0, :]
    p1 = target_beams[:, 1, :]
    u = torch.rand((target_beams.shape[0], 1), device=cfg.device)
    pos = p0 + u * (p1 - p0)
    
    # Define robust initial parameters (seed-like)
    init_std = cfg.sim.cell_size * 2.0
    std = torch.full((pos.shape[0],), init_std, dtype=torch.float32, device=cfg.device)
    concentration = torch.full((pos.shape[0],), 0.1, dtype=torch.float32, device=cfg.device)
    
    return InitializationData(pos=pos, concentration=concentration, std=std)

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