import math
import numpy as np
import torch
from scipy.sparse.linalg import lsqr
from skimage.feature import peak_local_max

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import (
    MeasurementBatch,
    create_system_matrix_sparse,
    cell2xy
)

def lsqr_initialization(batch: MeasurementBatch, cfg: Config):
    """
    Runs a fast algebraic reconstruction (Least Squares) to get an initial estimate of the map and gaussians parameters.
    """
    
    # Extract parameters from config
    map_size = cfg.sim.map_size
    min_gaussians = cfg.init.min_gaussians
    
    max_dim = max(map_size[0], map_size[1])
    coarse_cell_size = cfg.init.coarse_proportion * max_dim
    
    # Extract data from batch
    beams = batch.beams.tolist()
    if isinstance(batch.measurements, torch.Tensor):
        b = batch.measurements.cpu().numpy()
    else:
        b = np.array(batch.measurements)
        
    # Build system matrix
    coarse_w = math.ceil(map_size[0] / coarse_cell_size)
    coarse_h = math.ceil(map_size[1] / coarse_cell_size)
    A_sparse = create_system_matrix_sparse((coarse_h, coarse_w), beams, coarse_cell_size)
        
    # Solve Ax = b (Least Squares)
    result = lsqr(A_sparse, b, damp=0.1, iter_lim=50)
    x_coarse = result[0]
    
    # Avoid negative values
    x_coarse[x_coarse < 0] = 0
    img_coarse = x_coarse.reshape((coarse_h, coarse_w))
    
    # Find peaks in reconstruction
    coordinates_int = peak_local_max(img_coarse, min_distance=1, threshold_rel=0.6)
    
    pos = []
    concentration = []
    
    # Peaks found. Place exactly one Gaussian per peak.
    if len(coordinates_int) > 0:
        for coord in coordinates_int:
            row, col = coord
            x, y = cell2xy((row, col), coarse_cell_size)
            
            x = min(max(x, 0.0), map_size[0] - 1e-5)
            y = min(max(y, 0.0), map_size[1] - 1e-5)
            
            pos.append([x, y])
            concentration.append(img_coarse[row, col])
            
    # No peaks found. Fallback to importance sampling.
    else:
        weights = img_coarse.flatten()
        sum_weights = np.sum(weights)
        
        if sum_weights > 1e-6:
            # Importance Sampling based on the coarse map density
            probabilities = weights / sum_weights
            sampled_indices = np.random.choice(
                a=len(probabilities), 
                size=min_gaussians, 
                replace=True, 
                p=probabilities
            )
            rows, cols = np.unravel_index(sampled_indices, img_coarse.shape)
            
            for r, c in zip(rows, cols):
                x, y = cell2xy((r, c), coarse_cell_size)
                x = min(max(x, 0.0), map_size[0] - 1e-5)
                y = min(max(y, 0.0), map_size[1] - 1e-5)
                
                pos.append([x, y])
                concentration.append(img_coarse[r, c] + 0.01) # Small offset
        else:
            # Map is completely empty (all zeros)
            for _ in range(min_gaussians):
                pos.append([
                    np.random.uniform(0, map_size[0]), 
                    np.random.uniform(0, map_size[1])
                ])
                concentration.append(0.01)

    std = coarse_cell_size * 1.5 

    return (torch.tensor(pos, dtype=torch.float32), 
            torch.tensor(concentration, dtype=torch.float32), 
            torch.tensor(std, dtype=torch.float32), 
            img_coarse)


def setup_gs_model(batch: MeasurementBatch, cfg: Config):
    """
    Initializes Gas Splatting model using simulation data.

    Args:
        batch (MeasurementBatch): Batch containing beams geometry and their measurements.
        cfg (Config): Configuration object with model and simulation parameters.

    Returns:
        tuple[GasSplattingModel, torch.Tensor, np.ndarray]: A tuple containing:
            - model: The initialized GasSplattingModel.
            - init_pos: Tensor with the initial positions of the gaussians.
            - img_coarse: Visual result of the coarse initialization phase.
    """

    init_pos, init_concentration, init_std, img_coarse = lsqr_initialization(batch, cfg)
    initial_gaussians = init_pos.shape[0]

    if initial_gaussians > 0:
        model = GasSplattingModel(initial_gaussians, cfg).to(cfg.device)
        model.initialize_gaussians(
            init_pos.to(cfg.device), 
            init_concentration.to(cfg.device), 
            init_std.to(cfg.device)
        )
    else:
        model = GasSplattingModel(1, cfg).to(cfg.device)
        init_pos = model.get_pos().detach().cpu().numpy()

    return model, init_pos, img_coarse