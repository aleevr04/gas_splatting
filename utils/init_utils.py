import math
import numpy as np
import torch
from scipy.sparse.linalg import lsqr
from skimage.feature import peak_local_max

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import (
    SimulationData,
    create_system_matrix_sparse,
    cell2xy
)

def lsqr_initialization(beams: list, measurements, map_size: tuple[float, float], num_gaussians=None, coarse_cell_size: float = 2.5):
    """
    Runs a fast algebraic reconstruction (Least Squares) to get an initial estimate of the map and gaussians parameters.
    """
    
    # Build system matrix
    coarse_w = math.ceil(map_size[0] / coarse_cell_size)
    coarse_h = math.ceil(map_size[1] / coarse_cell_size)
    A_sparse = create_system_matrix_sparse((coarse_h, coarse_w), beams, coarse_cell_size)
    
    if isinstance(measurements, torch.Tensor):
        b = measurements.cpu().numpy()
    else:
        b = np.array(measurements)
        
    # Solve Ax = b (Least Squares)
    result = lsqr(A_sparse, b, damp=0.1, iter_lim=50)
    x_coarse = result[0]
    
    # Avoid negative values
    x_coarse[x_coarse < 0] = 0
    img_coarse = x_coarse.reshape((coarse_h, coarse_w))

    if num_gaussians:
        coordinates_int = peak_local_max(img_coarse, min_distance=1, threshold_rel=0.6, num_peaks=num_gaussians)
    else:
        coordinates_int = peak_local_max(img_coarse, min_distance=1, threshold_rel=0.6)
    
    pos = []
    concentration = []
    
    for coord in coordinates_int:
        row, col = coord
        
        x, y = cell2xy((row, col), coarse_cell_size)
        
        x = min(max(x, 0.0), map_size[0] - 1e-5)
        y = min(max(y, 0.0), map_size[1] - 1e-5)

        val = img_coarse[row, col]
        pos.append([x, y])
        concentration.append(val)
        
    # Fill until num_gaussians
    if num_gaussians:
        while len(pos) < num_gaussians:
            pos.append([
                np.random.uniform(0, map_size[0]), # Width
                np.random.uniform(0, map_size[1])  # Height
            ])
            concentration.append(0.1)

    std = coarse_cell_size * 1.5 

    return (torch.tensor(pos, dtype=torch.float32), 
            torch.tensor(concentration, dtype=torch.float32), 
            torch.tensor(std, dtype=torch.float32), 
            img_coarse)

def setup_gs_model(sim_data: SimulationData, cfg: Config):
    """
    Initializes Gas Splatting model using simulation data.

    Args:
        sim_data (SimulationData): Simulation data containing beams and measurements.
        cfg (Config): Configuration object with model and simulation parameters.

    Returns:
        tuple[GasSplattingModel, torch.Tensor, np.ndarray]: A tuple containing:
            - model: The initialized GasSplattingModel.
            - init_pos: Tensor with the initial positions of the gaussians.
            - img_coarse: Visual result of the coarse initialization phase.
    """

    max_dim = max(cfg.sim.map_size[0], cfg.sim.map_size[1])
    coarse_cell_size = cfg.init.coarse_proportion * max_dim

    init_pos, init_concentration, init_std, img_coarse = lsqr_initialization(
        sim_data.beams.tolist(), 
        sim_data.measurements, 
        cfg.sim.map_size, 
        num_gaussians=cfg.init.initial_gaussians,
        coarse_cell_size=coarse_cell_size
    )
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