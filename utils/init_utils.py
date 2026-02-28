import numpy as np
import torch
from scipy.sparse.linalg import lsqr
from skimage.feature import peak_local_max

from utils.sim_utils import create_system_matrix_sparse, cell2xy

def lsqr_initialization(beams, measurements, map_size, num_gaussians=None, coarse_res=20):
    """
    Runs a fast algebraic reconstruction (Least Squares) to get an initial estimate of the map and where to place the gaussians.
    """
    
    # Build system matrix
    cell_size = map_size / coarse_res
    A_sparse = create_system_matrix_sparse((coarse_res, coarse_res), beams, cell_size)
    
    if isinstance(measurements, torch.Tensor):
        b = measurements.cpu().numpy()
    else:
        b = np.array(measurements)
        
    # Solve Ax = b (Least Squares)
    result = lsqr(A_sparse, b, damp=0.1, iter_lim=50)
    x_coarse = result[0]
    
    # Avoid negative values
    x_coarse[x_coarse < 0] = 0
    img_coarse = x_coarse.reshape((coarse_res, coarse_res))

    if num_gaussians:
        coordinates_int = peak_local_max(img_coarse, min_distance=1, num_peaks=num_gaussians)
    else:
        coordinates_int = peak_local_max(img_coarse, min_distance=1)
    
    pos = []
    concentration = []
    
    for coord in coordinates_int:
        row, col = coord
        
        x, y = cell2xy((row, col), cell_size)
        
        val = img_coarse[row, col]
        pos.append([x, y])
        concentration.append(val)
        
    # Fill until num_gaussians
    if num_gaussians:
        while len(pos) < num_gaussians:
            pos.append([np.random.uniform(0, map_size), np.random.uniform(0, map_size)])
            concentration.append(0.1)

    std = cell_size * 1.5 

    return (torch.tensor(pos, dtype=torch.float32), 
            torch.tensor(concentration, dtype=torch.float32), 
            std, 
            img_coarse)