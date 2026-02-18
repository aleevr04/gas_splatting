import numpy as np
import torch
from scipy.sparse.linalg import lsqr
from skimage.feature import peak_local_max

from utils.tomo_utils import create_system_matrix_sparse, cell2xy

def lsqr_initialization(beams, measurements, map_size, num_gaussians, coarse_res=20):
    """
    Runs a fast algebraic reconstruction (Least Squares) to get an initial estimate of the map and where to place the gaussians.
    """
    
    # 1. Crear matriz del sistema para un grid de baja resolución
    cell_size = map_size / coarse_res
    A_sparse = create_system_matrix_sparse((coarse_res, coarse_res), beams, cell_size)
    
    # 2. Convertir medidas a numpy si son tensores
    if isinstance(measurements, torch.Tensor):
        b = measurements.cpu().numpy()
    else:
        b = np.array(measurements)
        
    # 3. Resolver Ax = b (Mínimos Cuadrados) -> Reconstrucción "Sucia"
    #    LSQR es muy rápido y robusto para sistemas dispersos y mal condicionados
    result = lsqr(A_sparse, b, damp=0.1, iter_lim=50)
    x_coarse = result[0]
    
    # Limpiar valores negativos (físicamente imposibles)
    x_coarse[x_coarse < 0] = 0
    img_coarse = x_coarse.reshape((coarse_res, coarse_res))

    coordinates_int = peak_local_max(img_coarse, min_distance=1, num_peaks=num_gaussians)
    
    # Si peak_local_max no encuentra suficientes, rellenamos con aleatorios
    pos = []
    concentration = []
    
    for coord in coordinates_int:
        row, col = coord
        # Convertir índices de matriz (row, col) a coordenadas físicas (x, y)
        # Nota: en imagen row es Y invertida o Y normal dependiendo de la convención.
        # Usamos la función auxiliar de tomo_utils.
        x, y = cell2xy((row, col), cell_size)
        
        val = img_coarse[row, col]
        pos.append([x, y])
        concentration.append(val)
        
    # Rellenar si faltan (por si la imagen es muy plana)
    while len(pos) < num_gaussians:
        pos.append([np.random.uniform(0, map_size), np.random.uniform(0, map_size)])
        concentration.append(0.1) # Peso bajo por defecto

    # 5. Estimar el tamaño (sigma) inicial
    #    Una buena heurística es el tamaño de celda del grid grueso
    std = cell_size * 1.5 

    return (torch.tensor(pos, dtype=torch.float32), 
            torch.tensor(concentration, dtype=torch.float32), 
            std, 
            img_coarse)