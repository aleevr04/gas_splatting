import math
import torch
import numpy as np
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon
from tqdm import tqdm
from scipy.sparse import dok_matrix
from scipy.ndimage import gaussian_filter

from config import Config


@dataclass
class SimulationData:
    beams: torch.Tensor
    img_gt: np.ndarray
    measurements: torch.Tensor
    y_true: torch.Tensor


# ==========================================
#       GEOMETRY FUNCTIONS
# ==========================================

def xy2cell(pos_m: tuple, cell_size_m: float) -> tuple:
    """Translates (x, y) coordinates in meters to (row, column) indices in a 2D array."""    
    column = int(pos_m[0] // cell_size_m)
    row = int(pos_m[1] // cell_size_m)
    return row, column

def cell2xy(cell_rc: tuple, cell_size_m: float) -> tuple:
    """Translates (row, column) cell coordinates to (x, y) coordinates in meters."""    
    x = cell_rc[1] * cell_size_m + cell_size_m/2
    y = cell_rc[0] * cell_size_m + cell_size_m/2
    return x, y


# ==========================================
#           GAS DISTRIBUTION
# ==========================================

def generate_gas_distribution(grid_size: tuple, num_blobs: int = 5) -> np.ndarray:
    """Generates a grid gas distribution given the amount of gas sources."""
    
    rows, cols = grid_size
    gas_map = np.zeros(grid_size)

    # Place random concetration in random cells
    for _ in range(num_blobs):
        r = np.random.randint(rows // 5, 4 * rows // 5)
        c = np.random.randint(cols // 5, 4 * cols // 5)
        gas_map[r, c] = np.random.uniform(5.0, 10.0)

    # Smooth result to get cloudy shapes
    sigma_r = rows / np.random.uniform(6, 12)
    sigma_c = cols / np.random.uniform(6, 12)
    gas_map = gaussian_filter(gas_map, sigma=(sigma_r, sigma_c))

    # Add noise
    noise = np.random.rand(rows, cols)
    gas_map += noise * (gas_map.max() * 0.1)

    # Remove concentration from cells below threshold
    threshold = gas_map.max() * 0.3
    gas_map[gas_map < threshold] = 0

    # Normalize [0,1]
    if gas_map.max() > 0:
        gas_map = gas_map / gas_map.max()

    return gas_map

# ==========================================
#            BEAM FUNCTIONS
# ==========================================

def generate_radial_beams(map_size_m: tuple, num_beams: int):
    """Generates beams starting from bottom corners with endpoints distributed homogeneously in angle."""
    beams = []
    map_x, map_y = map_size_m

    num_beams_left = int(num_beams // 2)
    if num_beams_left > 0:
        angles_left = np.linspace(0, np.pi/2, num_beams_left)
        for angle in angles_left:
            x0, y0 = 0.0, 0.0
            if angle == 0:
                x1, y1 = map_x, 0.0
            elif angle == np.pi / 2:
                x1, y1 = 0.0, map_y
            else:
                if angle <= math.atan2(map_y, map_x):
                    x1 = map_x
                    y1 = map_x * np.tan(angle)
                else:
                    x1 = map_y / np.tan(angle)
                    y1 = map_y
            beams.append(((x0, y0), (x1, y1)))

    num_beams_right = num_beams - num_beams_left
    if num_beams_right > 0:
        angles_right = np.linspace(0, np.pi/2, num_beams_right)
        for angle in angles_right:
            x0, y0 = map_x, 0.0
            if angle == 0:
                x1, y1 = 0.0, 0.0
            elif angle == np.pi / 2:
                x1, y1 = map_x, map_y
            else:
                if angle <= math.atan2(map_y, map_x):
                    x1 = 0.0
                    y1 = map_x * np.tan(angle)
                else:
                    x1 = map_x - (map_y / np.tan(angle))
                    y1 = map_y
            beams.append(((x0, y0), (x1, y1)))
    
    return beams

def generate_random_beams(map_size_m: tuple, num_beams: int):
    """Generates random TDLAS beams from the perimeter of a map in meters."""
    beams = []
    map_x, map_y = map_size_m

    for _ in range(num_beams):
        start_edge = np.random.choice(['left', 'right', 'bottom', 'top'])
        if start_edge == 'left':
            x0, y0 = 0.0, np.random.uniform(0, map_y)
        elif start_edge == 'right':
            x0, y0 = map_x, np.random.uniform(0, map_y)
        elif start_edge == 'bottom':
            x0, y0 = np.random.uniform(0, map_x), 0.0
        else:
            x0, y0 = np.random.uniform(0, map_x), map_y

        end_edges = [edge for edge in ['left', 'right', 'bottom', 'top'] if edge != start_edge]
        end_edge = np.random.choice(end_edges)
        if end_edge == 'left':
            x1, y1 = 0.0, np.random.uniform(0, map_y)
        elif end_edge == 'right':
            x1, y1 = map_x, np.random.uniform(0, map_y)
        elif end_edge == 'bottom':
            x1, y1 = np.random.uniform(0, map_x), 0.0
        else:
            x1, y1 = np.random.uniform(0, map_x), map_y

        beams.append(((x0, y0), (x1, y1)))
        
    return beams

def generate_horizontal_vertical_beams(map_size_m: tuple, num_beams: int):
    """Generates half horizontal and half vertical beams, evenly distributed."""
    beams = []
    map_x, map_y = map_size_m
    
    h_beams = int(num_beams*map_y // (map_x+map_y))

    if h_beams > 0:
        y_positions = np.linspace(0, map_y, h_beams, endpoint=False)
        for y in y_positions:
            beams.append(((0.0, y), (map_x, y)))
            
    remaining_beams = num_beams - h_beams
    if remaining_beams > 0:
        x_positions = np.linspace(0, map_x, remaining_beams, endpoint=False)
        for x in x_positions:
            beams.append(((x, 0.0), (x, map_y)))

    return beams


# ==========================================
#     BEAM GAS INTEGRAL / SYSTEM MATRIX
# ==========================================

def simulate_gas_integrals(gas_concentration_map: np.ndarray, beams: list, cell_dimensions_meters: float) -> list[float]:
    """Simulates a TDLAS raytracing measurement with path length calculation within cells."""
    integral_concentrations = []
    rows, cols = gas_concentration_map.shape
    map_width = cols * cell_dimensions_meters
    map_height = rows * cell_dimensions_meters

    for (x0, y0), (x1, y1) in tqdm(beams, desc="Gas Integrals Simulation"):
        if not (0 <= x0 <= map_width and 0 <= y0 <= map_height and
                0 <= x1 <= map_width and 0 <= y1 <= map_height):
            print(f"Warning: Beam ({x0}, {y0}) - ({x1}, {y1}) is out of map boundaries. Skipping.")
            integral_concentrations.append(0.0)
            continue

        beam_line = LineString([(x0, y0), (x1, y1)])
        weighted_concentration = 0.0

        min_x_cell = max(0, int(np.floor(min(x0, x1) / cell_dimensions_meters)))
        max_x_cell = min(cols - 1, int(np.floor(max(x0, x1) / cell_dimensions_meters)))
        min_y_cell = max(0, int(np.floor(min(y0, y1) / cell_dimensions_meters)))
        max_y_cell = min(rows - 1, int(np.floor(max(y0, y1) / cell_dimensions_meters)))

        for r in range(min_y_cell, max_y_cell + 1):
            for c in range(min_x_cell, max_x_cell + 1):
                x_min = c * cell_dimensions_meters
                x_max = (c + 1) * cell_dimensions_meters
                y_min = r * cell_dimensions_meters
                y_max = (r + 1) * cell_dimensions_meters
                
                cell_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                intersection = beam_line.intersection(cell_polygon)

                if not intersection.is_empty and intersection.geom_type == 'LineString':
                    path_length_in_cell = intersection.length
                    concentration = gas_concentration_map[r, c]
                    weighted_concentration += concentration * path_length_in_cell
                    
        integral_concentrations.append(weighted_concentration)

    return integral_concentrations

def create_system_matrix_sparse(grid_size: tuple, beams: list, cell_dimensions_meters: float) -> dok_matrix:
    """Creates the sparse system matrix A for TDLAS tomography."""
    rows, cols = grid_size
    num_cells = rows * cols
    num_beams = len(beams)
    A = dok_matrix((num_beams, num_cells), dtype=float)

    for i, ((x0, y0), (x1, y1)) in tqdm(enumerate(beams), desc="Building System Matrix", total=num_beams):
        beam_line = LineString([(x0, y0), (x1, y1)])

        min_c = max(0, int(np.floor(min(x0, x1) / cell_dimensions_meters)))
        max_c = min(cols - 1, int(np.floor(max(x0, x1) / cell_dimensions_meters)))
        min_r = max(0, int(np.floor(min(y0, y1) / cell_dimensions_meters)))
        max_r = min(rows - 1, int(np.floor(max(y0, y1) / cell_dimensions_meters)))

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                x_min = c * cell_dimensions_meters
                y_min = r * cell_dimensions_meters
                x_max = (c + 1) * cell_dimensions_meters
                y_max = (r + 1) * cell_dimensions_meters

                cell_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                intersection = beam_line.intersection(cell_polygon)

                if not intersection.is_empty and intersection.geom_type == 'LineString':
                    path_length_in_cell = intersection.length
                    cell_index = r * cols + c
                    A[i, cell_index] = path_length_in_cell

    return A


# ==========================================
#         GENERATE GROUND TRUTH
# ==========================================

def add_measurement_noise(y_true, snr_db=30):
    y_true_np = y_true.cpu().numpy()
    signal_power = np.mean(y_true_np**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=y_true_np.shape)
    y_noisy = y_true_np + noise
    y_noisy[y_noisy < 0] = 0
    return torch.tensor(y_noisy, dtype=torch.float32, device=y_true.device)

def generate_simulation_data(cfg: Config) -> SimulationData:
    """Generates gas distribution (ground truth), beams and measurements"""
    
    if cfg.sim.seed:
        np.random.seed(cfg.sim.seed)

    # ----- Ground Truth -----
    if cfg.sim.gt_file:
        print(f"Loading ground truth from {cfg.sim.gt_file}...")
        img_gt = np.loadtxt(cfg.sim.gt_file, delimiter=',')
       
        rows, cols = img_gt.shape
        map_w = cols * cfg.sim.cell_size
        map_h = rows * cfg.sim.cell_size
        cfg.sim.map_size = (map_w, map_h)
    else:
        print("Generating procedural ground truth...")
        map_w, map_h = cfg.sim.map_size
        grid_w = int(map_w / cfg.sim.cell_size)
        grid_h = int(map_h / cfg.sim.cell_size)

        img_gt = generate_gas_distribution(
            grid_size=(grid_h, grid_w), 
            num_blobs=cfg.sim.num_blobs
        )

    # ------ Beams ------
    print("Generating beams...")
    
    num_random_beams = cfg.sim.num_beams // 2
    num_radial_beams = cfg.sim.num_beams - num_random_beams 
    
    beams_list = generate_random_beams(cfg.sim.map_size, num_random_beams)
    beams_list += generate_radial_beams(cfg.sim.map_size, num_radial_beams)

    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)

    # ------- Measurements --------
    measurements_list = simulate_gas_integrals(img_gt, beams_list, cfg.sim.cell_size)
    y_true = torch.tensor(measurements_list, dtype=torch.float32, device=cfg.device)

    if cfg.sim.noise:
        print(f"Adding noise to the measurements ({cfg.sim.snr_db} dB)...")
        measurements = add_measurement_noise(y_true, snr_db=cfg.sim.snr_db)
    else:
        measurements = y_true

    return SimulationData(
        beams=beams_tensor,
        img_gt=img_gt,
        measurements=measurements,
        y_true=y_true
    )