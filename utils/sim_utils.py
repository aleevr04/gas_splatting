import os
import math
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon
from scipy.sparse import dok_matrix
from scipy.ndimage import gaussian_filter

from config import Config


@dataclass
class MeasurementBatch:
    """
    Sensor readings and their geometries.
    """
    beams: torch.Tensor         # Shape: (N, 2, 2)
    measurements: torch.Tensor  # Shape: (N,)

@dataclass
class SimulationData:
    """
    Represents a complete simulated world. 
    It holds the ground truth and the resulting sensor readings.
    """
    ground_truth: np.ndarray    # Shape: (H, W)
    batch: MeasurementBatch
    
    # Noise-free measurements
    y_true: torch.Tensor        # Shape: (N,)
    
    # Obstacle occupancy grid
    obstacles: np.ndarray | None = None


# ==========================================
#       GEOMETRY FUNCTIONS
# ==========================================

def xy2cell(pos_m: tuple, cell_size_m: float) -> tuple[int, int]:
    """Translates (x, y) coordinates in meters to (row, column) indices in a 2D array."""    
    column = int(pos_m[0] // cell_size_m)
    row = int(pos_m[1] // cell_size_m)
    return row, column

def cell2xy(cell_rc: tuple, cell_size_m: float) -> tuple[float, float]:
    """Translates (row, column) cell coordinates to (x, y) coordinates in meters."""    
    x = cell_rc[1] * cell_size_m + cell_size_m/2
    y = cell_rc[0] * cell_size_m + cell_size_m/2
    return x, y


# ==========================================
#           GAS DISTRIBUTION
# ==========================================

def generate_fractal_gas_distribution(
        grid_size, scale_fraction=0.2, octaves=3, 
        threshold=0.3, power=2.0, center_bias=2.0):
    """
    Generates center-biased, scale-invariant gas distribution using fractal noise.
    
    - scale_fraction: Size of base blobs relative to map size (e.g., 0.15 = 15%).
    - center_bias: How aggressively to pull clouds to the center. 
                   0.0 = no bias (everywhere), >2.0 = tightly packed in center.
    """
    height, width = grid_size
    
    # Dynamic Scale
    base_scale = min(height, width) * scale_fraction
    
    # Generate Fractal Noise
    noise = np.zeros(grid_size)
    frequency, amplitude = 1.0, 1.0
    
    for _ in range(octaves):
        base = np.random.rand(*grid_size)
        smoothed = gaussian_filter(base, sigma=base_scale / frequency)
        noise += smoothed * amplitude
        amplitude *= 0.5
        frequency *= 2.0
        
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Create Gaussian Envelope (Center Mask)
    if center_bias > 0.0:
        y, x = np.indices(grid_size)
        center_y, center_x = height / 2.0, width / 2.0
        
        # Calculate squared distance from the center for every cell
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        max_dist_sq = (width / 2.0)**2 + (height / 2.0)**2
        
        # Create the mask: e^(-(distance^2) / variance)
        variance = max_dist_sq / center_bias
        mask = np.exp(-dist_sq / variance)
        
        # Apply the envelope to force edges towards zero
        noise = noise * mask

    # Threshold and Power Curve
    gas_map = np.maximum(0, noise - threshold)
    
    if gas_map.max() > 0:
        gas_map = gas_map / gas_map.max()
        
    gas_map = gas_map ** power
    
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

def simulate_gas_integrals(gas_concentration_map: np.ndarray, beams: list, cell_dimensions_meters: float, quiet: bool = False) -> list[float]:
    """Simulates a TDLAS raytracing measurement with path length calculation within cells."""
    integral_concentrations = []
    rows, cols = gas_concentration_map.shape
    map_width = cols * cell_dimensions_meters
    map_height = rows * cell_dimensions_meters

    for (x0, y0), (x1, y1) in tqdm(beams, desc="Gas Integrals Simulation", dynamic_ncols=True, disable=quiet):
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

def create_system_matrix_sparse(grid_size: tuple, beams: list, cell_dimensions_meters: float, quiet: bool = False) -> dok_matrix:
    """Creates the sparse system matrix A for TDLAS tomography."""
    rows, cols = grid_size
    num_cells = rows * cols
    num_beams = len(beams)
    A = dok_matrix((num_beams, num_cells), dtype=float)

    for i, ((x0, y0), (x1, y1)) in tqdm(enumerate(beams), desc="Building System Matrix", total=num_beams, dynamic_ncols=True, disable=quiet):
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

    quiet = cfg.quiet

    # ----- Ground Truth -----
    if cfg.sim.gt_file is not None:
        if not os.path.exists(cfg.sim.gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {cfg.sim.gt_file}")
        if not quiet: print(f"Loading ground truth from {cfg.sim.gt_file}...")
        img_gt = np.loadtxt(cfg.sim.gt_file, delimiter=',')
       
        rows, cols = img_gt.shape
        map_w = cols * cfg.sim.cell_size
        map_h = rows * cfg.sim.cell_size
        cfg.sim.map_size = (map_w, map_h)
    else:
        if not quiet: print("Generating procedural ground truth...")
        map_w, map_h = cfg.sim.map_size
        grid_w = int(map_w / cfg.sim.cell_size)
        grid_h = int(map_h / cfg.sim.cell_size)

        img_gt = generate_fractal_gas_distribution(grid_size=(grid_h, grid_w))

    # ------ Beams ------
    if not quiet: print("Generating beams...")
    beams_list = []

    num_random_beams = cfg.sim.num_beams // 2
    num_radial_beams = cfg.sim.num_beams - num_random_beams 
        
    beams_list += generate_random_beams(cfg.sim.map_size, num_random_beams)
    beams_list += generate_radial_beams(cfg.sim.map_size, num_radial_beams)

    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)

    # ------- Measurements --------
    measurements_list = simulate_gas_integrals(img_gt, beams_list, cfg.sim.cell_size, quiet=quiet)
    y_true = torch.tensor(measurements_list, dtype=torch.float32, device=cfg.device)

    if cfg.sim.noise:
        if not quiet: print(f"Adding noise to the measurements ({cfg.sim.snr_db} dB)...")
        num_beams = len(beams_list)
        if num_beams > 0:
            measurements = add_measurement_noise(y_true[:num_beams], snr_db=cfg.sim.snr_db)
        else:
            measurements = y_true
    else:
        measurements = y_true

    return SimulationData(
        ground_truth=img_gt,
        batch=MeasurementBatch(beams=beams_tensor, measurements=measurements),
        y_true=y_true
    )

# ================================
#      INJECTION UTILITIES
# ================================

def extract_candidate_positions(beams: torch.Tensor, importance_scores: torch.Tensor, min_dist: float, device: torch.device) -> torch.Tensor:
    """
    Extracts optimal locations for new Gaussians by computing geometric intersections 
    of anomalous beams and applying Continuous Spatial NMS.

    Args:
        beams: Tensor of shape (N, 2, 2) representing the start and end points of the beams.
        importance_scores: Tensor of shape (N,) representing the metric to evaluate (measurements or residuals).
        min_dist: Minimum spatial distance between generated points to avoid redundancy.
        device: Torch device.
        
    Returns:
        Tensor of shape (M, 2) with the filtered candidate positions.
    """
    p0 = beams[:, 0, :]
    p1 = beams[:, 1, :]
    lengths = torch.norm(p1 - p0, dim=1)
    
    normalized_scores = importance_scores / torch.clamp(lengths, min=1e-5)
    
    pos_mask = normalized_scores > 0
    if not pos_mask.any():
        return torch.empty((0, 2), device=device)
        
    # 1. Statistical Filter: Mean + Std
    valid_scores = normalized_scores[pos_mask]
    mean_score = torch.mean(valid_scores)
    std_score = torch.std(valid_scores)
    
    threshold = mean_score + 1.0 * (std_score + 1e-6)
    
    candidate_mask = pos_mask & (normalized_scores > threshold)
    candidate_indices = torch.where(candidate_mask)[0]
    
    num_candidates = candidate_indices.shape[0]
    if num_candidates < 2:
        return torch.empty((0, 2), device=device)

    # 2. Geometric Intersections of candidate segments
    candidate_beams = beams[candidate_indices]
    candidate_scores = normalized_scores[candidate_indices]
    
    i, j = torch.triu_indices(num_candidates, num_candidates, offset=1, device=device)
    
    b_i = candidate_beams[i] 
    b_j = candidate_beams[j] 
    
    x1, y1 = b_i[:, 0, 0], b_i[:, 0, 1]
    x2, y2 = b_i[:, 1, 0], b_i[:, 1, 1]
    x3, y3 = b_j[:, 0, 0], b_j[:, 0, 1]
    x4, y4 = b_j[:, 1, 0], b_j[:, 1, 1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    valid_intersection = torch.abs(denom) > 1e-7
    
    # Parametric line equations (t for beam_i, u for beam_j)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (denom + 1e-9)
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (denom + 1e-9)
    
    # A true segment intersection ONLY occurs if both t and u are between 0 and 1.
    # We use 0.01 and 0.99 to avoid intersections exactly at the emitters (radial origins).
    in_bounds = (t >= 0.01) & (t <= 0.99) & (u >= 0.01) & (u <= 0.99)
    keep_mask = valid_intersection & in_bounds
    
    if not keep_mask.any():
        return torch.empty((0, 2), device=device)
        
    # Calculate exact coordinates for valid intersections
    px = x1[keep_mask] + t[keep_mask] * (x2[keep_mask] - x1[keep_mask])
    py = y1[keep_mask] + t[keep_mask] * (y2[keep_mask] - y1[keep_mask])
    
    intersections = torch.stack([px, py], dim=-1)
    int_scores = candidate_scores[i[keep_mask]] + candidate_scores[j[keep_mask]]

    # 3. Continuous Spatial Non-Maximum Suppression (NMS)
    sorted_idx = torch.argsort(int_scores, descending=True)
    sorted_points = intersections[sorted_idx]
    
    keep_points = []
    for pt in sorted_points:
        if len(keep_points) > 0:
            kept_tensor = torch.stack(keep_points)
            distances = torch.norm(kept_tensor - pt, dim=1)
            if torch.min(distances) < min_dist:
                continue 
        keep_points.append(pt)
            
    return torch.stack(keep_points)