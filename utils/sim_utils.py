import os
import math
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

from config import Config
from utils.geometry_utils import iter_ray_cell_intersections


@dataclass
class MeasurementBatch:
    """Sensor readings and their geometries."""
    beams: torch.Tensor         # Shape: (N, 2, 2)
    measurements: torch.Tensor  # Shape: (N,)

@dataclass
class GroundTruth:
    """Holds perfect knowledge of the world."""
    gas_map: np.ndarray         # Shape: (H, W)
    y_true: torch.Tensor        # Shape: (N,) - Noise-free measurements

@dataclass
class EnvironmentContext:
    """Static physical properties of the environment."""
    obstacles: np.ndarray | None = None
    ground_truth: GroundTruth | None = None


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

def simulate_gas_integrals(gas_concentration_map: np.ndarray, beams: list, cell_dimensions_meters: float, quiet: bool = False) -> list[float]:
    """Simulates a TDLAS raytracing measurement with path length calculation within cells."""
    integral_concentrations = []
    rows, cols = gas_concentration_map.shape
    map_width = cols * cell_dimensions_meters
    map_height = rows * cell_dimensions_meters

    for beam in tqdm(beams, desc="Gas Integrals Simulation", dynamic_ncols=True, disable=quiet):
        (x0, y0), (x1, y1) = beam
        if not (0 <= x0 <= map_width and 0 <= y0 <= map_height and
                0 <= x1 <= map_width and 0 <= y1 <= map_height):
            print(f"Warning: Beam ({x0}, {y0}) - ({x1}, {y1}) is out of map boundaries. Skipping.")
            integral_concentrations.append(0.0)
            continue

        weighted_concentration = 0.0

        for row, col, path_length in iter_ray_cell_intersections(
            beam, gas_concentration_map.shape, cell_dimensions_meters
        ):
            weighted_concentration += gas_concentration_map[row, col] * path_length
                    
        integral_concentrations.append(weighted_concentration)

    return integral_concentrations

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

def generate_simulation_data(cfg: Config) -> tuple[MeasurementBatch, EnvironmentContext]:
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
        map_w = cols * cfg.env.cell_size
        map_h = rows * cfg.env.cell_size
        cfg.env.map_size = (map_w, map_h)
    else:
        if not quiet: print("Generating procedural ground truth...")
        map_w, map_h = cfg.env.map_size
        grid_w = int(map_w / cfg.env.cell_size)
        grid_h = int(map_h / cfg.env.cell_size)

        img_gt = generate_fractal_gas_distribution(grid_size=(grid_h, grid_w))

    # ------ Beams ------
    if not quiet: print("Generating beams...")
    beams_list = []

    num_random_beams = cfg.sim.num_beams // 2
    num_radial_beams = cfg.sim.num_beams - num_random_beams 
        
    beams_list += generate_random_beams(cfg.env.map_size, num_random_beams)
    beams_list += generate_radial_beams(cfg.env.map_size, num_radial_beams)

    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)

    # ------- Measurements --------
    measurements_list = simulate_gas_integrals(img_gt, beams_list, cfg.env.cell_size, quiet=quiet)
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

    return (
        MeasurementBatch(beams=beams_tensor, measurements=measurements),
        EnvironmentContext(ground_truth=GroundTruth(gas_map=img_gt, y_true=y_true))
    )