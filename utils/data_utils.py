import os
import sys
import json
import math
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.sim_utils import (
    EnvironmentContext,
    GroundTruth,
    MeasurementBatch,
    xy2cell, 
    simulate_gas_integrals, 
    generate_random_beams, 
    generate_radial_beams
)

def load_real_tdlas_data(filepath, cfg: Config) -> MeasurementBatch:
    """
    Extracts real measurements from a file and returns a MeasurementBatch. 
    It also updates the config map size properly.
    """
    beams_list = []
    measurements_list = []
    
    # Extract beams geometry and measurements
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Sensor and Reflector positions
            sx, sy, sz = data['sensorTF']['pose']['position']['x'], data['sensorTF']['pose']['position']['y'], data['sensorTF']['pose']['position']['z']
            rx, ry, rz = data['reflectorTF']['pose']['position']['x'], data['reflectorTF']['pose']['position']['y'], data['reflectorTF']['pose']['position']['z']
            
            # Concentration reading
            ppmxm = float(data['reading']['average_ppmxm'])
            
            # 2D projection
            len_3d = math.hypot(math.hypot(rx - sx, ry - sy), rz - sz)
            len_2d = math.hypot(rx - sx, ry - sy)
            
            ppmxm_2d = ppmxm * (len_2d / len_3d) if len_3d > 0 else 0
            
            beams_list.append(((sx, sy), (rx, ry)))
            measurements_list.append(ppmxm_2d)
            
    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)
    measurements_tensor = torch.tensor(measurements_list, dtype=torch.float32, device=cfg.device)
    
    # New map boundaries based on beams geometry
    all_coords = beams_tensor.view(-1, 2)
    min_coords = torch.min(all_coords, dim=0).values
    max_coords = torch.max(all_coords, dim=0).values

    margin = 0.5
    min_x, min_y = min_coords[0].item() - margin, min_coords[1].item() - margin
    max_x, max_y = max_coords[0].item() + margin, max_coords[1].item() + margin

    # Apply offset to beams
    offset = torch.tensor([-min_x, -min_y], device=cfg.device)
    beams_tensor += offset

    # Compute map size
    raw_w = max_x - min_x
    raw_h = max_y - min_y
    
    grid_w = math.ceil(raw_w / cfg.env.cell_size)
    grid_h = math.ceil(raw_h / cfg.env.cell_size)
    
    # Update new map size in cfg
    map_w = grid_w * cfg.env.cell_size
    map_h = grid_h * cfg.env.cell_size
    cfg.env.map_size = (map_w, map_h)

    # Return exclusively the raw measurements
    return MeasurementBatch(
        beams=beams_tensor,
        measurements=measurements_tensor
    )

def build_custom_real_scenario(
    cfg: Config, 
    real_data_path: str, 
    use_sim_beams: bool = True, 
    use_sim_gas: bool = False
):  
    if use_sim_beams and not use_sim_gas:
        raise ValueError(
            "Invalid combination: Synthetic beams require a simulated gas map to compute meaningful measurements."
        )

    # Load real data batch
    if os.path.exists(real_data_path):
        batch = load_real_tdlas_data(real_data_path, cfg)
    else:
        raise FileNotFoundError(f"Real data path is missing or invalid: {real_data_path}")

    if use_sim_beams:
        num_random_beams = cfg.sim.num_beams // 2
        num_radial_beams = cfg.sim.num_beams - num_random_beams 
        beams = []
        beams += generate_random_beams(cfg.env.map_size, num_random_beams)
        beams += generate_radial_beams(cfg.env.map_size, num_radial_beams)
        
        batch.beams = torch.tensor(beams, dtype=torch.float32, device=cfg.device)

    # If simulated gas is injected, generate ground truth for the environment context.
    if use_sim_gas:
        grid_w = math.ceil(cfg.env.map_size[0] / cfg.env.cell_size)
        grid_h = math.ceil(cfg.env.map_size[1] / cfg.env.cell_size)
        gas_map = np.zeros((grid_h, grid_w))

        source1 = (5.0, 7.0)
        source2 = (9.0, 4.0)

        s1r, s1c = xy2cell(source1, cfg.env.cell_size)
        s2r, s2c = xy2cell(source2, cfg.env.cell_size)

        if 0 <= s1r < grid_h and 0 <= s1c < grid_w:
            gas_map[s1r][s1c] = 60.0
        if 0 <= s2r < grid_h and 0 <= s2c < grid_w:
            gas_map[s2r][s2c] = 60.0

        gas_map = gaussian_filter(gas_map, sigma=1.0)
        
        # Recompute measurements
        measurements = simulate_gas_integrals(gas_map, batch.beams.tolist(), cfg.env.cell_size, quiet=cfg.quiet)
        batch.measurements = torch.tensor(measurements, dtype=torch.float32, device=cfg.device)
        
        return (
            batch,
            EnvironmentContext(ground_truth=GroundTruth(gas_map=gas_map, y_true=batch.measurements))
        )

    # If no simulation is injected, return the pure real batch
    return batch, EnvironmentContext()