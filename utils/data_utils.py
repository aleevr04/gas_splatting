import os
import sys
import json
import datetime
import math
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.sim_utils import (
    SimulationData, 
    xy2cell, 
    simulate_gas_integrals, 
    generate_random_beams, 
    generate_radial_beams
)

def save_experiment_results(metadata, results, folder="results"):
    """
    Saves metadata and results data in a JSON file
    """
    os.makedirs(folder, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = metadata.get("experiment_name", "exp")
    filename = f"{exp_name}_{timestamp}.json"
    filepath = os.path.join(folder, filename)
    
    data_to_save = {
        "metadata": metadata,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Experiment results saved in: {filepath}")
    return filepath

def load_real_tdlas_data(filepath, cfg: Config) -> SimulationData:
    """
    Extracts real simulation data from a given file. It also updates config map size properly.
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
            
            # 2D proyection
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
    
    grid_w = math.ceil(raw_w / cfg.sim.cell_size)
    grid_h = math.ceil(raw_h / cfg.sim.cell_size)
    
    # Update new map size in cfg
    map_w = grid_w * cfg.sim.cell_size
    map_h = grid_h * cfg.sim.cell_size
    cfg.sim.map_size = (map_w, map_h)

    # Empty ground truth
    img_gt = np.zeros((grid_h, grid_w))

    # Build simulation data
    return SimulationData(
        beams=beams_tensor,
        img_gt=img_gt,
        measurements=measurements_tensor,
        y_true=measurements_tensor
    )

def build_custom_real_scenario(
    cfg: Config, 
    real_data_path: str, 
    use_real_geometry: bool = True, 
    inject_simulated_gas: bool = False
) -> SimulationData:
    
    if not use_real_geometry and not inject_simulated_gas:
        raise ValueError(
            "Invalid combination: Synthetic beams require a simulated gas map to compute meaningful measurements."
        )

    # Load real data
    if os.path.exists(real_data_path):
        sim_data = load_real_tdlas_data(real_data_path, cfg)
    else:
        raise FileNotFoundError(f"Real data path is missing or invalid: {real_data_path}")

    if not use_real_geometry:
        num_random_beams = cfg.sim.num_beams // 2
        num_radial_beams = cfg.sim.num_beams - num_random_beams 
        beams = []
        beams += generate_random_beams(cfg.sim.map_size, num_random_beams)
        beams += generate_radial_beams(cfg.sim.map_size, num_radial_beams)
        
        sim_data.beams = torch.tensor(beams, dtype=torch.float32, device=cfg.device)

    if inject_simulated_gas:
        grid_h, grid_w = sim_data.img_gt.shape
        gas_map = np.zeros((grid_h, grid_w))

        source1 = (5.0, 7.0)
        source2 = (11.0, 4.0)

        s1r, s1c = xy2cell(source1, cfg.sim.cell_size)
        s2r, s2c = xy2cell(source2, cfg.sim.cell_size)

        if 0 <= s1r < grid_h and 0 <= s1c < grid_w:
            gas_map[s1r][s1c] = 60.0
        if 0 <= s2r < grid_h and 0 <= s2c < grid_w:
            gas_map[s2r][s2c] = 60.0

        gas_map = gaussian_filter(gas_map, sigma=1.5)
        
        # Recompute measurements
        measurements = simulate_gas_integrals(gas_map, sim_data.beams.tolist(), cfg.sim.cell_size)
        
        sim_data.img_gt = gas_map
        sim_data.measurements = torch.tensor(measurements, dtype=torch.float32, device=cfg.device)
        sim_data.y_true = sim_data.measurements

    return sim_data