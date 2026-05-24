import os
import sys
import json
import datetime
import math
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.sim_utils import SimulationData

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

def load_real_tdlas_data(filepath, cfg: Config):
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