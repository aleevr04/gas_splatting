import json
import os
import datetime
import math
import torch

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

def load_real_tdlas_data(filepath, device="cpu"):
    beams_list = []
    measurements_list = []
    
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
            
    # Build simulation data
    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=device)
    measurements_tensor = torch.tensor(measurements_list, dtype=torch.float32, device=device)
    
    return SimulationData(
        beams=beams_tensor,
        img_gt=None, # There is no ground truth
        measurements=measurements_tensor,
        y_true=measurements_tensor
    )