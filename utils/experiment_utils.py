import os
import copy
import json
import datetime
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.sim_utils import generate_simulation_data
from methods.tomography import create_system_matrix_sparse
from methods.registry import run_gas_splatting

def save_experiment_results(metadata, results, folder="results"):
    """
    Saves metadata and results data in a JSON file.
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
    
    print(f"[+] Experiment results saved in: {filepath}")
    return filepath

def setup_worker_env():
    """Prevents PyTorch/NumPy from crashing the CPU with too many internal threads."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def calculate_metrics(gt_img, res_img):
    """Returns RMSE and SSIM for a given estimation."""
    rmse = rmse_loss(gt_img, res_img)
    data_range = gt_img.max() - gt_img.min()
    
    # Prevent ssim from breaking if the map is completely flat
    data_range = data_range if data_range > 0 else 1.0
    
    ssim_val = ssim(gt_img, res_img, data_range=data_range)
    return rmse, ssim_val

def warmup_worker(cfg, seed):
    """Executes a tiny run to wake up JIT/PyTorch and prevent cold start penalties."""
    warmup_cfg = copy.deepcopy(cfg)
    warmup_res = 20
    warmup_cfg.env.cell_size = warmup_cfg.env.map_size[0] / warmup_res
    warmup_cfg.train.iterations = 5 
    
    warmup_batch, warmup_environment = generate_simulation_data(warmup_cfg)
    _ = create_system_matrix_sparse(
        (warmup_res, warmup_res), 
        warmup_batch.beams.tolist(),
        warmup_cfg.env.cell_size,
        quiet=True
    ).tocsr()

    run_gas_splatting(batch=warmup_batch, cfg=warmup_cfg, environment=warmup_environment)

def yield_parallel_experiment(worker_func, seeds, max_workers=4, **kwargs):
    """
    Agnostic parallel orchestrator that YIELDS results on the fly.
    The worker_func must accept 'seed' as its first argument, followed by **kwargs.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_func, seed, **kwargs): seed 
            for seed in seeds
        }
        
        for future in as_completed(futures):
            yield future.result()

def merge_local_results(results, local_results):
    """Merge per-seed local results into the shared experiment summary."""
    for metric, metric_data in local_results.items():
        for method, method_value in metric_data.items():
            if isinstance(method_value, dict):
                for x_value, value in method_value.items():
                    results[metric][method][x_value].append(value)
            else:
                results[metric][method].append(method_value)