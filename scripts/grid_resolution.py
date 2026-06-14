import os
import sys
import time
import copy
import torch
import numpy as np
from simple_parsing import ArgumentParser
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.tomo_utils as tm
from config import ExperimentConfig
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import plot_experiment_evolution
from utils.data_utils import save_experiment_results
from utils.methods_registry import AVAILABLE_METHODS

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def evaluate_single_seed(seed, base_cfg, resolutions, methods):
    """
    Isolated function that evaluates a single seed.
    It runs in an independent worker process.
    """
    # Prevent PyTorch/NumPy from crashing the CPU by spawning 
    # too many internal threads when running multiple parallel processes.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"[Worker Process] -> Starting Seed: {seed}", flush=True)
    
    # Work with a local copy of the configuration
    cfg = copy.deepcopy(base_cfg)
    cfg.sim.seed = seed

    # --- WARM-UP (Prevents Cold Start Penalty) ---
    # We do a tiny, untimed run to wake up PyTorch and SciPy memory allocators.
    # We also build a small sparse matrix to warm up the SciPy C-backend.
    print(f"[Worker Process] -> Warming up Seed: {seed}...", flush=True)
    _warmup_cfg = copy.deepcopy(cfg)
    _warmup_res = 20  # Use the smallest resolution for speed
    _warmup_cfg.sim.cell_size = _warmup_cfg.sim.map_size[0] / _warmup_res
    _warmup_cfg.train.iterations = 5  # Just 5 iterations
    
    _warmup_sim = generate_simulation_data(_warmup_cfg)
    _warmup_matrix = create_system_matrix_sparse(
        (_warmup_res, _warmup_res), 
        _warmup_sim.beams.tolist(), 
        _warmup_cfg.sim.cell_size
    ).tocsr()
    
    _func = AVAILABLE_METHODS["Gas Splatting"]["func"]
    _func(
        system_matrix=_warmup_matrix, 
        measurements=_warmup_sim.measurements.cpu().numpy(), 
        sim_data=_warmup_sim, 
        cfg=_warmup_cfg, 
        setup_time=0.0
    )
    # ----------------------------------------------------
    
    # Local data structures to store the results exclusively for THIS seed
    local_rmse = {m: {r: 0.0 for r in resolutions} for m in methods}
    local_ssim = {m: {r: 0.0 for r in resolutions} for m in methods}
    local_time = {m: {r: 0.0 for r in resolutions} for m in methods}

    for res in resolutions:
        # Update cell size based on resolution (assuming square map size)
        cfg.sim.cell_size = cfg.sim.map_size[0] / res
        
        # In this experiment, since the grid resolution changes, the dimensions 
        # of the Ground Truth image also change. We MUST regenerate the simulation.
        sim_data = generate_simulation_data(cfg)
        measurements = sim_data.measurements.cpu().numpy()
        gt_img = sim_data.img_gt

        grid_size = (res, res)

        matrix_setup_start = time.time()
        system_matrix = create_system_matrix_sparse(grid_size, sim_data.beams.tolist(), cfg.sim.cell_size).tocsr()
        matrix_setup_time = time.time() - matrix_setup_start
            
        for method_name in methods:
            method_info = AVAILABLE_METHODS[method_name]
            func = method_info["func"]
            
            res_img, total_time = func(
                system_matrix=system_matrix, 
                measurements=measurements, 
                sim_data=sim_data, 
                cfg=cfg, 
                setup_time=matrix_setup_time
            )

            local_time[method_name][res] = total_time
            local_rmse[method_name][res] = rmse_loss(gt_img, res_img)
            data_range = gt_img.max() - gt_img.min()
            local_ssim[method_name][res] = ssim(gt_img, res_img, data_range=data_range)

    print(f"[Worker Process] -> Completed Seed: {seed}", flush=True)
    return seed, local_rmse, local_ssim, local_time


def main():
    # --- Configuration ---
    resolutions = [20, 30, 40, 60, 80]

    # Pull methods directly from the central registry
    methods = list(AVAILABLE_METHODS.keys())
    
    # Global structures to collect the outputs from the parallel processes
    results_rmse = {m: {r: [] for r in resolutions} for m in methods}
    results_ssim = {m: {r: [] for r in resolutions} for m in methods}
    results_time = {m: {r: [] for r in resolutions} for m in methods}

    parser = ArgumentParser(description="Compare methods results when grid resolution grows")
    parser.add_arguments(ExperimentConfig, dest="cfg")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.cfg

    num_seeds = cfg.num_seeds
    seeds = np.random.randint(0, 100000, size=num_seeds).tolist()

    # Deactivate tomo methods progress bar
    tm.tqdm = lambda x, **kwargs: x

    print(f"Starting experiment: {len(resolutions)} resolutions x {len(seeds)} seeds.")

    # --- Parallelization ---
    max_workers=4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Dispatch all seeds to the process pool
        futures = {
            executor.submit(evaluate_single_seed, seed, cfg, resolutions, methods): seed 
            for seed in seeds
        }
        
        # As each process finishes, append its local data into the global dictionaries
        for future in as_completed(futures):
            seed, local_rmse, local_ssim, local_time = future.result()
            
            for m in methods:
                for r in resolutions:
                    results_rmse[m][r].append(local_rmse[m][r])
                    results_ssim[m][r].append(local_ssim[m][r])
                    results_time[m][r].append(local_time[m][r])

    # --- Save results ---
    metadata = {
        "experiment_name": "grid_resolution",
        "resolutions": resolutions,
        "seeds": seeds,
        "map_size": cfg.sim.map_size,
        "num_beams": cfg.sim.num_beams
    }

    all_results = {
        "rmse": results_rmse,
        "ssim": results_ssim,
        "time": results_time
    }

    save_experiment_results(metadata, all_results)

    # --- Plot results ---
    print("\nGenerating plots...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'grid_resolution_experiment.png')

    plot_experiment_evolution(
        x_values=resolutions,
        x_label="Resolution (MxM)",
        methods_info=AVAILABLE_METHODS,
        results_rmse=results_rmse,
        results_ssim=results_ssim,
        results_time=results_time,
        save_path=save_path
    )

if __name__ == "__main__":
    main()