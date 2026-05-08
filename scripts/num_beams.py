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
from config import Config
from utils.sim_utils import SimulationData
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import plot_experiment_evolution
from utils.data_utils import save_experiment_results
from utils.methods_registry import AVAILABLE_METHODS

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def evaluate_single_seed(seed, base_cfg, num_beams_list, methods):
    """
    Isolated function that evaluates a single seed. 
    It runs in an independent worker process.
    """
    # CRITICAL! Prevent PyTorch/NumPy from crashing the CPU by spawning 
    # too many internal threads when running multiple parallel processes.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"[Worker Process] -> Starting Seed: {seed}", flush=True)
    
    # Work with a local copy of the configuration to avoid cross-process contamination
    cfg = copy.deepcopy(base_cfg)
    cfg.sim.seed = seed
    
    # Local data structures to store the results exclusively for THIS seed
    local_rmse = {m: {b: 0.0 for b in num_beams_list} for m in methods}
    local_ssim = {m: {b: 0.0 for b in num_beams_list} for m in methods}
    local_time = {m: {b: 0.0 for b in num_beams_list} for m in methods}

    base_sim_data = generate_simulation_data(cfg)
    gt_img = base_sim_data.img_gt
    
    for n_beams in num_beams_list:
        sim_data = SimulationData(
            beams=base_sim_data.beams[:n_beams],
            measurements=base_sim_data.measurements[:n_beams],
            y_true=base_sim_data.y_true[:n_beams],
            img_gt=gt_img
        )
        measurements = sim_data.measurements.cpu().numpy()

        grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
        grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
        grid_size = (grid_w, grid_h)

        matrix_setup_start = time.time()
        system_matrix = create_system_matrix_sparse(grid_size, sim_data.beams.tolist(), cfg.sim.cell_size).tocsr()
        matrix_setup_time = time.time() - matrix_setup_start
        
        for method_name in methods:
            func = AVAILABLE_METHODS[method_name]["func"]
            
            res_img, total_time = func(
                system_matrix=system_matrix, 
                measurements=measurements, 
                sim_data=sim_data, 
                cfg=cfg, 
                setup_time=matrix_setup_time
            )
            
            local_time[method_name][n_beams] = total_time
            local_rmse[method_name][n_beams] = rmse_loss(gt_img, res_img)

            data_range = gt_img.max() - gt_img.min()
            local_ssim[method_name][n_beams] = ssim(gt_img, res_img, data_range=data_range)

    print(f"[Worker Process] -> Completed Seed: {seed}", flush=True)
    return seed, local_rmse, local_ssim, local_time


def main():
    # --- Configuration ---
    num_beams_list = [10, 20, 30, 40, 50, 60] 
    seeds = [42, 100, 1234, 777, 999]

    methods = list(AVAILABLE_METHODS.keys())
    
    # Global structures to collect the outputs from the parallel processes
    results_rmse = {m: {b: [] for b in num_beams_list} for m in methods}
    results_ssim = {m: {b: [] for b in num_beams_list} for m in methods}
    results_time = {m: {b: [] for b in num_beams_list} for m in methods}

    parser = ArgumentParser(description="Compare methods results when the number of beams changes")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg = args.cfg

    cfg.sim.num_beams = num_beams_list[-1]

    tm.tqdm = lambda x, **kwargs: x

    print(f"Starting experiment: {len(num_beams_list)} beam configurations x {len(seeds)} seeds.")

    # --- Parallelization ---
    # max_workers=None defaults to the number of physical CPU cores available
    with ProcessPoolExecutor() as executor:
        # Dispatch all seeds to the process pool
        futures = {
            executor.submit(evaluate_single_seed, seed, cfg, num_beams_list, methods): seed 
            for seed in seeds
        }
        
        # As each process finishes, append its local data into the global dictionaries
        for future in as_completed(futures):
            seed, local_rmse, local_ssim, local_time = future.result()
            
            for m in methods:
                for b in num_beams_list:
                    results_rmse[m][b].append(local_rmse[m][b])
                    results_ssim[m][b].append(local_ssim[m][b])
                    results_time[m][b].append(local_time[m][b])

    # --- Save results ---
    metadata = {
        "experiment_name": "num_beams_evolution",
        "num_beams_list": num_beams_list,
        "seeds": seeds,
        "map_size": cfg.sim.map_size,
        "cell_size": cfg.sim.cell_size
    }

    all_results = {
        "rmse": results_rmse,
        "ssim": results_ssim,
        "time": results_time
    }

    save_experiment_results(metadata, all_results)

    # --- Plot results ---
    print("\nGenerating plots...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'num_beams_experiment.png')

    plot_experiment_evolution(
        x_values=num_beams_list,
        x_label="Number of Beams",
        methods_info=AVAILABLE_METHODS,
        results_rmse=results_rmse,
        results_ssim=results_ssim,
        results_time=results_time,
        save_path=save_path
    )

if __name__ == "__main__":
    main()