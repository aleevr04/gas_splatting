import os
import sys
import time
import copy
import torch
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ExperimentConfig
from utils.sim_utils import generate_simulation_data
from methods.tomography import create_system_matrix_sparse
from utils.plot_utils import plot_experiment_evolution
from methods.registry import AVAILABLE_METHODS
from utils.experiment_utils import (
    setup_worker_env, 
    warmup_worker, 
    calculate_metrics, 
    merge_local_results,
    yield_parallel_experiment, 
    save_experiment_results
)

def evaluate_single_seed(seed, base_cfg, resolutions, methods):
    """
    Isolated function that evaluates a single seed.
    It runs in an independent worker process.
    """
    setup_worker_env()
    
    cfg = copy.deepcopy(base_cfg)
    cfg.sim.seed = seed
    warmup_worker(cfg, seed)
    
    # Local data structures to store the results exclusively for THIS seed
    local_rmse = {m: {r: 0.0 for r in resolutions} for m in methods}
    local_ssim = {m: {r: 0.0 for r in resolutions} for m in methods}
    local_time = {m: {r: 0.0 for r in resolutions} for m in methods}

    for res in resolutions:
        # Update cell size based on resolution (assuming square map size)
        cfg.env.cell_size = cfg.env.map_size[0] / res
        
        # In this experiment, since the grid resolution changes, the dimensions 
        # of the Ground Truth image also change. We MUST regenerate the simulation.
        batch, environment = generate_simulation_data(cfg)
        gt_img = environment.ground_truth.gas_map

        grid_size = (res, res)

        matrix_setup_start = time.time()
        system_matrix = create_system_matrix_sparse(grid_size, batch.beams.tolist(), cfg.env.cell_size, quiet=True).tocsr()
        matrix_setup_time = time.time() - matrix_setup_start
            
        for method_name in methods:
            func = AVAILABLE_METHODS[method_name]["func"]
            
            res_img, total_time = func(
                batch=batch,
                cfg=cfg,
                environment=environment,
                system_matrix=system_matrix, 
                matrix_setup_time=matrix_setup_time
            )

            rmse, ssim_val = calculate_metrics(gt_img, res_img)
            local_time[method_name][res] = total_time
            local_rmse[method_name][res] = rmse
            local_ssim[method_name][res] = ssim_val

    # Prevent GPU OOM errors in long multi-seed pools
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    local_results = {
        "rmse": local_rmse,
        "ssim": local_ssim,
        "time": local_time
    }

    return seed, local_results


def main():
    parser = ArgumentParser(description="Compare methods results when grid resolution grows")
    parser.add_arguments(ExperimentConfig, dest="cfg")
    parser.add_argument(
        "--resolutions", dest="resolutions", nargs="+", type=int, default=[20, 30, 40, 60, 80],
        help="List of resolutions (MxM) to test"
    )
    args = parser.parse_args()
    cfg: ExperimentConfig = args.cfg

    resolutions = args.resolutions
    seeds = np.random.randint(0, 100000, size=cfg.num_seeds).tolist()
    methods = list(AVAILABLE_METHODS.keys())

    # Deactivate training evaluation, live visualization and enable quiet mode
    cfg.train.do_eval = False
    cfg.train.live_vis = False
    cfg.quiet = True

    results = {
        "rmse": {m: {r: [] for r in resolutions} for m in methods},
        "ssim": {m: {r: [] for r in resolutions} for m in methods},
        "time": {m: {r: [] for r in resolutions} for m in methods}
    }

    print(f"Starting experiment: {len(resolutions)} resolutions x {len(seeds)} seeds.")
    
    global_pbar = tqdm(total=len(seeds), desc="Experiment Progress (Seeds)", dynamic_ncols=True)

    # --- Shared execution loop ---
    for seed, local_res in yield_parallel_experiment(
        worker_func=evaluate_single_seed, 
        seeds=seeds, 
        max_workers=4, 
        base_cfg=cfg, 
        resolutions=resolutions, 
        methods=methods
    ):
        merge_local_results(results, local_res)
        global_pbar.update(1)
        global_pbar.set_postfix({"Last completed": seed})
        
    global_pbar.close()

    # --- Save results ---
    save_experiment_results({
        "experiment_name": "grid_resolution",
        "resolutions": resolutions,
        "seeds": seeds,
        "map_size": cfg.env.map_size,
        "num_beams": cfg.sim.num_beams
    }, results)

    # --- Plot results ---
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'grid_resolution_experiment.png')

    plot_experiment_evolution(
        x_values=resolutions,
        x_label="Resolution (MxM)",
        methods_info=AVAILABLE_METHODS,
        results_rmse=results["rmse"],
        results_ssim=results["ssim"],
        results_time=results["time"],
        save_path=save_path
    )

if __name__ == "__main__":
    main()