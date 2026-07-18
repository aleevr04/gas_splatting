import os
import sys
import time
import copy
import torch
import numpy as np
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.tomo_utils as tm
from config import ExperimentConfig
from utils.sim_utils import MeasurementBatch, generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import plot_experiment_evolution
from utils.methods_registry import AVAILABLE_METHODS
from utils.experiment_utils import (
    setup_worker_env, 
    warmup_worker, 
    calculate_metrics, 
    merge_local_results,
    yield_parallel_experiment, 
    save_experiment_results
)

def evaluate_single_seed(seed, base_cfg, num_beams_list, methods):
    """
    Isolated function that evaluates a single seed. 
    It runs in an independent worker process.
    """
    setup_worker_env()

    cfg = copy.deepcopy(base_cfg)
    cfg.sim.seed = seed
    warmup_worker(cfg, seed)

    print(f"[Worker Process] -> Starting Seed: {seed}", flush=True)
    
    local_rmse = {m: {b: 0.0 for b in num_beams_list} for m in methods}
    local_ssim = {m: {b: 0.0 for b in num_beams_list} for m in methods}
    local_time = {m: {b: 0.0 for b in num_beams_list} for m in methods}

    # We generate simulation data ONCE, with maximum number of beams
    cfg.sim.num_beams = max(num_beams_list)
    base_sim_data = generate_simulation_data(cfg)
    gt_img = base_sim_data.ground_truth

    for n_beams in num_beams_list:
        # Select only the number of beams needed for this iteration
        batch = MeasurementBatch(
            beams=base_sim_data.batch.beams[:n_beams],
            measurements=base_sim_data.batch.measurements[:n_beams]
        )

        grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
        grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)

        matrix_setup_start = time.time()
        system_matrix = create_system_matrix_sparse((grid_w, grid_h), batch.beams.tolist(), cfg.sim.cell_size).tocsr()
        matrix_setup_time = time.time() - matrix_setup_start
        
        for method_name in methods:
            func = AVAILABLE_METHODS[method_name]["func"]
            
            res_img, total_time = func(
                batch=batch, 
                cfg=cfg, 
                ground_truth=gt_img,
                system_matrix=system_matrix,
                matrix_setup_time=matrix_setup_time
            )
            
            rmse, ssim_val = calculate_metrics(gt_img, res_img)
            local_time[method_name][n_beams] = total_time
            local_rmse[method_name][n_beams] = rmse
            local_ssim[method_name][n_beams] = ssim_val

    print(f"[Worker Process] -> Completed Seed: {seed}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    local_results = {
        "rmse": local_rmse,
        "ssim": local_ssim,
        "time": local_time
    }

    return seed, local_results


def main():
    parser = ArgumentParser(description="Compare methods results when the number of beams changes")
    parser.add_arguments(ExperimentConfig, dest="cfg")
    parser.add_argument(
        "--num_beams_list", dest="num_beams_list", nargs="+", type=int, default=[10, 20, 30, 40, 50, 60],
        help="List of number of beams to test"
    )
    args = parser.parse_args()
    cfg: ExperimentConfig = args.cfg

    num_beams_list = args.num_beams_list
    seeds = np.random.randint(0, 100000, size=cfg.num_seeds).tolist()
    methods = list(AVAILABLE_METHODS.keys())

    # Deactivate tomo methods progress bar and live evaluations
    tm.tqdm = lambda x, **kwargs: x
    cfg.train.do_eval = False
    cfg.train.live_vis = False

    print(f"Starting experiment: {len(num_beams_list)} beam configurations x {len(seeds)} seeds.")

    results = {
        "rmse": {m: {b: [] for b in num_beams_list} for m in methods},
        "ssim": {m: {b: [] for b in num_beams_list} for m in methods},
        "time": {m: {b: [] for b in num_beams_list} for m in methods}
    }

    # --- Shared execution loop ---
    for seed, local_results in yield_parallel_experiment(
        worker_func=evaluate_single_seed,
        seeds=seeds,
        max_workers=4,
        base_cfg=cfg,
        num_beams_list=num_beams_list,
        methods=methods
    ):
        merge_local_results(results, local_results)

    # --- Save results ---
    save_experiment_results({
        "experiment_name": "num_beams",
        "num_beams_list": num_beams_list,
        "seeds": seeds,
        "map_size": cfg.sim.map_size,
        "cell_size": cfg.sim.cell_size
    }, results)

    # --- Plot results ---
    print("\nGenerating plots...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'num_beams_experiment.png')

    plot_experiment_evolution(
        x_values=num_beams_list,
        x_label="Number of beams",
        methods_info=AVAILABLE_METHODS,
        results_rmse=results["rmse"],
        results_ssim=results["ssim"],
        results_time=results["time"],
        save_path=save_path
    )

if __name__ == "__main__":
    main()