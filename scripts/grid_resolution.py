import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.tomo_utils as tm
from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import render_gaussian_map

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    # --- Configuration ---
    resolutions = [20, 30, 40, 60, 80]
    seeds = [42, 100, 1234]
    
    methods = ["ART", "Tikhonov", "LFD", "LTD", "Gas Splatting"]
    
    results_rmse = {m: {r: [] for r in resolutions} for m in methods}
    results_time = {m: {r: [] for r in resolutions} for m in methods}

    parser = ArgumentParser(description="Compare methods results when grid resolution grows")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg = args.cfg

    # Deactivate tomo methods progress bar
    tm.tqdm = lambda x, **kwargs: x

    print(f"Starting experiment: {len(resolutions)} resolutions x {len(seeds)} seeds.")

    # --- Main loop ---
    for res in resolutions:
        print(f"\n=========================================")
        print(f" Evaluating Grid: {res}x{res}")
        print(f"=========================================")

        for seed in seeds:
            print(f" -> Seed: {seed}...", flush=True)
            
            # Configure seed and resolution
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = seed
            cfg.sim.grid_res = res
            
            # Simulation data
            sim_data = generate_simulation_data(cfg)
            measurements = sim_data.measurements.cpu().numpy()
            gt_img = sim_data.img_gt
            cell_size = cfg.sim.map_size / res
            
            # -----------------------------------------------------------
            #        TRADITIONAL METHODS SETUP
            # -----------------------------------------------------------
            t_setup_start = time.time()
            system_matrix = create_system_matrix_sparse((res, res), sim_data.beams.tolist(), cell_size).tocsr()
            trad_setup_time = time.time() - t_setup_start
            
            # --- ART ---
            t0 = time.time()
            art_res = tm.art(system_matrix, measurements, num_iterations=500, relaxation_factor=0.1)
            results_time["ART"][res].append(trad_setup_time + (time.time() - t0))
            results_rmse["ART"][res].append(rmse_loss(gt_img, art_res.reshape((res, res))))
            
            # --- Tikhonov (Direct) ---
            t0 = time.time()
            tik_res = tm.tikhonov_direct(system_matrix, measurements, alpha=0.1)
            results_time["Tikhonov"][res].append(trad_setup_time + (time.time() - t0))
            results_rmse["Tikhonov"][res].append(rmse_loss(gt_img, tik_res.reshape((res, res))))
            
            # --- LFD ---
            t0 = time.time()
            lfd_res = tm.lfd(system_matrix, measurements, grid_size=(res, res), alpha=0.05)
            results_time["LFD"][res].append(trad_setup_time + (time.time() - t0))
            results_rmse["LFD"][res].append(rmse_loss(gt_img, lfd_res))
            
            # --- LTD ---
            t0 = time.time()
            ltd_res = tm.ltd(system_matrix, measurements, grid_size=(res, res), alpha=0.01)
            results_time["LTD"][res].append(trad_setup_time + (time.time() - t0))
            results_rmse["LTD"][res].append(rmse_loss(gt_img, ltd_res))

            # -----------------------------------------------------------
            #        GAS SPLATTING
            # -----------------------------------------------------------
            # Setup (LSQR Initialization + Model)
            t_gs_setup = time.time()
            model, _, _ = setup_gs_model(sim_data, cfg)
            gs_setup_time = time.time() - t_gs_setup
            
            # Training
            t_gs_train = time.time()
            trainer = Trainer(model, cfg)
            trainer.train(sim_data)
            gs_train_time = time.time() - t_gs_train
            
            results_time["Gas Splatting"][res].append(gs_setup_time + gs_train_time)
            
            # Evaluation
            gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
            results_rmse["Gas Splatting"][res].append(rmse_loss(gt_img, gs_img))

    # --- Graphs ---
    print("\nGenerating graphs...")
    plot_results(resolutions, methods, results_rmse, results_time)

def plot_results(resolutions, methods, results_rmse, results_time):
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    styles = {
        "ART":           {"color": "tab:orange", "marker": "o"},
        "Tikhonov":      {"color": "tab:green",  "marker": "s"},
        "LFD":           {"color": "tab:red",    "marker": "^"},
        "LTD":           {"color": "tab:purple", "marker": "v"},
        "Gas Splatting": {"color": "tab:blue",   "marker": "*", "linewidth": 3, "markersize": 12}
    }

    for method in methods:
        style = styles[method]
        
        # Means and std deviations for nmse
        rmse_means = [np.mean(results_rmse[method][r]) for r in resolutions]
        rmse_stds  = [np.std(results_rmse[method][r]) for r in resolutions]
        
        # Means and std deviations for time
        time_means = [np.mean(results_time[method][r]) for r in resolutions]
        time_stds  = [np.std(results_time[method][r]) for r in resolutions]

        # Error
        ax1.plot(resolutions, rmse_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax1.fill_between(resolutions, 
                         np.array(rmse_means) - np.array(rmse_stds), 
                         np.array(rmse_means) + np.array(rmse_stds), 
                         color=style["color"], alpha=0.15)

        # Time
        ax2.plot(resolutions, time_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax2.fill_between(resolutions, 
                         np.array(time_means) - np.array(time_stds), 
                         np.array(time_means) + np.array(time_stds), 
                         color=style["color"], alpha=0.15)

    # Error graph details
    ax1.set_title("RMSE Evolution", pad=15)
    ax1.set_xlabel("Resolution (NxN)")
    ax1.set_ylabel("RMSE vs Ground Truth")
    ax1.set_xticks(resolutions)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Time graph details
    ax2.set_title("Time Evolution", pad=15)
    ax2.set_xlabel("Resolution (NxN)")
    ax2.set_ylabel("Total Time (seconds)")
    ax2.set_xticks(resolutions)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'grid_resolution.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()