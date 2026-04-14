import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.tomo_utils as tm
from config import Config
from trainer import Trainer
from utils.sim_utils import SimulationData
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import render_gaussian_map

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    # --- Configuration ---
    # Number of beams to test (from very sparse to dense)
    num_beams_list = [10, 20, 30, 40, 50, 60] 
    seeds = [42, 100, 1234, 777, 999]
    
    methods = ["ART", "Tikhonov", "LFD", "LTD", "Gas Splatting"]
    
    results_rmse = {m: {b: [] for b in num_beams_list} for m in methods}
    results_time = {m: {b: [] for b in num_beams_list} for m in methods}

    parser = ArgumentParser(description="Compare methods results when the number of beams changes")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg = args.cfg
    cfg.train.no_live_vis = True

    # We generate all beams and then we select the amount of beams we want to test
    cfg.sim.num_beams = num_beams_list[-1]

    # Deactivate tomo methods progress bar
    tm.tqdm = lambda x, **kwargs: x

    print(f"Starting experiment: {len(num_beams_list)} beam configurations x {len(seeds)} seeds.")

    # --- Main loop ---
    for seed in seeds:
        print(f" -> Seed: {seed}...", flush=True)
        cfg.sim.seed = seed
        
        # Simulation data only changes when using other seed
        # so we only generate it once for each seed
        base_sim_data = generate_simulation_data(cfg)
        gt_img = base_sim_data.img_gt
        
        for n_beams in num_beams_list:
            print(f"\n=========================================")
            print(f" Evaluating {n_beams} Beams")
            print(f"=========================================")  

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

            # -----------------------------------------------------------
            #        TRADITIONAL METHODS
            # -----------------------------------------------------------
            t_setup_start = time.time()
            system_matrix = create_system_matrix_sparse(grid_size, sim_data.beams.tolist(), cfg.sim.cell_size).tocsr()
            trad_setup_time = time.time() - t_setup_start
            
            # --- ART ---
            t0 = time.time()
            art_res = tm.art(system_matrix, measurements, num_iterations=400, relaxation_factor=1.6)
            results_time["ART"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_rmse["ART"][n_beams].append(rmse_loss(gt_img, art_res.reshape(grid_size)))
            
            # --- Tikhonov (Direct) ---
            t0 = time.time()
            tik_res = tm.tikhonov_direct(system_matrix, measurements, alpha=0.2)
            results_time["Tikhonov"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_rmse["Tikhonov"][n_beams].append(rmse_loss(gt_img, tik_res.reshape(grid_size)))
            
            # --- LFD ---
            t0 = time.time()
            lfd_res = tm.lfd(system_matrix, measurements, grid_size=grid_size, alpha=0.07)
            results_time["LFD"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_rmse["LFD"][n_beams].append(rmse_loss(gt_img, lfd_res))
            
            # --- LTD ---
            t0 = time.time()
            ltd_res = tm.ltd(system_matrix, measurements, grid_size=grid_size, alpha=5.0)
            results_time["LTD"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_rmse["LTD"][n_beams].append(rmse_loss(gt_img, ltd_res))

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
            trainer.train(sim_data.beams, sim_data.measurements)
            gs_train_time = time.time() - t_gs_train
            
            results_time["Gas Splatting"][n_beams].append(gs_setup_time + gs_train_time)
            
            # Evaluation
            gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
            results_rmse["Gas Splatting"][n_beams].append(rmse_loss(gt_img, gs_img))

    # --- Plots ---
    print("\nGenerating plots...")
    plot_results(num_beams_list, methods, results_rmse, results_time)

def plot_results(num_beams_list, methods, results_rmse, results_time):
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
        
        # Means and std deviations for rmse
        rmse_means = [np.mean(results_rmse[method][b]) for b in num_beams_list]
        rmse_stds  = [np.std(results_rmse[method][b]) for b in num_beams_list]
        
        # Means and std deviations for time
        time_means = [np.mean(results_time[method][b]) for b in num_beams_list]
        time_stds  = [np.std(results_time[method][b]) for b in num_beams_list]

        # Error
        ax1.plot(num_beams_list, rmse_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax1.fill_between(num_beams_list, 
                         np.array(rmse_means) - np.array(rmse_stds), 
                         np.array(rmse_means) + np.array(rmse_stds), 
                         color=style["color"], alpha=0.15)

        # Time
        ax2.plot(num_beams_list, time_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax2.fill_between(num_beams_list, 
                         np.array(time_means) - np.array(time_stds), 
                         np.array(time_means) + np.array(time_stds), 
                         color=style["color"], alpha=0.15)

    # Error graph details
    ax1.set_title("RMSE Evolution", pad=15)
    ax1.set_xlabel("Number of Beams")
    ax1.set_ylabel("RMSE vs Ground Truth")
    ax1.set_xticks(num_beams_list)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Time graph details
    ax2.set_title("Time Evolution", pad=15)
    ax2.set_xlabel("Number of Beams")
    ax2.set_ylabel("Total Time (seconds)")
    ax2.set_xticks(num_beams_list)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'num_beams_experiment.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()