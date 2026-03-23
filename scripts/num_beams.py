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
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import render_gaussian_map

def nmse_loss(measurements, y_pred):
    return np.sum((y_pred - measurements)**2) / (np.sum(measurements**2) + 1e-8)

def main():
    # --- Configuration ---
    # Number of beams to test (from very sparse to dense)
    num_beams_list = [10, 20, 30, 40, 50, 60] 
    seeds = [42, 100, 1234, 777, 999]
    
    methods = ["ART", "Tikhonov", "LFD", "LTD", "Gas Splatting"]
    
    results_nmse = {m: {b: [] for b in num_beams_list} for m in methods}
    results_time = {m: {b: [] for b in num_beams_list} for m in methods}

    parser = ArgumentParser(description="Compare methods results when the number of beams changes")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg = args.cfg
    base_cfg.train.no_live_vis = True

    # Deactivate tomo methods progress bar
    tm.tqdm = lambda x, **kwargs: x

    print(f"Starting experiment: {len(num_beams_list)} beam configurations x {len(seeds)} seeds.")

    # --- Main loop ---
    for n_beams in num_beams_list:
        print(f"\n=========================================")
        print(f" Evaluating {n_beams} Beams")
        print(f"=========================================")

        for seed in seeds:
            print(f" -> Seed: {seed}...", flush=True)
            
            # Configure seed and number of beams
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = seed
            cfg.sim.num_beams = n_beams
            res = cfg.sim.grid_res # Grid resolution stays constant
            
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
            results_time["ART"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_nmse["ART"][n_beams].append(nmse_loss(gt_img, art_res.reshape((res, res))))
            
            # --- Tikhonov (Direct) ---
            t0 = time.time()
            tik_res = tm.tikhonov_direct(system_matrix, measurements, alpha=0.1)
            results_time["Tikhonov"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_nmse["Tikhonov"][n_beams].append(nmse_loss(gt_img, tik_res.reshape((res, res))))
            
            # --- LFD ---
            t0 = time.time()
            lfd_res = tm.lfd(system_matrix, measurements, grid_size=(res, res), alpha=0.05)
            results_time["LFD"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_nmse["LFD"][n_beams].append(nmse_loss(gt_img, lfd_res))
            
            # --- LTD ---
            t0 = time.time()
            ltd_res = tm.ltd(system_matrix, measurements, grid_size=(res, res), alpha=0.01)
            results_time["LTD"][n_beams].append(trad_setup_time + (time.time() - t0))
            results_nmse["LTD"][n_beams].append(nmse_loss(gt_img, ltd_res))

            # -----------------------------------------------------------
            #        GAS SPLATTING
            # -----------------------------------------------------------
            # Setup (LSQR Initialization + Model)
            t_gs_setup = time.time()
            init_pos, init_concentration, init_std, _ = lsqr_initialization(
                sim_data.beams.tolist(), sim_data.measurements, cfg.sim.map_size, 
                num_gaussians=cfg.init.initial_gaussians, coarse_res=cfg.init.coarse_res
            )
            model = GasSplattingModel(init_pos.shape[0], cfg).to(cfg.device)
            model.initialize_gaussians(init_pos.to(cfg.device), init_concentration.to(cfg.device), init_std)
            gs_setup_time = time.time() - t_gs_setup
            
            # Training
            t_gs_train = time.time()
            trainer = Trainer(model, cfg)
            trainer.train(sim_data.beams, sim_data.measurements)
            gs_train_time = time.time() - t_gs_train
            
            results_time["Gas Splatting"][n_beams].append(gs_setup_time + gs_train_time)
            
            # Evaluation
            gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, grid_res=res)
            results_nmse["Gas Splatting"][n_beams].append(nmse_loss(gt_img, gs_img))

    # --- Graphs ---
    print("\nGenerating graphs...")
    plot_results(num_beams_list, methods, results_nmse, results_time)

def plot_results(num_beams_list, methods, results_nmse, results_time):
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
        nmse_means = [np.mean(results_nmse[method][b]) for b in num_beams_list]
        nmse_stds  = [np.std(results_nmse[method][b]) for b in num_beams_list]
        
        # Means and std deviations for time
        time_means = [np.mean(results_time[method][b]) for b in num_beams_list]
        time_stds  = [np.std(results_time[method][b]) for b in num_beams_list]

        # Error
        ax1.plot(num_beams_list, nmse_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax1.fill_between(num_beams_list, 
                         np.array(nmse_means) - np.array(nmse_stds), 
                         np.array(nmse_means) + np.array(nmse_stds), 
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
    ax1.set_title("NMSE Evolution", pad=15)
    ax1.set_xlabel("Number of Beams")
    ax1.set_ylabel("NMSE vs Ground Truth")
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