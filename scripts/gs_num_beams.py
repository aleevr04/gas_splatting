import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.sim_utils import SimulationData
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data
from utils.plot_utils import render_gaussian_map

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    # --- Configuration ---
    # Number of beams to test (from very sparse to dense)
    num_beams_list = [10, 20, 30, 50, 70, 90] 
    seeds = [35, 58, 469, 851, 925, 9, 700, 349, 83, 891]
    
    results_rmse = {b: [] for b in num_beams_list}
    results_time = {b: [] for b in num_beams_list}

    parser = ArgumentParser(description="Evaluate Gas Splatting results when the number of beams changes")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg = args.cfg
    cfg.train.no_live_vis = True

    # We generate all beams and then we select the amount of beams we want to test
    cfg.sim.num_beams = num_beams_list[-1]

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
            
            results_time[n_beams].append(gs_setup_time + gs_train_time)
            
            # Evaluation
            gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
            results_rmse[n_beams].append(rmse_loss(gt_img, gs_img))

    # --- Plots ---
    print("\nGenerating plots...")
    plot_results(num_beams_list, results_rmse, results_time)

def plot_results(num_beams_list, results_rmse, results_time):
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Means and std deviations for rmse
    rmse_means = [np.mean(results_rmse[b]) for b in num_beams_list]
    rmse_stds  = [np.std(results_rmse[b]) for b in num_beams_list]
    
    # Means and std deviations for time
    time_means = [np.mean(results_time[b]) for b in num_beams_list]
    time_stds  = [np.std(results_time[b]) for b in num_beams_list]

    # Error Plot
    ax1.plot(num_beams_list, rmse_means, label="Gas Splatting", 
             color="tab:blue", marker="*", linewidth=3, markersize=12)
    ax1.fill_between(num_beams_list, 
                     np.array(rmse_means) - np.array(rmse_stds), 
                     np.array(rmse_means) + np.array(rmse_stds), 
                     color="tab:blue", alpha=0.15)

    # Time Plot
    ax2.plot(num_beams_list, time_means, label="Gas Splatting", 
             color="tab:blue", marker="*", linewidth=3, markersize=12)
    ax2.fill_between(num_beams_list, 
                     np.array(time_means) - np.array(time_stds), 
                     np.array(time_means) + np.array(time_stds), 
                     color="tab:blue", alpha=0.15)

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
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'gs_only_num_beams_experiment.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()