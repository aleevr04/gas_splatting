import os
import sys
import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data
from utils.plot_utils import render_gaussian_map

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    # --- Experiment settings ---
    iteration_list = [500, 1000, 2000, 3000, 5000]
    seeds = [123, 555, 13, 69, 200]

    parser = ArgumentParser()
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg = args.cfg
    cfg.train.no_live_vis = True

    # Dictionaries for error and time results
    results_nmse = {it: [] for it in iteration_list}
    results_time = {it: [] for it in iteration_list}

    print(f"Starting experiment: {len(iteration_list)} diferent iterations x {len(seeds)} seeds.")

    # --- Main Loop ---
    for seed in seeds:
        print(f" -> Seed: {seed}...", flush=True)
        cfg.seed = seed
        
        # Simulation data (generated once for each seed)
        sim_data = generate_simulation_data(cfg)
        gt_img = sim_data.img_gt

        for iters in iteration_list:
            print(f"\n=========================================")
            print(f" Evaluating {iters} Iterations")
            print(f"=========================================")
            
            cfg.train.iterations = iters
            
            # Adapt densify parameters according to current number of iterations
            cfg.densify.densify_from = int(iters * (200 / 3000))
            cfg.densify.densify_until = int(iters * (1500 / 3000))
            cfg.densify.densify_interval = int(iters * (100 / 3000))

            # Initialization + Training
            t_start = time.time()
            model, _, _ = setup_gs_model(sim_data, cfg)
            trainer = Trainer(model, cfg)
            trainer.train(sim_data.beams, sim_data.measurements)
            train_time = time.time() - t_start

            # Evaluation
            gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
            rmse = rmse_loss(gt_img, gs_img)

            results_nmse[iters].append(rmse)
            results_time[iters].append(train_time)
            
            print(f"RMSE: {rmse:.4f} | Time: {train_time:.1f}s")

    # --- Visualization ---
    print("\nGenerating plot...")
    plot_results(iteration_list, results_nmse, results_time)


def plot_results(iteration_list, results_nmse, results_time):
    nmse_means = [np.mean(results_nmse[it]) for it in iteration_list]
    nmse_stds = [np.std(results_nmse[it]) for it in iteration_list]
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw means
    ax.plot(iteration_list, nmse_means, 'b-o', linewidth=2, label='RMSE Mean')
    
    # Draw standard deviation
    ax.fill_between(iteration_list, 
                    np.array(nmse_means) - np.array(nmse_stds), 
                    np.array(nmse_means) + np.array(nmse_stds), 
                    color='blue', alpha=0.2, label='Standard Deviation')

    ax.set_title("Error (RMSE) vs Number of Iterations")
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Root Mean Square Error (RMSE)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Logaritmic scale if error's variance is high
    ax.set_yscale('log') 

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'iterations_experiment.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()