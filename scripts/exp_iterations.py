import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.sim_utils import generate_simulation_data
from utils.plot_utils import render_gaussian_map

def nmse_loss(y_true, y_pred):
    return np.sum((y_pred - y_true)**2) / (np.sum(y_true**2) + 1e-8)

def main():
    # --- Experiment settings ---
    iteration_list = [500, 1000, 2000, 3000, 5000]
    seeds = [42, 100, 1234, 777, 999]

    parser = ArgumentParser()
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg = args.cfg
    base_cfg.train.no_live_vis = True

    # Dictionaries for error and time results
    results_nmse = {it: [] for it in iteration_list}
    results_time = {it: [] for it in iteration_list}

    print(f"Starting experiment: {len(iteration_list)} iterations x {len(seeds)} seeds.")

    # --- Main Loop ---
    for iters in iteration_list:
        print(f"\n=========================================")
        print(f" Evaluating {iters} Iterations")
        print(f"=========================================")

        for seed in seeds:
            print(f" -> Seed: {seed}...", end=" ", flush=True)
            
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = seed
            cfg.train.iterations = iters
            
            # Adapt densify parameters according to current number of iterations
            cfg.densify.densify_from = int(iters * (200 / 3000))
            cfg.densify.densify_until = int(iters * (1500 / 3000))
            cfg.densify.densify_interval = int(iters * (100 / 3000))

            # Simulation data
            sim_data = generate_simulation_data(cfg)
            gt_img = sim_data.img_gt
            grid_res = gt_img.shape[0]

            # Initialization
            init_pos, init_concentration, init_std, _ = lsqr_initialization(
                sim_data.beams.tolist(), sim_data.y_true, cfg.sim.map_size, 
                num_gaussians=cfg.init.initial_gaussians, coarse_res=cfg.init.coarse_res
            )
            
            model = GasSplattingModel(init_pos.shape[0], cfg).to(cfg.device)
            model.initialize_gaussians(init_pos.to(cfg.device), init_concentration.to(cfg.device), init_std)
            
            # Train
            t_start = time.time()
            trainer = Trainer(model, cfg)
            trainer.train(sim_data.beams, sim_data.y_true)
            train_time = time.time() - t_start

            # Evaluation
            gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, grid_res=grid_res)
            nmse = nmse_loss(gt_img, gs_img)

            results_nmse[iters].append(nmse)
            results_time[iters].append(train_time)
            
            print(f"NMSE: {nmse:.4f} | Time: {train_time:.1f}s")

    # --- Visualization ---
    print("\nGenerating plot...")
    plot_results(iteration_list, results_nmse, results_time)


def plot_results(iteration_list, results_nmse, results_time):
    nmse_means = [np.mean(results_nmse[it]) for it in iteration_list]
    nmse_stds = [np.std(results_nmse[it]) for it in iteration_list]
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw means
    ax.plot(iteration_list, nmse_means, 'b-o', linewidth=2, label='NMSE Mean')
    
    # Draw standard deviation
    ax.fill_between(iteration_list, 
                    np.array(nmse_means) - np.array(nmse_stds), 
                    np.array(nmse_means) + np.array(nmse_stds), 
                    color='blue', alpha=0.2, label='Desviación Estándar')

    ax.set_title("Error (NMSE) vs Number of Iterations")
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Normalized Mean Square Error (NMSE)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Logaritmic scale if error's variance is high
    # ax.set_yscale('log') 

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'iterations_experiment.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()