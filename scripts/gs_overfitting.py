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
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.sim_utils import generate_simulation_data

def nmse_loss(y_true, y_pred):
    error = torch.sum((y_pred - y_true)**2) / (torch.sum(y_true**2) + 1e-8)
    return error.item()

def main():
    # --- Experiment settings ---
    iteration_list = [500, 1000, 1500, 2000, 3000]
    seeds = [42, 100, 1234, 777, 999, 123, 333, 10, 555, 22]

    # Parse configuration
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg = args.cfg
    base_cfg.train.no_live_vis = True

    # Generate beams for training and validating the model
    training_beams = base_cfg.sim.num_beams
    validation_beams = int(training_beams * 0.5)

    base_cfg.sim.num_beams = training_beams + validation_beams

    # Dictionaries for training and validation errors
    training_error   = {it: [] for it in iteration_list}
    validation_error = {it: [] for it in iteration_list}

    print(f"Starting experiment: {len(iteration_list)} diferent iterations x {len(seeds)} seeds.")

    # --- Main Loop ---
    for iters in iteration_list:
        print(f"\n=========================================")
        print(f" Evaluating {iters} Iterations")
        print(f"{training_beams} Training Beams")
        print(f"{validation_beams} Validation Beams")
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

            y_true_training = sim_data.y_true[:training_beams]
            y_true_validation = sim_data.y_true[training_beams:]

            beams_training = sim_data.beams[:training_beams]
            beams_validation = sim_data.beams[training_beams:]            

            # Initialization
            init_pos, init_concentration, init_std, _ = lsqr_initialization(
                beams_training.tolist(), y_true_training, cfg.sim.map_size, 
                num_gaussians=cfg.init.initial_gaussians, coarse_res=cfg.init.coarse_res
            )
            
            model = GasSplattingModel(init_pos.shape[0], cfg).to(cfg.device)
            model.initialize_gaussians(init_pos.to(cfg.device), init_concentration.to(cfg.device), init_std)
            
            # Train
            trainer = Trainer(model, cfg)
            trainer.train(beams_training, y_true_training)

            # Evaluation
            with torch.no_grad():
                y_pred_training = model(beams_training)
                y_pred_validation = model(beams_validation)

            training_error[iters].append(nmse_loss(y_true_training, y_pred_training))
            validation_error[iters].append(nmse_loss(y_true_validation, y_pred_validation))

    # --- Visualization ---
    print("\nGenerating plot...")
    plot_results(iteration_list, training_error, validation_error)


def plot_results(iteration_list, training_error, validation_error):
    training_means = [np.mean(training_error[it]) for it in iteration_list]
    validation_means = [np.mean(validation_error[it]) for it in iteration_list]
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw means
    ax.plot(iteration_list, training_means, 'g-o', linewidth=2, label='Training Error')
    ax.plot(iteration_list, validation_means, 'b-o', linewidth=2, label='Validation Error')

    ax.set_title("Overfitting Test\n")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Normalized Mean Square Error (NMSE)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Logaritmic scale if error's variance is high
    # ax.set_yscale('log') 

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'overfitting.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()