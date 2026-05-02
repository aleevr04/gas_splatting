import os
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data, SimulationData

def rmse_loss(y_true, y_pred):
    mse = torch.mean((y_pred - y_true)**2)
    return torch.sqrt(mse).item()

def main():
    # --- Experiment settings ---
    iteration_list = [500, 1000, 1500, 2000, 3000]
    seeds = [42, 100, 1234, 777, 999]#, 123, 333, 10, 555, 22]

    # Parse configuration
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg = args.cfg

    # Generate beams for training and validating the model
    training_beams = cfg.sim.num_beams
    validation_beams = int(training_beams * 0.5)

    cfg.sim.num_beams = training_beams + validation_beams

    # Dictionaries for training and validation errors
    training_error   = {it: [] for it in iteration_list}
    validation_error = {it: [] for it in iteration_list}

    print(f"Starting experiment: {len(iteration_list)} diferent iterations x {len(seeds)} seeds.")

    # --- Main Loop ---
    for seed in seeds:
        print(f" -> Seed: {seed}...", flush=True) 
        cfg.seed = seed
        
        # Full simulation data
        sim_data = generate_simulation_data(cfg)

        beams_training = sim_data.beams[:training_beams]
        y_true_training = sim_data.measurements[:training_beams]
        
        beams_validation = sim_data.beams[training_beams:]
        y_true_validation = sim_data.measurements[training_beams:]            

        # Training simulation data
        sim_data_train = SimulationData(
            beams=beams_training,
            measurements=y_true_training,
            y_true=sim_data.y_true[:training_beams],
            img_gt=sim_data.img_gt
        )

        for iters in iteration_list:
            print(f" -> Testing {iters} Iterations...", flush=True)
            cfg.train.iterations = iters
            
            # Adapt densify parameters according to current number of iterations
            cfg.densify.densify_from = int(iters * (200 / 3000))
            cfg.densify.densify_until = int(iters * (1500 / 3000))
            cfg.densify.densify_interval = int(iters * (100 / 3000))

            # Initialization
            model, _, _ = setup_gs_model(sim_data_train, cfg)
            
            # Train
            trainer = Trainer(model, cfg)
            trainer.train(sim_data_train)

            # Evaluation
            with torch.no_grad():
                y_pred_training = model(beams_training)
                y_pred_validation = model(beams_validation)

            training_error[iters].append(rmse_loss(y_true_training, y_pred_training))
            validation_error[iters].append(rmse_loss(y_true_validation, y_pred_validation))

    # --- Visualization ---
    print("\nGenerating plot...")
    plot_results(iteration_list, training_error, validation_error)


def plot_results(iteration_list, training_error, validation_error):
    # Calculate means
    training_means = [np.mean(training_error[it]) for it in iteration_list]
    validation_means = [np.mean(validation_error[it]) for it in iteration_list]
    
    # Calculate standard deviations
    training_stds = [np.std(training_error[it]) for it in iteration_list]
    validation_stds = [np.std(validation_error[it]) for it in iteration_list]
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw Training Error (Mean + Std)
    ax.plot(iteration_list, training_means, 'g-o', linewidth=2, label='Training Error Mean')
    ax.fill_between(iteration_list, 
                    np.array(training_means) - np.array(training_stds), 
                    np.array(training_means) + np.array(training_stds), 
                    color='green', alpha=0.15, label='Training Error Std')

    # Draw Validation Error (Mean + Std)
    ax.plot(iteration_list, validation_means, 'b-o', linewidth=2, label='Validation Error Mean')
    ax.fill_between(iteration_list, 
                    np.array(validation_means) - np.array(validation_stds), 
                    np.array(validation_means) + np.array(validation_stds), 
                    color='blue', alpha=0.15, label='Validation Error Std')

    ax.set_title("Overfitting Test\n")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Root Mean Square Error (RMSE)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Move legend to a place where it doesn't overlap the lines easily
    ax.legend(loc='upper right', fontsize=10)

    # Logarithmic scale if error's variance is high
    # ax.set_yscale('log') 

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'overfitting.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()