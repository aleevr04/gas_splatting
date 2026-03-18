import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.tomo_utils as tm
from config import Config
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse

def nmse_loss(y_true, y_pred):
    return np.sum((y_pred - y_true)**2) / (np.sum(y_true**2) + 1e-8)

def add_measurement_noise(y_true, snr_db=30):
    """Adds white gaussian noise to measurements"""
    signal_power = np.mean(y_true**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=y_true.shape)
    y_noisy = y_true + noise
    y_noisy[y_noisy < 0] = 0
    return y_noisy

def main():
    parser = ArgumentParser(description="Optimize ART Iterations")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg

    # --- Generate simulation data ---
    snr_db = 30
    print(f"Generating noisy simulation data (SNR={snr_db}dB)...")
    sim_data = generate_simulation_data(cfg)
    gt_img = sim_data.img_gt
    y_noisy = add_measurement_noise(sim_data.y_true.cpu().numpy(), snr_db=snr_db)
    # y_noisy = sim_data.y_true.cpu().numpy()

    grid_res = gt_img.shape[0] 
    cell_size = cfg.sim.map_size / grid_res

    system_matrix = create_system_matrix_sparse((grid_res, grid_res), sim_data.beams.tolist(), cell_size).tocsr()

    # --- Experiment configuration ---
    relax_factors = [0.05, 0.2, 0.8, 1.6]
    
    max_iterations = 1000
    step = 10
    iterations_x = list(range(step, max_iterations + 1, step))
    
    # Deactivate ART internal progress bar
    tm.tqdm = lambda x, **kwargs: x

    plt.figure(figsize=(10, 6))

    # --- Main loop ---
    for rf in relax_factors:
        print(f"\nEvaluating ART with relaxation factor = {rf}...")
        errors = []
        
        # Initial guess
        current_guess = np.zeros(system_matrix.shape[1])
        
        for _ in tqdm(range(len(iterations_x)), desc=f"RF={rf}", leave=False):
            # Use current guess to make "step" iterations
            current_guess = tm.art(
                system_matrix, 
                y_noisy, 
                num_iterations=step, 
                initial_guess=current_guess, 
                relaxation_factor=rf
            )
            
            # Evaluate reconstruction
            recon_img = current_guess.reshape((grid_res, grid_res))
            errors.append(nmse_loss(gt_img, recon_img))
            
        # Find minimum
        min_idx = np.argmin(errors)
        best_iter = iterations_x[min_idx]
        best_err = errors[min_idx]
        
        # Plot curve
        line, = plt.plot(iterations_x, errors, linewidth=2, label=f'Relaxation = {rf} (Minimum at iter {best_iter})')
        plt.plot(best_iter, best_err, marker='*', color=line.get_color(), markersize=12)

    # --- Graph details ---
    plt.title('ART Relaxation Optimization')
    plt.xlabel('Iterations')
    plt.ylabel('NMSE vs Ground Truth')
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save plot and show
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'art_iterations.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved in: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()