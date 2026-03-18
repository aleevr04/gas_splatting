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
    """Normalized Mean Square Error"""
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
    # --- Configuration ---
    parser = ArgumentParser(description="Optimize Hyperparameters for Traditional Methods")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg

    # --- Generate simulation data ---
    print("--- Simulation data ---")
    sim_data = generate_simulation_data(cfg)
    y_true_clean = sim_data.y_true.cpu().numpy()
    gt_img = sim_data.img_gt
    
    # Add noise
    # snr_db = 30
    # print(f"Adding noise to measurements (SNR = {snr_db}dB)...")
    # y_noisy = add_measurement_noise(y_true_clean, snr_db=snr_db)
    y_noisy = y_true_clean
    
    grid_res = gt_img.shape[0] 
    cell_size = cfg.sim.map_size / grid_res
    
    system_matrix = create_system_matrix_sparse((grid_res, grid_res), sim_data.beams.tolist(), cell_size).tocsr()

    # --- Grid Search configuration ---
    alphas = np.logspace(-4, 2, 20) 
    relax_factors = np.linspace(0.01, 1.0, 20) 

    results = {
        "Tikhonov": {"x": alphas, "errors": []},
        "LFD": {"x": alphas, "errors": []},
        "LSD": {"x": alphas, "errors": []},
        "LTD": {"x": alphas, "errors": []},
        "ART": {"x": relax_factors, "errors": []}
    }

    print("\n--- Starting hyperparameters sweep ---")

    # TIKHONOV
    print("Evaluating Tikhonov...")
    for alpha in tqdm(alphas, leave=False):
        recon = tm.tikhonov_direct(system_matrix, y_noisy, alpha)
        recon_img = recon.reshape((grid_res, grid_res))
        results["Tikhonov"]["errors"].append(nmse_loss(gt_img, recon_img))

    # LFD
    print("Evaluating LFD...")
    for alpha in tqdm(alphas, leave=False):
        recon = tm.lfd(system_matrix, y_noisy, (grid_res, grid_res), alpha)
        results["LFD"]["errors"].append(nmse_loss(gt_img, recon))
        
    # LSD
    print("Evaluating LSD...")
    for alpha in tqdm(alphas, leave=False):
        recon = tm.lsd(system_matrix, y_noisy, (grid_res, grid_res), alpha)
        results["LSD"]["errors"].append(nmse_loss(gt_img, recon))

    # LTD
    print("Evaluating LTD...")
    for alpha in tqdm(alphas, leave=False):
        recon = tm.ltd(system_matrix, y_noisy, (grid_res, grid_res), alpha)
        results["LTD"]["errors"].append(nmse_loss(gt_img, recon))

    # ART
    print("Evaluating ART...")
    for rf in tqdm(relax_factors, leave=False):
        # Deactivate ART internal progress bar
        import utils.tomo_utils as temp_tm
        temp_tm.tqdm = lambda x, **kwargs: x 
        recon = temp_tm.art(system_matrix, y_noisy, num_iterations=300, relaxation_factor=rf)
        recon_img = recon.reshape((grid_res, grid_res))
        results["ART"]["errors"].append(nmse_loss(gt_img, recon_img))

    # --- Visualization ---
    print("\n--- Generating graphs ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Alpha regularization parameter
    ax1 = axes[0]
    for method in ["Tikhonov", "LFD", "LSD", "LTD"]:
        errors = results[method]["errors"]
        min_idx = np.argmin(errors)
        best_alpha = alphas[min_idx]
        best_err = errors[min_idx]
        
        ax1.plot(alphas, errors, marker='.', label=f'{method} (Min: {best_alpha:.2e})')
        ax1.plot(best_alpha, best_err, 'r*', markersize=10) # Mark minimum
        
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Regularization Parameter $\alpha$ (Log Scale)')
    ax1.set_ylabel('NMSE vs Ground Truth')
    ax1.set_title('Regularization Optimization')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # ART Relaxation parameter
    ax2 = axes[1]
    art_errors = results["ART"]["errors"]
    min_idx_art = np.argmin(art_errors)
    best_rf = relax_factors[min_idx_art]
    best_err_art = art_errors[min_idx_art]

    ax2.plot(relax_factors, art_errors, 'g-o', label=f'ART (Min: {best_rf:.2f})')
    ax2.plot(best_rf, best_err_art, 'r*', markersize=10)
    ax2.set_xlabel('Relaxation Parameter (Linear)')
    ax2.set_ylabel('NMSE vs Ground Truth')
    ax2.set_title('ART Optimization')
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'plots'), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'tomo_optimization.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved in: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()