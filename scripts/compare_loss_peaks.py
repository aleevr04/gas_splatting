import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.sim_utils import generate_simulation_data
from utils.init_utils import setup_gs_model
from utils.plot_utils import render_gaussian_map, set_publication_style

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    parser = ArgumentParser(description="Compare Loss Peaks, Reconstructions, and Densification")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg = args.cfg
    base_cfg.train.early_stopping_min_delta = 0.0 # Deactivate early stopping
    
    sim_data = generate_simulation_data(base_cfg)
    gt_img = sim_data.img_gt
    
    methods = ["Original Split", "Long-Axis Split"]
    results = {}
    
    for method in methods:
        print(f"\nTraining with {method}...")
        test_cfg = copy.deepcopy(base_cfg)
        test_cfg.densify.long_axis_split = (method == "Long-Axis Split")
        
        # Enable mid-training evaluation
        test_cfg.train.do_eval = True
        test_cfg.train.eval_interval = 25
        
        model, _, _ = setup_gs_model(sim_data, test_cfg)
        trainer = Trainer(model, test_cfg)
        
        train_results = trainer.train(sim_data)
        
        # Render the final 2D image
        gs_img = render_gaussian_map(model, test_cfg.sim.map_size, test_cfg.device, cell_size=test_cfg.sim.cell_size)
        rmse = rmse_loss(gt_img, gs_img)
        
        # Store all relevant data
        results[method] = {
            "loss": train_results.loss_history,
            "densify": train_results.densify_history,
            "rmse_history": train_results.rmse_history,
            "img": gs_img
        }

    print("\nGenerating detailed comparison plot...")
    
    # --- Plotting ---
    set_publication_style()
    
    map_w, map_h = base_cfg.sim.map_size
    extent = (0, map_w, 0, map_h)
    vmin = 0
    vmax = max(gt_img.max(), results["Original Split"]["img"].max(), results["Long-Axis Split"]["img"].max())
    
    # --- FIGURE 1: Reconstructions ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # GT
    axes1[0].set_title("Ground Truth")
    im_gt = axes1[0].imshow(gt_img, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    
    # Original Split Recon
    axes1[1].set_title(f"Original Split")
    axes1[1].imshow(results["Original Split"]["img"], origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    
    # Long-Axis Split Recon
    axes1[2].set_title(f"Long-Axis Split")
    axes1[2].imshow(results["Long-Axis Split"]["img"], origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    
    # Shared colorbar
    fig1.colorbar(im_gt, ax=axes1.tolist(), label="ppm", fraction=0.015, pad=0.04)
    
    save_path_recon = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_loss_peaks_recon.png')
    os.makedirs(os.path.dirname(save_path_recon), exist_ok=True)
    fig1.savefig(save_path_recon)
    plt.close(fig1)
    print(f"Reconstruction plot saved in: {save_path_recon}")

    # --- FIGURE 2: Loss & RMSE ---
    fig2, (ax_loss, ax_rmse) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    d_from = base_cfg.densify.densify_from
    d_until = base_cfg.densify.densify_until
    d_interval = base_cfg.densify.densify_interval
    densify_iters = list(range(d_from, d_until + 1, d_interval))
    
    window_start = 0
    window_end = base_cfg.train.iterations
    
    # Loss
    ax_loss.plot(results["Original Split"]["loss"], label="Original Split", color="tab:orange", alpha=0.8)
    ax_loss.plot(results["Long-Axis Split"]["loss"], label="Long-Axis Split", color="tab:blue", alpha=0.8)
    
    for i, d_iter in enumerate(densify_iters):
        label = "Densification Event" if i == 0 else None
        ax_loss.axvline(x=d_iter, color='gray', linestyle='--', alpha=0.5, label=label)

    ax_loss.set_title("Loss Function and RMSE Evolution")
    ax_loss.set_ylabel("Total Loss (Log)")
    ax_loss.set_yscale('log')
    ax_loss.set_xlim(window_start, window_end)
    ax_loss.legend(loc='upper right')
    
    # RMSE
    iters_orig = list(results["Original Split"]["rmse_history"].keys())
    rmse_orig_vals = list(results["Original Split"]["rmse_history"].values())
    ax_rmse.plot(iters_orig, rmse_orig_vals, label="Original Split", color="tab:orange", alpha=0.8)
    
    iters_long = list(results["Long-Axis Split"]["rmse_history"].keys())
    rmse_long_vals = list(results["Long-Axis Split"]["rmse_history"].values())
    ax_rmse.plot(iters_long, rmse_long_vals, label="Long-Axis Split", color="tab:blue", alpha=0.8)
    
    for i, d_iter in enumerate(densify_iters):
        ax_rmse.axvline(x=d_iter, color='gray', linestyle='--', alpha=0.5)

    ax_rmse.set_xlabel("Iterations")
    ax_rmse.set_ylabel("RMSE (ppm)")
    # ax_rmse.legend()

    plt.tight_layout()

    # Save and show
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_loss_peaks.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved in: {save_path}")

if __name__ == "__main__":
    main()