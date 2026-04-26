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
from utils.plot_utils import render_gaussian_map

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    parser = ArgumentParser(description="Compare Loss Peaks, Reconstructions, and Densification")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg = args.cfg
    base_cfg.train.no_live_vis = True
    
    sim_data = generate_simulation_data(base_cfg)
    gt_img = sim_data.img_gt
    
    methods = ["Original Split", "Long-Axis Split"]
    results = {}
    
    for method in methods:
        print(f"\nTraining with {method}...")
        test_cfg = copy.deepcopy(base_cfg)
        test_cfg.densify.long_axis_split = (method == "Long-Axis Split")
        
        model, _, _ = setup_gs_model(sim_data, test_cfg)
        trainer = Trainer(model, test_cfg)
        
        # trainer.train returns loss_history, densify_history
        loss_history, densify_history = trainer.train(sim_data.beams, sim_data.measurements)
        
        # Render the final 2D image
        gs_img = render_gaussian_map(model, test_cfg.sim.map_size, test_cfg.device, cell_size=test_cfg.sim.cell_size)
        rmse = rmse_loss(gt_img, gs_img)
        
        # Store all relevant data
        results[method] = {
            "loss": loss_history,
            "densify": densify_history,
            "img": gs_img,
            "rmse": rmse,
            "num_gaussians": model.num_gaussians
        }

    print("\nGenerating detailed comparison plot...")
    
    # --- Plotting ---
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 3x6 grid. 
    # Row 0: 3 Images (2 columns each)
    # Row 1: 1 Loss plot (all 6 columns)
    # Row 2: 2 Densification plots (3 columns each)
    gs = gridspec.GridSpec(3, 6, height_ratios=[1.2, 1, 1.2], hspace=0.4, wspace=0.3)
    
    map_w, map_h = base_cfg.sim.map_size
    extent = (0, map_w, 0, map_h)
    vmin = 0
    vmax = max(gt_img.max(), results["Original Split"]["img"].max(), results["Long-Axis Split"]["img"].max())
    
    # --- Row 1: Images ---
    # GT
    ax_gt = fig.add_subplot(gs[0, 0:2])
    ax_gt.set_title("Ground Truth")
    im_gt = ax_gt.imshow(gt_img, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    
    # Original Split Recon
    ax_orig = fig.add_subplot(gs[0, 2:4])
    rmse_orig = results["Original Split"]["rmse"]
    n_orig = results["Original Split"]["num_gaussians"]
    ax_orig.set_title(f"Original Split\nRMSE: {rmse_orig:.4f} | Gaussians: {n_orig}")
    ax_orig.imshow(results["Original Split"]["img"], origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    
    # Long-Axis Split Recon
    ax_long = fig.add_subplot(gs[0, 4:6])
    rmse_long = results["Long-Axis Split"]["rmse"]
    n_long = results["Long-Axis Split"]["num_gaussians"]
    ax_long.set_title(f"Long-Axis Split\nRMSE: {rmse_long:.4f} | Gaussians: {n_long}")
    ax_long.imshow(results["Long-Axis Split"]["img"], origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    
    fig.colorbar(im_gt, ax=[ax_gt, ax_orig, ax_long], label="ppm", fraction=0.015, pad=0.02)
    
    # --- Row 2: Loss ---
    ax_loss = fig.add_subplot(gs[1, :])
    ax_loss.plot(results["Original Split"]["loss"], label="Original Split", color="tab:orange", alpha=0.8, linewidth=1.5)
    ax_loss.plot(results["Long-Axis Split"]["loss"], label="Long-Axis Split", color="tab:blue", alpha=0.8, linewidth=1.5)
    
    d_from = base_cfg.densify.densify_from
    d_until = base_cfg.densify.densify_until
    d_interval = base_cfg.densify.densify_interval
    
    densify_iters = list(range(d_from, d_until + 1, d_interval))
    for i, d_iter in enumerate(densify_iters):
        label = "Densification Event" if i == 0 else None
        ax_loss.axvline(x=d_iter, color='gray', linestyle='--', alpha=0.5, label=label)

    ax_loss.set_title("Loss Function Peaks During Densification")
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Total Loss (Log Scale)")
    ax_loss.set_yscale('log')
    
    window_start = max(0, d_from - 50)
    window_end = min(base_cfg.train.iterations, d_until + 150)
    ax_loss.set_xlim(window_start, window_end)
    ax_loss.grid(True, which="both", linestyle='--', alpha=0.3)
    ax_loss.legend(loc='upper right')
    
    # --- Row 3: Densification Stats ---
    bar_width = d_interval * 0.4
    
    def plot_densify_stats(ax, densify_history, title):
        if not densify_history:
            ax.set_title(title + " (No Densification)")
            return
        iters = list(densify_history.keys())
        clones = [d['clones'] for d in densify_history.values()]
        splits = [d['splits'] for d in densify_history.values()]
        prunes = [d['prunes'] for d in densify_history.values()]
        
        ax.bar(iters, clones, width=bar_width, label='Clones', color='skyblue')
        ax.bar(iters, splits, width=bar_width, bottom=clones, label='Splits', color='orange')
        bottom_prunes = [c + s for c, s in zip(clones, splits)]
        ax.bar(iters, prunes, width=bar_width, bottom=bottom_prunes, label='Prunes', color='red')
        
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Count")
        ax.set_xlim(window_start, window_end)
        ax.grid(True, axis='y', ls="--", alpha=0.3)
        ax.legend(loc='upper right')
        
    ax_dens_orig = fig.add_subplot(gs[2, 0:3])
    plot_densify_stats(ax_dens_orig, results["Original Split"]["densify"], "Densification Stats: Original Split")
    
    ax_dens_long = fig.add_subplot(gs[2, 3:6], sharey=ax_dens_orig)
    plot_densify_stats(ax_dens_long, results["Long-Axis Split"]["densify"], "Densification Stats: Long-Axis Split")

    # Save and show
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_loss_peaks_and_recon.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved in: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()