import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import Config
from gs_model import GasSplattingModel
from trainer import TrainingResults
from utils.sim_utils import SimulationData

def set_publication_style():
    """Configures Matplotlib global style"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'grid.alpha': 0.7,
        'grid.linestyle': '--',
        # --- Saving configuration ---
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.05
    })

def plot_initial_guess(img_gt, img_coarse, init_pos, cfg: Config):
    """Shows ground truth and initial reconstruction image, and saves it to the plots directory"""

    map_w, map_h = cfg.sim.map_size

    vmin = 0
    vmax = max(img_gt.max(), img_coarse.max())

    fig = plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(f"Ground Truth ({img_gt.shape[0]}x{img_gt.shape[1]})")
    im1 = ax1.imshow(img_gt, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet', vmin=vmin, vmax=vmax)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f"Algebraic Initialization ({img_coarse.shape[0]}x{img_coarse.shape[1]})")
    ax2.imshow(img_coarse, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet', vmin=vmin, vmax=vmax)
    ax2.scatter(init_pos[:, 0], init_pos[:, 1], marker='X', c='w', edgecolors='k', s=90, linewidths=1.2, label='Peaks')
    ax2.legend()

    fig.colorbar(im1, ax=[ax1, ax2], label="ppm", fraction=0.025, pad=0.05)

    # Save plot instead of showing it
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'initial_guess.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[+] Initial guess plot saved in: {save_path}")
    
    plt.close(fig) # Free memory

def plot_training_results(gaussians: GasSplattingModel, sim_data: SimulationData, results: TrainingResults, cfg: Config):
    """Saves a plot with GT, GS reconstruction, loss history (with optional RMSE), and densification events"""

    map_w, map_h = cfg.sim.map_size
    grid_w = int(map_w / cfg.sim.cell_size)
    grid_h = int(map_h / cfg.sim.cell_size)
    max_map_dim = max(map_w, map_h)

    # Generate images
    img_pred_gaussian = gaussians.render_map(cell_size=max_map_dim / 100)
    img_pred = gaussians.render_map(cell_size=cfg.sim.cell_size)
    
    # Colormap min and max values
    vmin = 0
    vmax = max(sim_data.img_gt.max(), img_pred.max())

    # RMSE
    mse = np.mean((img_pred - sim_data.img_gt)**2)
    rmse = np.sqrt(mse)

    fig = plt.figure(figsize=(12, 8)) 
    
    # Grid of 3 rows. Images on top, Loss in the middle, Densification below
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1], hspace=0.3)

    # 1. GT (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f"Ground Truth ({grid_w}x{grid_h})")
    im1 = ax1.imshow(sim_data.img_gt, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet', vmin=vmin, vmax=vmax)
    for i in range(len(sim_data.beams)):
        (x0, y0), (x1, y1) = sim_data.beams[i]
        ax1.plot([x0, x1], [y0, y1], 'w-', alpha=0.3, linewidth=1.0)

    # 2. Reconstruction (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    pos = gaussians.get_pos().detach().cpu().numpy()
    ax2.set_title("GS Reconstruction")
    ax2.imshow(img_pred_gaussian, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax2.scatter(pos[:, 0], pos[:, 1], marker='P', c='k', edgecolors='w', s=30, linewidths=1.0)

    # 3. Reconstruction Grid (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title(f"GS Reconstruction (Grid)\nRMSE = {rmse:.4f}")
    ax3.imshow(img_pred, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet', vmin=vmin, vmax=vmax)

    # Single Colorbar
    fig.colorbar(im1, ax=[ax1, ax2, ax3], label="ppm", fraction=0.015, pad=0.02)

    # ==========================================
    # 4. Loss History (Middle Row)
    # ==========================================
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title(f"Loss History (Final Loss = {results.loss_history[-1]:.4f})")
    
    # Plot standard Loss on the left axis
    line_loss = ax4.plot(results.loss_history, color='blue', alpha=0.6, label="Total Loss (L1)")
    ax4.set_ylabel("Total Loss", color='blue')
    ax4.set_yscale('log')
    ax4.set_xlim(0, len(results.loss_history))
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.tick_params(labelbottom=False)

    lines = line_loss
    
    # If rmse_history has data, plot it on a twin right axis
    if results.rmse_history:
        ax4_rmse = ax4.twinx()
        iters = list(results.rmse_history.keys())
        rmse_vals = list(results.rmse_history.values())
        
        line_rmse = ax4_rmse.plot(iters, rmse_vals, color='red', alpha=0.8, linewidth=2, marker='o', markersize=4, label="Spatial RMSE")
        ax4_rmse.set_ylabel("RMSE (ppm)", color='red')
        ax4_rmse.tick_params(axis='y', labelcolor='red')
        
        lines += line_rmse # Combine lines for a single legend
        
    labels = [str(l.get_label()) for l in lines]
    ax4.legend(lines, labels, loc='upper right')

    # ==========================================
    # 5. Densification Events (Bottom Row)
    # ==========================================
    ax5 = fig.add_subplot(gs[2, :], sharex=ax4)
    ax5.set_title("Densification Events")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Count")

    if results.densify_history:
        iters = list(results.densify_history.keys())
        clones = [d['clones'] for d in results.densify_history.values()]
        splits = [d['splits'] for d in results.densify_history.values()]
        prunes = [d['prunes'] for d in results.densify_history.values()]

        bar_width = cfg.densify.densify_interval * 0.4 
        ax5.bar(iters, clones, width=bar_width, label='Clones', color='skyblue')
        ax5.bar(iters, splits, width=bar_width, bottom=clones, label='Splits', color='orange')
        
        bottom_prunes = [c + s for c, s in zip(clones, splits)]
        ax5.bar(iters, prunes, width=bar_width, bottom=bottom_prunes, label='Prunes', color='red')

        ax5.legend(loc='upper right')
        
    ax5.grid(True, axis='y', ls="--", alpha=0.3)

    # Save plot instead of showing it
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'training_results.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[+] Training results plot saved in: {save_path}")
    
    plt.close(fig) # Free memory

def plot_experiment_evolution(x_values, x_label, methods_info, results_rmse, results_ssim, results_time, save_path):
    """
    Generates a generic 1x3 plot (RMSE, SSIM, Time) for any experiment and saves it.
    """
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    for method in methods_info.keys():
        style = methods_info[method]["style"]
        
        # Means and std
        rmse_means = [np.mean(results_rmse[method][x]) for x in x_values]
        rmse_stds  = [np.std(results_rmse[method][x]) for x in x_values]
        
        ssim_means = [np.mean(results_ssim[method][x]) for x in x_values]
        ssim_stds  = [np.std(results_ssim[method][x]) for x in x_values]
        
        time_means = [np.mean(results_time[method][x]) for x in x_values]
        time_stds  = [np.std(results_time[method][x]) for x in x_values]

        # --- RMSE ---
        ax1.plot(x_values, rmse_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax1.fill_between(x_values, 
                         np.array(rmse_means) - np.array(rmse_stds), 
                         np.array(rmse_means) + np.array(rmse_stds), 
                         color=style["color"], alpha=0.15)

        # --- SSIM ---
        ax2.plot(x_values, ssim_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax2.fill_between(x_values, 
                         np.array(ssim_means) - np.array(ssim_stds), 
                         np.array(ssim_means) + np.array(ssim_stds), 
                         color=style["color"], alpha=0.15)

        # --- Time ---
        ax3.plot(x_values, time_means, label=method, 
                 color=style["color"], marker=style["marker"], 
                 linewidth=style.get("linewidth", 1.5), 
                 markersize=style.get("markersize", 6))
        ax3.fill_between(x_values, 
                         np.array(time_means) - np.array(time_stds), 
                         np.array(time_means) + np.array(time_stds), 
                         color=style["color"], alpha=0.15)

    # --- Plot details ---
    ax1.set_title("RMSE Evolution", pad=15)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("RMSE (ppm)")
    ax1.set_xticks(x_values)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title("SSIM Evolution", pad=15)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("SSIM")
    ax2.set_xticks(x_values)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    ax3.set_title("Time Evolution", pad=15)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Total Time (seconds)")
    ax3.set_xticks(x_values)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()

    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n[+] Experiment plot saved in: {save_path}")
    
    plt.close(fig) # Free memory