import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import SimulationData
from trainer import TrainingResults

def render_gaussian_map(gaussians: GasSplattingModel, map_size: tuple[float, float], device: torch.device, cell_size):
    """
    Turns gaussians into a 2D image (numpy matrix)
    """

    w_cells = int(map_size[0] / cell_size)
    h_cells = int(map_size[1] / cell_size)

    # Grid
    x = torch.linspace(0, map_size[0], w_cells, device=device)
    y = torch.linspace(0, map_size[1], h_cells, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_pos = torch.stack([X, Y], dim=-1) # (H, W, 2)

    # PyTorch expects (H, W) -> (h_cells, w_cells)
    final_img = torch.zeros((h_cells, w_cells), device=device)

    with torch.no_grad():
        pos = gaussians.get_pos()
        cov_inv = gaussians.get_covariance_inverse()
        concentration = gaussians.get_concentration()

        # Sum each Gaussian contribution
        for k in range(gaussians.num_gaussians):
            mu = pos[k]
            sig_inv = cov_inv[k]
            c = concentration[k]
            
            # Evaluate Gaussian at each cell
            d = grid_pos - mu
            d = d.unsqueeze(-1) 
            
            sig_inv_exp = sig_inv.view(1, 1, 2, 2)
            dist = torch.matmul(d.transpose(-1, -2), torch.matmul(sig_inv_exp, d)).squeeze()
            
            final_img += c * torch.exp(-0.5 * dist)

    return final_img.detach().cpu().numpy()

def plot_initial_guess(img_gt, img_coarse, init_pos, cfg: Config):
    """Shows ground truth and initial reconstruction image"""

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
    ax2.scatter(init_pos[:, 0], init_pos[:, 1], marker='X', c='w', edgecolors='k', s=50, linewidths=1.2, label='Peaks')
    ax2.legend()

    fig.colorbar(im1, ax=[ax1, ax2], label="ppm", fraction=0.025, pad=0.05)

    plt.show()

def plot_training_results(gaussians: GasSplattingModel, sim_data: SimulationData, results: TrainingResults, cfg: Config):
    """Shows GT, GS reconstruction, loss history, and densification events"""

    map_w, map_h = cfg.sim.map_size
    grid_w = int(map_w / cfg.sim.cell_size)
    grid_h = int(map_h / cfg.sim.cell_size)
    max_map_dim = max(map_w, map_h)

    # Generate images
    img_pred_gaussian = render_gaussian_map(gaussians, cfg.sim.map_size, cfg.device, cell_size=max_map_dim / 100)
    img_pred = render_gaussian_map(gaussians, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
    
    # Colormap min and max values
    vmin = 0
    vmax = max(sim_data.img_gt.max(), img_pred.max())

    # RMSE
    mse = np.mean((img_pred - sim_data.img_gt)**2)
    rmse = np.sqrt(mse)

    fig = plt.figure(figsize=(15, 8)) 
    
    # Grid of 3 rows. Images on top, Loss in the middle, Densification below
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1], hspace=0.3)

    fig.suptitle(f"Initial Gaussians = {gaussians.initial_gaussians}  |  Final = {gaussians.num_gaussians}  |  Beams = {cfg.sim.num_beams}")

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

    # 3. Reconstruction Grid (Top Righ)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title(f"GS Reconstruction (Grid)\nRMSE = {rmse:.4f}")
    ax3.imshow(img_pred, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet', vmin=vmin, vmax=vmax)

    # Single Colorbar
    fig.colorbar(im1, ax=[ax1, ax2, ax3], label="ppm", fraction=0.015, pad=0.02)

    # 4. Loss History (Middle Row)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title(f"Loss History (Final Loss = {results.loss_history[-1]:.4f})")
    ax4.plot(results.loss_history, color='blue', alpha=0.8)
    ax4.set_ylabel("Total Loss")
    ax4.set_yscale('log')
    ax4.set_xlim(0, len(results.loss_history))
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    ax4.tick_params(labelbottom=False) # Ocultamos los números de X para que no rocen con el gráfico de abajo

    # 5. Densification Events (Bottom Row)
    ax5 = fig.add_subplot(gs[2, :], sharex=ax4)
    ax5.set_title("Densification Events")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Count")

    # Extract densification stats
    iters = list(results.densify_history.keys())
    clones = [d['clones'] for d in results.densify_history.values()]
    splits = [d['splits'] for d in results.densify_history.values()]
    prunes = [d['prunes'] for d in results.densify_history.values()]

    # Stacked bars
    bar_width = cfg.densify.densify_interval * 0.4 
    ax5.bar(iters, clones, width=bar_width, label='Clones', color='skyblue')
    ax5.bar(iters, splits, width=bar_width, bottom=clones, label='Splits', color='orange')
    
    bottom_prunes = [c + s for c, s in zip(clones, splits)]
    ax5.bar(iters, prunes, width=bar_width, bottom=bottom_prunes, label='Prunes', color='red')

    ax5.legend(loc='upper right')
    ax5.grid(True, axis='y', ls="--", alpha=0.3)

    plt.show()