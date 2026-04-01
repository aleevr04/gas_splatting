import torch
import matplotlib.pyplot as plt

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import SimulationData

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

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Ground Truth ({img_gt.shape[0]}x{img_gt.shape[1]})")
    plt.imshow(img_gt, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')
    plt.colorbar(label="ppm", fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.title(f"Algebraic Initialization ({img_coarse.shape[0]}x{img_coarse.shape[1]})")
    plt.imshow(img_coarse, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')
    plt.colorbar(label="ppm", fraction=0.046, pad=0.04)
    plt.scatter(init_pos[:,0], init_pos[:,1], c='r', marker='x', label='Peaks')
    plt.legend()

    plt.show()

def plot_training_results(gaussians: GasSplattingModel, sim_data: SimulationData, loss_history, cfg: Config):
    """Shows GT, Gas Splatting reconstruction and loss history"""

    map_w, map_h = cfg.sim.map_size
    grid_w = int(map_w / cfg.sim.cell_size)
    grid_h = int(map_h / cfg.sim.cell_size)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Initial Gaussians = {gaussians.initial_gaussians}\nFinal Gaussians = {gaussians.num_gaussians}\nBeams = {cfg.sim.num_beams}")

    # 1. GT
    plt.subplot(2, 3, 1)
    plt.title(f"Ground Truth (Grid {grid_w}x{grid_h})")
    plt.imshow(sim_data.img_gt, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')

    for i in range(0, len(sim_data.beams)):
        (x0, y0), (x1, y1) = sim_data.beams[i]
        plt.plot([x0, x1], [y0, y1], 'w-', alpha=0.3, linewidth=1.0)
    plt.colorbar(label="ppm")

    # 2. Reconstruction
    img_pred_gaussian = render_gaussian_map(gaussians, cfg.sim.map_size, cfg.device, cell_size=0.1)
    pos = gaussians.get_pos().detach().cpu().numpy()

    plt.subplot(2, 3, 2)
    plt.title(f"GS Reconstruction")
    plt.imshow(img_pred_gaussian, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')
    plt.colorbar(label="ppm")
    plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5)

    img_pred = render_gaussian_map(gaussians, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)

    plt.subplot(2, 3, 3)
    plt.title(f"GS Reconstruction (Grid)")
    plt.imshow(img_pred, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')
    plt.colorbar(label="ppm")

    # 3. Loss History
    plt.subplot(2, 1, 2)
    plt.title("Loss History")
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.show()