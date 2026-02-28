import torch
import matplotlib.pyplot as plt

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import SimulationData

def render_gaussian_map(gaussians: GasSplattingModel, map_size: float, device: torch.device, grid_res=100):
    """
    Turns gaussians into a 2D image (numpy matrix)
    """

    # Crear grid de coordenadas
    x = torch.linspace(0, map_size, grid_res, device=device)
    y = torch.linspace(0, map_size, grid_res, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_pos = torch.stack([X, Y], dim=-1) # (H, W, 2)

    final_img = torch.zeros((grid_res, grid_res), device=device)

    with torch.no_grad():
        pos = gaussians.get_pos()
        cov_inv = gaussians.get_covariance_inverse()
        concentration = gaussians.get_concentration()

        # Sumar contribución de cada gaussiana
        for k in range(gaussians.num_gaussians):
            mu = pos[k]
            sig_inv = cov_inv[k]
            c = concentration[k]
            
            d = grid_pos - mu
            d = d.unsqueeze(-1) 
            
            # Distancia de Mahalanobis: d^T * Sigma^-1 * d
            # Ajustamos dimensiones para matmul eficiente
            sig_inv_exp = sig_inv.view(1, 1, 2, 2)
            dist = torch.matmul(d.transpose(-1, -2), torch.matmul(sig_inv_exp, d)).squeeze()
            
            final_img += c * torch.exp(-0.5 * dist)

    # Retornar como numpy array (para tomo_utils)
    return final_img.detach().cpu().numpy()

def plot_initial_guess(img_coarse, init_pos, map_size):
    """Shows initial reconstruction image"""

    plt.figure()
    plt.title("Algebraic Initialization (Least Squares)")
    plt.imshow(img_coarse, origin='lower', extent=(0, map_size, 0, map_size))
    plt.scatter(init_pos[:,0], init_pos[:,1], c='r', marker='x', label='Peaks')
    plt.legend()
    plt.show()

def plot_final_results(gaussians: GasSplattingModel, sim_data: SimulationData, loss_history, cfg: Config, device: torch.device):
    """Shows GT, Gas Splatting reconstruction and loss history"""
    
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Initial Gaussians = {gaussians.initial_gaussians}\nFinal Gaussians = {gaussians.num_gaussians}\nBeams = {cfg.sim.num_beams}")

    # 1. GT
    plt.subplot(2, 3, 1)
    plt.title(f"Ground Truth (Grid {cfg.sim.grid_res}x{cfg.sim.grid_res})")
    plt.imshow(sim_data.img_gt, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size), cmap='viridis')

    for i in range(0, len(sim_data.beams)):
        (x0, y0), (x1, y1) = sim_data.beams[i]
        plt.plot([x0, x1], [y0, y1], 'w-', alpha=0.2, linewidth=0.5)
    plt.colorbar(label="ppm")

    # 2. Reconstruction
    img_pred_gaussian = render_gaussian_map(gaussians, cfg.sim.map_size, device, grid_res=100)
    pos = gaussians.get_pos().detach().cpu().numpy()

    plt.subplot(2, 3, 2)
    plt.title(f"GS Reconstruction")
    plt.imshow(img_pred_gaussian, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size), cmap='viridis')
    plt.colorbar(label="ppm")
    plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5)

    img_pred = render_gaussian_map(gaussians, cfg.sim.map_size, device, cfg.sim.grid_res)

    plt.subplot(2, 3, 3)
    plt.title(f"GS Reconstruction (Grid)")
    plt.imshow(img_pred, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size), cmap='viridis')
    plt.colorbar(label="ppm")

    # 3. Loss History
    plt.subplot(2, 1, 2)
    plt.title("Loss History")
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.show()