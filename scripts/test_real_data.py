import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.plot_utils import render_gaussian_map, plot_initial_guess, set_publication_style
from utils.data_utils import load_real_tdlas_data

def plot_real_results(model, sim_data, results, cfg, save_path):
    img_pred = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
    map_w, map_h = cfg.sim.map_size
    
    fig, (ax_map, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Reconstruction ---
    ax_map.set_title(f"GS Reconstruction (cell size = {cfg.sim.cell_size}m)\nGaussians: {model.num_gaussians}")
    im = ax_map.imshow(img_pred, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')
    
    ax_map.set_xlim(0, map_w)
    ax_map.set_ylim(0, map_h)
    ax_map.set_aspect('equal')
    
    # Beams
    beams_np = sim_data.beams.cpu().numpy()
    for i in range(len(beams_np)):
        (x0, y0), (x1, y1) = beams_np[i]
        ax_map.plot([x0, x1], [y0, y1], 'w-', alpha=0.15, linewidth=0.5)
        
    pos = model.get_pos().detach().cpu().numpy()
    ax_map.scatter(pos[:, 0], pos[:, 1], marker='+', c='k', alpha=0.5, label="Splats")
    
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    fig.colorbar(im, ax=ax_map, label="ppm", fraction=0.046, pad=0.04)
    
    # --- Loss ---
    ax_loss.set_title(f"Training\nFinal loss: {results.loss_history[-1]:.4f}")
    ax_loss.plot(results.loss_history, color='tab:blue', linewidth=2)
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log')
    ax_loss.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"\n[+] Real data results plot saved in: {save_path}")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Real data training")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg

    set_publication_style()

    print(f"Using device: {cfg.device}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'ground_truth', 'real_data', '1_1.json') 
    print(f"Loading real data from {data_path}...")
    sim_data = load_real_tdlas_data(data_path)
    
    # Map boundaries based on beams geometry
    all_coordinates = sim_data.beams.view(-1, 2)
    min_coords = torch.min(all_coordinates, dim=0).values
    max_coords = torch.max(all_coordinates, dim=0).values

    margin = 0.5
    min_x, min_y = min_coords[0].item() - margin, min_coords[1].item() - margin
    max_x, max_y = max_coords[0].item() + margin, max_coords[1].item() + margin

    # Apply offset to beams
    offset = torch.tensor([-min_x, -min_y], device=cfg.device)
    sim_data.beams += offset

    # New map size
    map_w = max_x - min_x
    map_h = max_y - min_y
    cfg.sim.map_size = (map_w, map_h)
    
    print(f"Computed map size: {map_w:.2f}m x {map_h:.2f}m")
    print(f"Total beams: {sim_data.beams.shape[0]}")

    # Empty GT
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    sim_data.img_gt = np.zeros((grid_h, grid_w))

    # Training
    model, init_pos, img_coarse = setup_gs_model(sim_data, cfg)
    print(f"Model initialized with {model.num_gaussians} gaussians.")
    plot_initial_guess(sim_data.img_gt, img_coarse, init_pos, cfg)

    trainer = Trainer(model, cfg)
    results = trainer.train(sim_data)
    
    # Plot results
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'real_data_results.png')
    plot_real_results(model, sim_data, results, cfg, save_path)

if __name__ == "__main__":
    main()