import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.plot_utils import render_gaussian_map, plot_initial_guess, set_publication_style
from utils.data_utils import load_real_tdlas_data
from utils.sim_utils import xy2cell, simulate_gas_integrals, generate_radial_beams, generate_random_beams

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
    sim_data = load_real_tdlas_data(data_path, cfg)
    print(f"Map size: {cfg.sim.map_size[0]}m x {cfg.sim.map_size[1]}m")

    # ------ INJECT ARTIFICIAL TEST DATA ------
    # Simulate beams
    beams = []
    num_random_beams = cfg.sim.num_beams // 2
    num_radial_beams = cfg.sim.num_beams - num_random_beams 
    
    beams += generate_random_beams(cfg.sim.map_size, num_random_beams)
    beams += generate_radial_beams(cfg.sim.map_size, num_radial_beams)

    # Place artificial sources
    source1 = (5.0, 7.0)
    source2 = (11.0, 4.0)

    s1r, s1c = xy2cell(source1, cfg.sim.cell_size)
    s2r, s2c = xy2cell(source2, cfg.sim.cell_size)

    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    gas_map = np.zeros((grid_h, grid_w))

    gas_map[s1r][s1c] = 60.0
    gas_map[s2r][s2c] = 60.0

    gas_map = gaussian_filter(gas_map, sigma=1.5)

    # Recompute integral measurements
    measurements = simulate_gas_integrals(gas_map, beams, cfg.sim.cell_size)
    
    # Update simulation data object
    sim_data.img_gt = gas_map
    sim_data.beams = torch.tensor(beams, dtype=torch.float32, device=cfg.device)
    sim_data.measurements = torch.tensor(measurements, dtype=torch.float32, device=cfg.device)
    sim_data.y_true = sim_data.measurements
    # ------------------------------------

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