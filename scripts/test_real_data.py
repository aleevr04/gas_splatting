import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from gs_model import GasSplattingModel
from utils.init_utils import setup_gs_model
from utils.data_utils import build_custom_real_scenario
from utils.plot_utils import render_gaussian_map, plot_initial_guess, set_publication_style

def plot_real_results(model, sim_data, results, cfg, save_path):
    img_pred = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
    map_w, map_h = cfg.sim.map_size
    
    fig, (ax_map, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Reconstruction ---
    ax_map.set_title(f"GS Reconstruction (cell size = {cfg.sim.cell_size}m)\nGaussians: {model.num_gaussians}")
    im = ax_map.imshow(img_pred, origin='lower', extent=(0, map_w, 0, map_h), cmap='jet')
    
    ax_map.set_xlim(0, map_w)
    ax_map.set_ylim(0, map_h)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_aspect('equal')

    # Colorbar
    fig.colorbar(im, ax=ax_map, label="ppm", fraction=0.046, pad=0.04)
    
    # --- Beams ---
    beams_np = sim_data.beams.cpu().numpy()
    meas_np = sim_data.measurements.cpu().numpy()
    
    # Normalize
    meas_min, meas_max = meas_np.min(), meas_np.max()
    if meas_max > meas_min:
        meas_norm = (meas_np - meas_min) / (meas_max - meas_min)
    else:
        meas_norm = np.zeros_like(meas_np)
        
    # Opacity range
    min_alpha = 0.05
    max_alpha = 0.3 
    alphas = min_alpha + meas_norm * (max_alpha - min_alpha)
    
    step = 1 
    for i in range(0, len(beams_np), step):
        (x0, y0), (x1, y1) = beams_np[i]
        ax_map.plot([x0, x1], [y0, y1], color='white', alpha=float(alphas[i]), linewidth=1.0)
    
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
    parser.add_argument('--sim_beams', dest='sim_beams', action='store_true', default=False, help="Use the same beams' geometry as in simulated environments (radial and random beams). If this option is set, simulated sources must be used aswell.")
    parser.add_argument('--sim_gas', dest='sim_gas', action='store_true', default=False, help='Inject simulated sources in the environment.')
    parser.add_argument('--force_init', dest='force_init', action='store_true', default=False, help="Force model initialization and place one Gaussian at each simulated gas source.")
    args = parser.parse_args()
    cfg: Config = args.cfg

    set_publication_style()

    print(f"Using device: {cfg.device}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'real_data', 'full_sweep.json')
    
    sim_data = build_custom_real_scenario(
        cfg=cfg,
        real_data_path=data_path,
        use_sim_beams=args.sim_beams,
        use_sim_gas=args.sim_gas
    )

    # ----- MODEL INITIALIZATION AND TRAINING ------
    if args.force_init:
        # Forced Initialization
        init_pos = np.array([[5.0, 7.0], [9.0, 4.0]])
        model = GasSplattingModel(initial_gaussians=2, cfg=cfg)
        model.initialize_gaussians(
            pos=torch.tensor(init_pos, device=cfg.device),
            concentration=torch.tensor(10.0),
            std=torch.tensor(1.0)
        )
        grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
        grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
        plot_initial_guess(sim_data.img_gt, np.zeros((grid_h, grid_w)), init_pos,cfg)
    else:
        # LQSR Initialization
        model, init_pos, img_coarse = setup_gs_model(sim_data, cfg)
        print(f"Model initialized with {model.num_gaussians} gaussians.")
        plot_initial_guess(sim_data.img_gt, img_coarse, init_pos, cfg)

    # Training
    trainer = Trainer(model, cfg)
    results = trainer.train(sim_data)
    # ------------------------------------------------

    # Plot results
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'real_data_results.png')
    plot_real_results(model, sim_data, results, cfg, save_path)

if __name__ == "__main__":
    main()