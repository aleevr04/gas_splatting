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
from utils.init_utils import InitializationData, setup_gs_model
from utils.sim_utils import SimulationData, MeasurementBatch
from utils.data_utils import build_custom_real_scenario
from utils.plot_utils import plot_initial_guess, set_publication_style

def plot_real_results(model: GasSplattingModel, batch_data: MeasurementBatch, results, cfg: Config, save_path):
    img_pred = model.render_map(cell_size=cfg.sim.cell_size)
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
    beams_np = batch_data.beams.cpu().numpy()
    meas_np = batch_data.measurements.cpu().numpy()
    
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
    print(f"[+] Real data results plot saved in: {save_path}")
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

    if not cfg.quiet: print(f"Using device: {cfg.device}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'real_data', 'full_sweep.json') 
    data = build_custom_real_scenario(
        cfg=cfg,
        real_data_path=data_path,
        use_sim_beams=args.sim_beams,
        use_sim_gas=args.sim_gas
    )

    # Safely extract batch and ground_truth
    if isinstance(data, SimulationData):
        batch = data.batch
        gt_img = data.ground_truth
    else:
        batch = data
        gt_img = None

    # ----- MODEL INITIALIZATION AND TRAINING ------
    if args.force_init:
        # Forced Initialization
        pos_tensor = torch.tensor([[5.0, 7.0], [9.0, 4.0]], dtype=torch.float32)
        init_data = InitializationData(
            pos=pos_tensor,
            concentration=torch.full((2,), 10.0, dtype=torch.float32),
            std=torch.full((2,), 1.0, dtype=torch.float32)
        )
        model = GasSplattingModel(initial_gaussians=2, cfg=cfg)
        model.initialize_gaussians(
            pos=init_data.pos.to(cfg.device),
            concentration=init_data.concentration.to(cfg.device),
            std=init_data.std.to(cfg.device)
        )
        plot_initial_guess(gt_img, init_data, cfg)
    else:
        # Automatic Initialization
        model, init_data = setup_gs_model(batch, cfg)
        if not cfg.quiet: print(f"Model initialized with {model.num_gaussians} Gaussians.")
        plot_initial_guess(gt_img, init_data, cfg)

    # Training
    trainer = Trainer(model, cfg)
    trainer.train(batch)
    results = trainer.finish()
    # ------------------------------------------------

    # Plot results
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'real_data_results.png')
    plot_real_results(model, batch, results, cfg, save_path)

if __name__ == "__main__":
    main()