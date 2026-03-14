import time
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from skimage.metrics import structural_similarity as ssim

import utils.tomo_utils as tm
from config import Config
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import render_gaussian_map

def nmse_loss(y_true, y_pred):
    return np.sum((y_pred - y_true)**2) / (np.sum(y_true**2) + 1e-8)

def main():
    # Configuration
    parser = ArgumentParser(description="Compare Gas Splatting to Traditional Methods")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg
    cfg.train.no_live_vis = True
    
    # ------ Simulation data -------
    print(f"--- Simulation data ---")
    sim_data = generate_simulation_data(cfg)
    y_true = sim_data.y_true.cpu().numpy()
    
    extent = (0, cfg.sim.map_size, 0, cfg.sim.map_size)

    grid_res = sim_data.img_gt.shape[0] 
    cell_size = cfg.sim.map_size / grid_res
    
    # Measure setup time for traditional methods (System Matrix)
    print("Building system matrix...")
    t_setup_start = time.time()
    system_matrix = create_system_matrix_sparse((grid_res, grid_res), sim_data.beams.tolist(), cell_size).tocsr()
    trad_setup_time = time.time() - t_setup_start

    reconstructions = {}
    # Dictionary to hold both setup and reconstruction times
    execution_times = {} 
    
    # ------- Traditional Methods -------
    print("\n--- Traditional Methods ---")
    
    print("ART...")
    art_iterations = 500
    art_name = f"ART ({art_iterations} it)"
    t_start = time.time()
    art_res = tm.art(system_matrix, y_true, num_iterations=art_iterations)
    execution_times[art_name] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    reconstructions[art_name] = art_res.reshape((grid_res, grid_res))
    
    print("Tikhonov (Iterative)...")
    tikhonov_iterations = 5000
    tikhonov_name = f"Tikhonov ({tikhonov_iterations} it)"
    t_start = time.time()
    tik_res = tm.tikhonov_iterative(system_matrix, y_true, alpha=0.1, num_iterations=tikhonov_iterations)
    execution_times[tikhonov_name] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    reconstructions[tikhonov_name] = tik_res.reshape((grid_res, grid_res))
    
    print("LFD (Low First Derivative)...")
    t_start = time.time()
    reconstructions["LFD"] = tm.lfd(system_matrix, y_true, grid_size=(grid_res, grid_res), alpha=0.05)
    execution_times["LFD"] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    
    print("LTD (Low Third Derivative)...")
    t_start = time.time()
    reconstructions["LTD"] = tm.ltd(system_matrix, y_true, grid_size=(grid_res, grid_res), alpha=0.01)
    execution_times["LTD"] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    
    # ------- Gas Splatting -------
    print("\n--- Gas Splatting ---")
    
    # Measure Setup Time for Gas Splatting
    t_gs_setup_start = time.time()
    init_pos, init_concentration, init_std, _ = lsqr_initialization(
        sim_data.beams.tolist(), sim_data.y_true, cfg.sim.map_size, 
        num_gaussians=cfg.init.initial_gaussians, coarse_res=cfg.init.coarse_res
    )
    
    model = GasSplattingModel(init_pos.shape[0], cfg).to(cfg.device)
    model.initialize_gaussians(init_pos.to(cfg.device), init_concentration.to(cfg.device), init_std)
    gs_setup_time = time.time() - t_gs_setup_start
    
    # Measure Reconstruction (Training) Time for Gas Splatting
    t_gs_recon_start = time.time()
    trainer = Trainer(model, cfg)
    trainer.train(sim_data.beams, sim_data.y_true)
    gs_recon_time = time.time() - t_gs_recon_start
    
    gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, grid_res=grid_res)
    execution_times["Gas Splatting"] = {'setup': gs_setup_time, 'recon': gs_recon_time}
    reconstructions["Gas Splatting"] = gs_img
    
    # ------ Evaluation and Visualization -----
    print("\n--- Results ---")
    gt_img = sim_data.img_gt
    
    num_methods = len(reconstructions)
    fig, axes = plt.subplots(2, (num_methods + 2) // 2, figsize=(16, 8))
    axes = axes.flatten()
    
    # GT
    im_gt = axes[0].imshow(gt_img, origin='lower', extent=extent, cmap='jet')
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    for i in range(0, len(sim_data.beams)):
        (x0, y0), (x1, y1) = sim_data.beams[i]
        axes[0].plot([x0, x1], [y0, y1], 'w-', alpha=0.3, linewidth=1.0)
    fig.colorbar(im_gt, ax=axes[0])
    
    # Plot reconstructions
    for idx, (name, img) in enumerate(reconstructions.items(), start=1):
        nmse = nmse_loss(gt_img, img)
        data_range = gt_img.max() - gt_img.min()
        ssim_val = ssim(gt_img, img, data_range=data_range)
        
        t_setup = execution_times[name]['setup']
        t_recon = execution_times[name]['recon']

        # Multi-line title for clarity
        title = f"{name}\nNMSE: {nmse:.4f} | SSIM: {ssim_val:.4f}\nSetup: {t_setup:.2f}s | Recon: {t_recon:.2f}s"
        print(f"{name:<20}: NMSE = {nmse:.4f} | SSIM = {ssim_val:.4f} | Setup = {t_setup:.2f}s | Recon = {t_recon:.2f}s")
            
        im = axes[idx].imshow(img, origin='lower', extent=extent, cmap='jet')
        axes[idx].set_title(title, fontsize=14)
        axes[idx].axis('off')
        fig.colorbar(im, ax=axes[idx])
        
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()