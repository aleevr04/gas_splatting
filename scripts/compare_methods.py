import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.tomo_utils as tm
from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.plot_utils import render_gaussian_map


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
    measurements = sim_data.measurements.cpu().numpy()
    
    extent = (0, cfg.sim.map_size[0], 0, cfg.sim.map_size[1])
    
    # Measure setup time for traditional methods (System Matrix)
    print("Building system matrix...")
    t_setup_start = time.time()
    system_matrix = create_system_matrix_sparse(sim_data.img_gt.shape, sim_data.beams.tolist(), cfg.sim.cell_size).tocsr()
    trad_setup_time = time.time() - t_setup_start

    reconstructions = {}
    # Dictionary to hold both setup and reconstruction times
    execution_times = {} 
    
    # ------- Traditional Methods -------
    print("\n--- Traditional Methods ---")
    
    print("ART...")
    art_iterations = 400
    art_name = f"ART ({art_iterations} it)"
    t_start = time.time()
    art_res = tm.art(system_matrix, measurements, num_iterations=art_iterations, relaxation_factor=1.6)
    execution_times[art_name] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    reconstructions[art_name] = art_res.reshape(sim_data.img_gt.shape)
    
    print("Tikhonov...")
    t_start = time.time()
    tik_res = tm.tikhonov_direct(system_matrix, measurements, alpha=0.2)
    execution_times["Tikhonov"] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    reconstructions["Tikhonov"] = tik_res.reshape(sim_data.img_gt.shape)

    print("LFD (Low First Derivative)...")
    t_start = time.time()
    reconstructions["LFD"] = tm.lfd(system_matrix, measurements, grid_size=sim_data.img_gt.shape, alpha=0.07)
    execution_times["LFD"] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    
    print("LTD (Low Third Derivative)...")
    t_start = time.time()
    reconstructions["LTD"] = tm.ltd(system_matrix, measurements, grid_size=sim_data.img_gt.shape, alpha=5.0)
    execution_times["LTD"] = {'setup': trad_setup_time, 'recon': time.time() - t_start}
    
    # ------- Gas Splatting -------
    print("\n--- Gas Splatting ---")
    
    # Measure Setup Time for Gas Splatting
    t_gs_setup_start = time.time()
    model, _, _ = setup_gs_model(sim_data, cfg)
    gs_setup_time = time.time() - t_gs_setup_start
    
    # Measure Reconstruction (Training) Time for Gas Splatting
    t_gs_recon_start = time.time()
    trainer = Trainer(model, cfg)
    trainer.train(sim_data.beams, sim_data.measurements)
    gs_recon_time = time.time() - t_gs_recon_start
    
    gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
    execution_times["Gas Splatting"] = {'setup': gs_setup_time, 'recon': gs_recon_time}
    reconstructions["Gas Splatting"] = gs_img
    
    # ------ Evaluation and Visualization -----
    print("\n--- Results ---")
    
    gt_img = sim_data.img_gt
    num_methods = len(reconstructions)

    all_imgs = [gt_img] + list(reconstructions.values())
    vmin_global = 0
    vmax_global = max(img.max() for img in all_imgs)

    # --- FIGURE 1: RECONSTRUCTIONS ---
    fig, axes = plt.subplots(2, (num_methods + 2) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # GT
    im_gt = axes[0].imshow(gt_img, origin='lower', extent=extent, cmap='jet', vmin=vmin_global, vmax=vmax_global)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    for i in range(0, len(sim_data.beams)):
        (x0, y0), (x1, y1) = sim_data.beams[i]
        axes[0].plot([x0, x1], [y0, y1], 'w-', alpha=0.3, linewidth=1.0)
    
    # Plot reconstructions
    for idx, (name, img) in enumerate(reconstructions.items(), start=1):
        rmse = np.sqrt(np.mean((img - gt_img)**2))
        data_range = gt_img.max() - gt_img.min()
        ssim_val = ssim(gt_img, img, data_range=data_range)
        
        t_setup = execution_times[name]['setup']
        t_recon = execution_times[name]['recon']

        # Multi-line title for clarity
        title = f"{name}\nRMSE: {rmse:.4f} | SSIM: {ssim_val:.4f}\nSetup: {t_setup:.2f}s | Recon: {t_recon:.2f}s"
        print(f"{name:<20}: RMSE = {rmse:.4f} | SSIM = {ssim_val:.4f} | Setup = {t_setup:.2f}s | Recon = {t_recon:.2f}s")
            
        axes[idx].imshow(img, origin='lower', extent=extent, cmap='jet', vmin=vmin_global, vmax=vmax_global)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')
        
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    fig.colorbar(im_gt, ax=axes.tolist(), label="ppm", fraction=0.03, pad=0.05)
    
    # Save reconstructions plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_methods.png')
    plt.savefig(save_path, dpi=300)
    print(f"Reconstructions plot saved in: {save_path}")

    # --- FIGURE 2: ERROR MAPS ---
    print("\n--- Generating Error Maps ---")
    fig_err, axes_err = plt.subplots(2, (num_methods + 1) // 2, figsize=(16, 8))
    axes_err = axes_err.flatten()
    fig_err.suptitle("Spatial Error Distribution (Absolute Difference)", fontsize=18)

    # Compute absolute error maps and find the global maximum for a consistent color scale
    error_maps = {}
    global_max_err = 0.0
    for name, img in reconstructions.items():
        err_map = np.abs(img - gt_img)
        error_maps[name] = err_map
        if err_map.max() > global_max_err:
            global_max_err = err_map.max()

    # Plot each error map
    for idx, (name, err_map) in enumerate(error_maps.items()):
        # Using a colormap like 'hot' or 'Reds' is standard for error visualization
        # vmin=0 and vmax=global_max_err ensures all plots use the exact same color scale
        im_err = axes_err[idx].imshow(err_map, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=global_max_err)
        axes_err[idx].set_title(f"{name} Error", fontsize=14)
        axes_err[idx].axis('off')

    # Hide any unused subplots
    for i in range(len(error_maps), len(axes_err)):
        axes_err[i].axis('off')

    fig_err.colorbar(im_err, ax=axes_err.tolist(), label="Absolute Error (ppm)", fraction=0.03, pad=0.05)
    
    # Save error maps plot
    save_path_err = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_methods_errors.png')
    plt.savefig(save_path_err, dpi=300)
    print(f"Error maps plot saved in: {save_path_err}")
    
    plt.show()

if __name__ == "__main__":
    main()