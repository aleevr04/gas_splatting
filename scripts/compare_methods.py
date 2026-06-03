import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from simple_parsing import ArgumentParser
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.sim_utils import generate_simulation_data, create_system_matrix_sparse
from utils.data_utils import build_custom_real_scenario
from utils.plot_utils import set_publication_style
from utils.methods_registry import AVAILABLE_METHODS

def main():
    # --- Configuration ---
    parser = ArgumentParser(description="Compare visually Gas Splatting vs Traditional Methods")
    parser.add_arguments(dataclass=Config, dest="cfg")
    parser.add_argument('--real_scenario', dest='real_scenario', action='store_true', default=False, help="Use a real scenario instead of a simulated one.")
    parser.add_argument('--sim_beams', dest='sim_beams', action='store_true', default=False, help="Use the same beams' geometry as in simulated environments (radial and random beams). If this option is set, simulated sources must be used aswell.")
    parser.add_argument('--sim_gas', dest='sim_gas', action='store_true', default=False, help='Inject simulated sources in the environment.')
    args = parser.parse_args()
    cfg: Config = args.cfg

    set_publication_style()
    
    # --- Simulation data ---
    print(f"--- Generating Simulation Data ---")
    if args.real_scenario:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'real_data', 'full_sweep.json')
        sim_data = build_custom_real_scenario(
            real_data_path=data_path,
            cfg=cfg,
            use_sim_beams=args.sim_beams,
            use_sim_gas=args.sim_gas
        )
    else:
        sim_data = generate_simulation_data(cfg)
    
    measurements = sim_data.measurements.cpu().numpy()
    gt_img = sim_data.img_gt
    
    extent = (0, cfg.sim.map_size[0], 0, cfg.sim.map_size[1])
    grid_size = gt_img.shape
    
    methods = list(AVAILABLE_METHODS.keys())
    reconstructions = {}
    execution_times = {} 
    
    # --- Matrix Setup ---
    t_setup_start = time.time()
    system_matrix = create_system_matrix_sparse(grid_size, sim_data.beams.tolist(), cfg.sim.cell_size).tocsr()
    matrix_setup_time = time.time() - t_setup_start

    # --- Execution Loop ---
    print("\n--- Running Methods ---")
    
    for method_name in methods:
        print(f"[*] Executing: {method_name}...")
        method_info = AVAILABLE_METHODS[method_name]
        func = method_info["func"]
        
        res_img, total_time = func(
            system_matrix=system_matrix, 
            measurements=measurements, 
            sim_data=sim_data, 
            cfg=cfg, 
            setup_time=matrix_setup_time
        )

        reconstructions[method_name] = res_img
        execution_times[method_name] = total_time
    
    # --- Evaluation and Visualization ---
    print("\n--- Calculating Metrics & Generating Plots ---")
    
    num_methods = len(reconstructions)
    all_imgs = [gt_img] + list(reconstructions.values())
    
    # Compute global min and max for consistent color scaling across all plots
    vmin_global = 0
    vmax_global = max(img.max() for img in all_imgs)

    # ---------------------------------------------------------
    # FIGURE 1: RECONSTRUCTIONS
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, (num_methods + 2) // 2, figsize=(10, 6))
    axes = axes.flatten()
    
    # Plot Ground Truth
    im_gt = axes[0].imshow(gt_img, origin='lower', extent=extent, cmap='jet', vmin=vmin_global, vmax=vmax_global)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    # Overlay beams on Ground Truth
    for i in range(0, len(sim_data.beams)):
        (x0, y0), (x1, y1) = sim_data.beams[i]
        axes[0].plot([x0, x1], [y0, y1], 'w-', alpha=0.1, linewidth=0.5)
    
    # Plot method reconstructions
    data_range = gt_img.max() - gt_img.min()
    
    for idx, (name, img) in enumerate(reconstructions.items(), start=1):
        rmse = np.sqrt(np.mean((img - gt_img)**2))
        ssim_val = ssim(gt_img, img, data_range=data_range)
        t_total = execution_times[name]

        print(f"{name:<20}: RMSE = {rmse:.4f} | SSIM = {ssim_val:.4f} | Total Time = {t_total:.2f}s")
            
        axes[idx].imshow(img, origin='lower', extent=extent, cmap='jet', vmin=vmin_global, vmax=vmax_global)
        axes[idx].set_title(name)
        axes[idx].axis('off')
        
    # Hide unused subplots
    for i in range(len(reconstructions) + 1, len(axes)):
        axes[i].axis('off')

    fig.colorbar(im_gt, ax=axes.tolist(), label="ppm", fraction=0.03, pad=0.05)
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_methods.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n[+] Reconstructions plot saved in: {save_path}")

    # ---------------------------------------------------------
    # FIGURE 2: ERROR MAPS
    # ---------------------------------------------------------
    fig_err, axes_err = plt.subplots(2, (num_methods + 1) // 2, figsize=(10, 6))
    axes_err = axes_err.flatten()

    # Compute absolute error maps and find the global maximum
    error_maps = {}
    global_max_err = 0.0
    
    for name, img in reconstructions.items():
        err_map = np.abs(img - gt_img)
        error_maps[name] = err_map
        if err_map.max() > global_max_err:
            global_max_err = err_map.max()

    for idx, (name, err_map) in enumerate(error_maps.items()):
        im_err = axes_err[idx].imshow(err_map, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=global_max_err)
        axes_err[idx].set_title(name)
        axes_err[idx].axis('off')

    for i in range(len(error_maps), len(axes_err)):
        axes_err[i].axis('off')

    fig_err.colorbar(im_err, ax=axes_err.tolist(), label="Absolute Error (ppm)", fraction=0.03, pad=0.05)
    
    save_path_err = os.path.join(os.path.dirname(__file__), '..', 'plots', 'compare_methods_errors.png')
    plt.savefig(save_path_err)
    print(f"[+] Error maps plot saved in: {save_path_err}\n")

if __name__ == "__main__":
    main()