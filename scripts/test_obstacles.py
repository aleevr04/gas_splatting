import os
import sys
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString, Polygon
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.sim_utils import (
    generate_fractal_gas_distribution,
    generate_random_beams,
    simulate_gas_integrals,
    MeasurementBatch,
    SimulationData
)

def get_unblocked_beams(cfg: Config, obstacles: np.ndarray) -> list:
    """Generates random beams and filters out any that intersect with obstacles."""
    valid_beams = []
    cell_size = cfg.sim.cell_size
    
    # Pre-build obstacle polygons for intersection checking
    obs_rows, obs_cols = np.where(obstacles > 0.5)
    obs_polys = []
    for r, c in zip(obs_rows, obs_cols):
        x_min, x_max = c * cell_size, (c + 1) * cell_size
        y_min, y_max = r * cell_size, (r + 1) * cell_size
        obs_polys.append(Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]))
        
    print("Generating line-of-sight beams (filtering blocked paths)...")
    while len(valid_beams) < cfg.sim.num_beams:
        # Generate a batch of candidate beams
        candidate_beams = generate_random_beams(cfg.sim.map_size, cfg.sim.num_beams)
        
        for (x0, y0), (x1, y1) in candidate_beams:
            if len(valid_beams) >= cfg.sim.num_beams:
                break
                
            beam_line = LineString([(x0, y0), (x1, y1)])
            
            # Check if the beam intersects any obstacle polygon
            blocked = any(beam_line.intersects(poly) for poly in obs_polys)
            
            if not blocked:
                valid_beams.append(((x0, y0), (x1, y1)))
                
    return valid_beams

def create_toy_environment(cfg: Config) -> SimulationData:
    """Generates a map with a central obstacle and gas around it."""
    
    # 1. Setup Grid
    map_w, map_h = cfg.sim.map_size
    grid_w = int(map_w / cfg.sim.cell_size)
    grid_h = int(map_h / cfg.sim.cell_size)
    
    # 2. Create Obstacles (e.g., a square block in the middle)
    obstacles = np.zeros((grid_h, grid_w))
    obs_min_x, obs_max_x = int(grid_w * 0.3), int(grid_w * 0.7)
    obs_min_y, obs_max_y = int(grid_h * 0.3), int(grid_h * 0.7)
    obstacles[obs_min_y:obs_max_y, obs_min_x:obs_max_x] = 1.0
    
    # 3. Create Gas Distribution
    gas_gt = generate_fractal_gas_distribution(
        grid_size=(grid_h, grid_w), 
        scale_fraction=0.3, 
        center_bias=0.0
    )
    
    # Mask out the gas inside the obstacle
    gas_gt[obstacles > 0.5] = 0.0
    
    # 4. Generate Beams and Measurements
    beams_list = get_unblocked_beams(cfg, obstacles)
    print("Simulating gas integrals...")
    measurements_list = simulate_gas_integrals(gas_gt, beams_list, cfg.sim.cell_size)
    
    beams_tensor = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)
    y_true = torch.tensor(measurements_list, dtype=torch.float32, device=cfg.device)
    
    return SimulationData(
        ground_truth=gas_gt,
        batch=MeasurementBatch(beams=beams_tensor, measurements=y_true),
        y_true=y_true,
        obstacles=obstacles
    )

def run_test(test_name: str, cfg: Config, sim_data: SimulationData):
    print(f"\n--- Running Test: {test_name} ---")
    
    # --- Initialization ---
    print(f"Running Least Squares initialization...")
    t0 = time.time()
    model, init_pos, img_coarse = setup_gs_model(sim_data.batch, cfg)
    setup_time = time.time() - t0
    print(f"Model initialized with {model.num_gaussians} Gaussians in {setup_time:.3f}s")
    
    # --- Training ---
    trainer = Trainer(
        model=model, 
        cfg=cfg, 
        ground_truth=sim_data.ground_truth, 
        obstacles=sim_data.obstacles
    )
    
    batch_results = trainer.train(sim_data.batch)
    training_results = trainer.finish()
    
    # --- Rename the generated GIF ---
    default_gif_path = "plots/training_evolution.gif" 
    
    if os.path.exists(default_gif_path):
        # Clean up the test name for a safe filename 
        safe_test_name = test_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        new_gif_path = f"plots/training_{safe_test_name}.gif"
        
        # Remove the destination file if it already exists to avoid errors
        if os.path.exists(new_gif_path):
            os.remove(new_gif_path)
            
        os.rename(default_gif_path, new_gif_path)
        print(f"[{test_name}] Saved training GIF to {new_gif_path}")
    
    # Render final map to evaluate gas inside obstacles
    final_map = model.render_map(cfg.sim.cell_size)
    gas_in_obstacles = np.sum(final_map * sim_data.obstacles)
    
    print(f"[{test_name}] Setup Time: {setup_time:.3f}s | Training Time: {training_results.training_time:.3f}s")
    print(f"[{test_name}] Final Data Loss: {batch_results.final_loss:.5f}")
    print(f"[{test_name}] Total Gas Inside Obstacles: {gas_in_obstacles:.5f}")
    
    return final_map, gas_in_obstacles, training_results.training_time

def main():
    # 1. Initialize Configuration
    parser = ArgumentParser(description="Gas Splatting parameters")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg
    
    print(f"Using device: {cfg.device}")
    
    # Generate Environment
    sim_data = create_toy_environment(cfg)
    
    results = {}
    
    # --- TEST 1: Baseline (No Obstacle Penalty) ---
    cfg_baseline = copy.deepcopy(cfg)
    cfg_baseline.train.obstacle_lambda = 0.0
    map_base, gas_base, time_base = run_test("Baseline (No Penalty)", cfg_baseline, sim_data)
    results['Baseline'] = {'map': map_base, 'gas_in_obs': gas_base, 'time': time_base}
    
    # --- TEST 2: SDF Repulsion (Lambda = 0.1) ---
    cfg_sdf_low = copy.deepcopy(cfg)
    cfg_sdf_low.train.obstacle_lambda = 0.1
    map_sdf_low, gas_sdf_low, time_sdf_low = run_test("SDF Repulsion (Weight 0.1)", cfg_sdf_low, sim_data)
    results['SDF_Low'] = {'map': map_sdf_low, 'gas_in_obs': gas_sdf_low, 'time': time_sdf_low}
    
    # --- TEST 3: SDF Repulsion (Lambda = 0.5) ---
    cfg_sdf_high = copy.deepcopy(cfg)
    cfg_sdf_high.train.obstacle_lambda = 0.5
    map_sdf_high, gas_sdf_high, time_sdf_high = run_test("SDF Repulsion (Weight 0.5)", cfg_sdf_high, sim_data)
    results['SDF_High'] = {'map': map_sdf_high, 'gas_in_obs': gas_sdf_high, 'time': time_sdf_high}
    
    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    red_cmap = mcolors.ListedColormap(['none', 'red'])
    
    # Ground Truth
    axes[0].imshow(sim_data.ground_truth, origin='lower')
    axes[0].imshow(sim_data.obstacles, cmap=red_cmap, origin='lower', interpolation='none', alpha=0.5)
    
    # Plot Beams (converted from meters to cell coordinates)
    beams_np = sim_data.batch.beams.cpu().numpy()
    cell_size = cfg.sim.cell_size
    for i in range(beams_np.shape[0]):
        x0 = beams_np[i, 0, 0] / cell_size
        y0 = beams_np[i, 0, 1] / cell_size
        x1 = beams_np[i, 1, 0] / cell_size
        y1 = beams_np[i, 1, 1] / cell_size
        axes[0].plot([x0, x1], [y0, y1], color='white', alpha=0.3, linewidth=0.8)
        
    axes[0].set_title("Ground Truth, Obstacle & Beams")
    
    # Baseline
    axes[1].imshow(results['Baseline']['map'], origin='lower', vmin=0, vmax=sim_data.ground_truth.max())
    axes[1].imshow(sim_data.obstacles, cmap=red_cmap, origin='lower', interpolation='none', alpha=0.5)
    axes[1].set_title(f"Baseline\nTime: {results['Baseline']['time']:.1f}s | Obs Gas: {results['Baseline']['gas_in_obs']:.1f}")
    
    # SDF Repulsion (Low Weight)
    axes[2].imshow(results['SDF_Low']['map'], origin='lower', vmin=0, vmax=sim_data.ground_truth.max())
    axes[2].imshow(sim_data.obstacles, cmap=red_cmap, origin='lower', interpolation='none', alpha=0.5)
    axes[2].set_title(f"SDF Repulsion (λ=0.1)\nTime: {results['SDF_Low']['time']:.1f}s | Obs Gas: {results['SDF_Low']['gas_in_obs']:.1f}")
    
    # SDF Repulsion (High Weight)
    axes[3].imshow(results['SDF_High']['map'], origin='lower', vmin=0, vmax=sim_data.ground_truth.max())
    axes[3].imshow(sim_data.obstacles, cmap=red_cmap, origin='lower', interpolation='none', alpha=0.5)
    axes[3].set_title(f"SDF Repulsion (λ=0.5)\nTime: {results['SDF_High']['time']:.1f}s | Obs Gas: {results['SDF_High']['gas_in_obs']:.1f}")
    
    # Set the limits of the axes to prevent the plot from expanding past the image boundaries
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    for ax in axes:
        ax.set_xlim(-0.5, grid_w - 0.5)
        ax.set_ylim(-0.5, grid_h - 0.5)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()