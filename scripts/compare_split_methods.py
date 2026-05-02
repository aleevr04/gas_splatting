import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.sim_utils import generate_simulation_data
from utils.init_utils import setup_gs_model
from utils.plot_utils import render_gaussian_map

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def main():
    parser = ArgumentParser(description="Compare Original vs Long-Axis Split")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    base_cfg: Config = args.cfg
    
    seeds = [42, 100, 1234, 777, 999]
    methods = ["Original Split", "Long-Axis Split"]
    
    results_rmse = {m: [] for m in methods}
    results_gaussians = {m: [] for m in methods}
    results_time = {m: [] for m in methods}
    
    print(f"Starting experiment: comparing splitting methods across {len(seeds)} seeds.")
    
    for seed in seeds:
        print(f"\nEvaluating Seed: {seed}...")
        
        # Base Data Generation
        cfg = copy.deepcopy(base_cfg)
        cfg.sim.seed = seed
        sim_data = generate_simulation_data(cfg)
        gt_img = sim_data.img_gt
        
        for method in methods:
            test_cfg = copy.deepcopy(cfg)
            
            # Toggle the splitting behavior
            test_cfg.densify.long_axis_split = (method == "Long-Axis Split")
            
            t_start = time.time()
            model, _, _ = setup_gs_model(sim_data, test_cfg)
            
            trainer = Trainer(model, test_cfg)
            trainer.train(sim_data)
            elapsed_time = time.time() - t_start
            
            gs_img = render_gaussian_map(model, test_cfg.sim.map_size, test_cfg.device, cell_size=test_cfg.sim.cell_size)
            
            rmse = rmse_loss(gt_img, gs_img)
            
            results_rmse[method].append(rmse)
            results_gaussians[method].append(model.num_gaussians)
            results_time[method].append(elapsed_time)
            
            print(f"  -> {method}: RMSE={rmse:.4f} | Gaussians={model.num_gaussians} | Time={elapsed_time:.1f}s")
            
    print("\n--- Final Results (Averaged across seeds) ---")
    for method in methods:
        avg_rmse = np.mean(results_rmse[method])
        avg_gaussians = np.mean(results_gaussians[method])
        avg_time = np.mean(results_time[method])
        print(f"{method}:")
        print(f"  Avg RMSE:      {avg_rmse:.5f}")
        print(f"  Avg Gaussians: {avg_gaussians:.1f}")
        print(f"  Avg Time:      {avg_time:.2f}s\n")

if __name__ == "__main__":
    main()