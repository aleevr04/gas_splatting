import os
import sys
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.sim_utils import generate_simulation_data
from utils.init_utils import setup_gs_model
from utils.plot_utils import render_gaussian_map
from utils.data_utils import save_experiment_results

def rmse_loss(img_gt, img_pred):
    return np.sqrt(np.mean((img_pred - img_gt)**2))

def evaluate_single_seed(seed, base_cfg, methods):
    """
    Isolated function to evaluate a single seed for the split comparison.
    Runs in an independent worker process.
    """
    # CRITICAL! Prevent PyTorch/NumPy from crashing the CPU with too many threads
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"[Worker Process] -> Starting Seed: {seed}", flush=True)
    
    cfg = copy.deepcopy(base_cfg)
    cfg.sim.seed = seed
    
    # --- WARM-UP (Prevents Cold Start Penalty) ---
    _warmup_cfg = copy.deepcopy(cfg)
    _warmup_cfg.train.iterations = 2 
    _warmup_sim = generate_simulation_data(_warmup_cfg)
    _model, _, _ = setup_gs_model(_warmup_sim, _warmup_cfg)
    _trainer = Trainer(_model, _warmup_cfg)
    _trainer.train(_warmup_sim)
    # ---------------------------------------------
    
    # Real data generation for this seed
    sim_data = generate_simulation_data(cfg)
    gt_img = sim_data.img_gt
    
    local_results = {
        "rmse": {},
        "ssim": {},
        "gaussians": {},
        "time": {}
    }
    
    for method in methods:
        test_cfg = copy.deepcopy(cfg)
        test_cfg.densify.long_axis_split = (method == "Long-Axis Split")
        
        # Setup model
        t_start = time.time()
        model, _, _ = setup_gs_model(sim_data, test_cfg)
        
        # Train
        trainer = Trainer(model, test_cfg)
        trainer.train(sim_data)
        elapsed_time = time.time() - t_start
        
        # Evaluate
        gs_img = render_gaussian_map(model, test_cfg.sim.map_size, test_cfg.device, cell_size=test_cfg.sim.cell_size)
        
        rmse = rmse_loss(gt_img, gs_img)
        data_range = gt_img.max() - gt_img.min()
        ssim_val = ssim(gt_img, gs_img, data_range=data_range)
        
        local_results["rmse"][method] = rmse
        local_results["ssim"][method] = ssim_val
        local_results["gaussians"][method] = model.num_gaussians
        local_results["time"][method] = elapsed_time

    print(f"[Worker Process] -> Completed Seed: {seed}", flush=True)
    return seed, local_results

def plot_split_comparison(methods, results, save_path):
    """
    Generates a 2x2 bar chart to compare A/B testing methods directly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(6, 6)) 
    
    axes = axes.flatten() 

    metrics = [
        {"key": "rmse", "title": "RMSE", "ax": axes[0], "color": "skyblue"},
        {"key": "ssim", "title": "SSIM", "ax": axes[1], "color": "lightgreen"},
        {"key": "gaussians", "title": "Number of Gaussians", "ax": axes[2], "color": "salmon"},
        {"key": "time", "title": "Total Time (s)", "ax": axes[3], "color": "mediumpurple"}
    ]

    x_pos = np.arange(len(methods))

    for metric in metrics:
        ax = metric["ax"]
        means = [np.mean(results[metric["key"]][m]) for m in methods]
        stds = [np.std(results[metric["key"]][m]) for m in methods]

        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, color=metric["color"], capsize=10, edgecolor='black')
        
        ax.set_title(metric["title"], pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # Add values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02 * max(means), 
                    f'{yval:.3f}' if "rmse" in metric["key"] or "ssim" in metric["key"] else f'{yval:.1f}', 
                    ha='center', va='bottom')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n[+] Comparison plot saved in: {save_path}")
    plt.close(fig)

def main():
    parser = ArgumentParser(description="Compare Original vs Long-Axis Split")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg
    
    # Generate random reproducible seeds for rigorous testing
    num_seeds = 30
    seeds = np.random.randint(0, 100000, size=num_seeds).tolist()
    
    methods = ["Original Split", "Long-Axis Split"]
    
    # Global structures
    results = {
        "rmse": {m: [] for m in methods},
        "ssim": {m: [] for m in methods},
        "gaussians": {m: [] for m in methods},
        "time": {m: [] for m in methods}
    }
    
    print(f"Starting experiment: comparing splitting methods across {len(seeds)} seeds.")
    
    # --- Parallelization ---
    max_workers=4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_seed, seed, cfg, methods): seed 
            for seed in seeds
        }
        
        for future in as_completed(futures):
            seed, local_results = future.result()
            
            for m in methods:
                results["rmse"][m].append(local_results["rmse"][m])
                results["ssim"][m].append(local_results["ssim"][m])
                results["gaussians"][m].append(local_results["gaussians"][m])
                results["time"][m].append(local_results["time"][m])

    # --- Print Summary ---
    print("\n--- Final Results (Averaged across seeds) ---")
    for method in methods:
        avg_rmse = np.mean(results["rmse"][method])
        avg_ssim = np.mean(results["ssim"][method])
        avg_gaussians = np.mean(results["gaussians"][method])
        avg_time = np.mean(results["time"][method])
        print(f"{method}:")
        print(f"  Avg RMSE:      {avg_rmse:.5f}")
        print(f"  Avg SSIM:      {avg_ssim:.4f}")
        print(f"  Avg Gaussians: {avg_gaussians:.1f}")
        print(f"  Avg Time:      {avg_time:.2f}s\n")

    # --- Save Data ---
    metadata = {
        "experiment_name": "split_methods_comparison",
        "methods": methods,
        "seeds": seeds,
        "map_size": cfg.sim.map_size,
        "cell_size": cfg.sim.cell_size
    }
    save_experiment_results(metadata, results)

    # --- Generate Plots ---
    save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'split_comparison.png')
    plot_split_comparison(methods, results, save_path)

if __name__ == "__main__":
    main()