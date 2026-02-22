import torch
import numpy as np
import matplotlib.pyplot as plt

from config import (
    InitParams,
    SimulationParams,
    TrainParams,
    DensificationParams,
    parse_args_into_dataclasses
)
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.plot_utils import render_gaussian_map
from utils.tomo_utils import (
    generate_random_beams,
    generate_radial_beams,
    generate_gas_distribution,
    simulate_gas_integrals
)

# ==========================================
#              CONFIGURATION
# ==========================================
init_cfg, sim_cfg, train_cfg, densify_cfg = parse_args_into_dataclasses(
    InitParams, SimulationParams, TrainParams, DensificationParams
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
#              SIMULATION
# ==========================================
# ------ Generate Simulation Beams ------
print("Generating simulated beams...")

raw_beams = generate_random_beams((sim_cfg.map_size, sim_cfg.map_size), sim_cfg.num_beams // 2)
raw_beams += generate_radial_beams((sim_cfg.map_size, sim_cfg.map_size), sim_cfg.num_beams // 2)

p_list = []
u_list = []
for (start, end) in raw_beams:
    p = np.array(start)
    u = np.array(end) - np.array(start) 
    p_list.append(p)
    u_list.append(u)

p_rays = torch.tensor(np.array(p_list), dtype=torch.float32).to(DEVICE)
u_rays = torch.tensor(np.array(u_list), dtype=torch.float32).to(DEVICE)

# ------------ Generate GT -------------
print("Generating Ground Truth...")

img_gt = generate_gas_distribution((sim_cfg.grid_res, sim_cfg.grid_res), num_blobs=sim_cfg.num_blobs, gauss_filter=not sim_cfg.no_gauss_filter)

# Simulate integral measurements
cell_size = sim_cfg.map_size / sim_cfg.grid_res
measurements_list = simulate_gas_integrals(img_gt, raw_beams, cell_size)

y_true = torch.tensor(measurements_list, dtype=torch.float32, device=DEVICE)

# ====================================================
#                INITIALIZE GAUSSIANS
# ====================================================
print(f"Running Least Squares initialization (Grid {init_cfg.coarse_res}x{init_cfg.coarse_res})")

init_pos, init_concentration, init_std, img_coarse = lsqr_initialization(
    raw_beams, 
    y_true, 
    sim_cfg.map_size, 
    init_cfg.initial_gaussians,
    coarse_res=init_cfg.coarse_res
)

model = GasSplattingModel(init_cfg.initial_gaussians, sim_cfg.map_size, densify_cfg).to(DEVICE)
model.initialize_gaussians(
    init_pos.to(DEVICE), 
    init_concentration.to(DEVICE), 
    init_std
)
print("Model initialized")

# Visualize initial guess
plt.figure()
plt.title("Algebraic Initialization (Least Squares)")
plt.imshow(img_coarse, origin='lower', extent=(0, sim_cfg.map_size, 0, sim_cfg.map_size))
plt.scatter(init_pos[:,0], init_pos[:,1], c='r', marker='x', label='Peaks')
plt.legend()
plt.show()

# ==========================================
#       TRAINING LOOP (OPTIMIZATION)
# ==========================================
trainer = Trainer(model, train_cfg, densify_cfg)

print("Starting Gas Splatting training...")
loss_history = trainer.train(p_rays, u_rays, y_true)

# ==========================================
#              VISUALIZATION
# ==========================================

fig = plt.figure(figsize=(15, 5))
fig.suptitle(f"Initial Gaussians = {init_cfg.initial_gaussians}\nFinal Gaussians = {model.num_gaussians}\nBeams = {sim_cfg.num_beams}")

# 1. GT
plt.subplot(2, 3, 1)
plt.title(f"Ground Truth (Grid {sim_cfg.grid_res}x{sim_cfg.grid_res})")
plt.imshow(img_gt, origin='lower', extent=(0, sim_cfg.map_size, 0, sim_cfg.map_size), cmap='viridis')

# Beams
for i in range(0, len(raw_beams)):
    (x0, y0), (x1, y1) = raw_beams[i]
    plt.plot([x0, x1], [y0, y1], 'w-', alpha=0.2, linewidth=0.5)
plt.colorbar(label="ppm")

# 2. Reconstruction
# a. Gaussians
img_pred_gaussian = render_gaussian_map(model, sim_cfg.map_size, DEVICE, grid_res=100)
pos = model.get_pos().detach().cpu().numpy()

plt.subplot(2, 3, 2)
plt.title(f"GS Reconstruction")
plt.imshow(img_pred_gaussian, origin='lower', extent=(0, sim_cfg.map_size, 0, sim_cfg.map_size), cmap='viridis')
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5)
plt.colorbar(label="ppm")

# b. Grid
img_pred = render_gaussian_map(model, sim_cfg.map_size, DEVICE, sim_cfg.grid_res)

plt.subplot(2, 3, 3)
plt.title(f"GS Reconstruction (Grid)")
plt.imshow(img_pred, origin='lower', extent=(0, sim_cfg.map_size, 0, sim_cfg.map_size), cmap='viridis')
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5)
plt.colorbar(label="ppm")

# 3. Loss History
plt.subplot(2, 1, 2)
plt.title("Loss History")
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.show()