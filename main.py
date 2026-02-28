import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

from config import Config
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.plot_utils import render_gaussian_map
from utils.sim_utils import generate_simulation_data

# ==========================================
#              CONFIGURATION
# ==========================================
parser = ArgumentParser(description="Gas Splatting parameters")
parser.add_arguments(Config, dest="cfg")
args = parser.parse_args()
cfg = args.cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
#              SIMULATION
# ==========================================

sim_data = generate_simulation_data(cfg, DEVICE)

# ====================================================
#                INITIALIZE GAUSSIANS
# ====================================================
print(f"Running Least Squares initialization (Grid {cfg.init.coarse_res}x{cfg.init.coarse_res})")

init_pos, init_concentration, init_std, img_coarse = lsqr_initialization(
    sim_data.beams, 
    sim_data.y_true, 
    cfg.sim.map_size, 
    num_gaussians=cfg.init.initial_gaussians,
    coarse_res=cfg.init.coarse_res
)
initial_gaussians = init_pos.shape[0]

model = GasSplattingModel(initial_gaussians, cfg).to(DEVICE)
model.initialize_gaussians(
    init_pos.to(DEVICE), 
    init_concentration.to(DEVICE), 
    init_std
)
print("Model initialized")

# Visualize initial guess
plt.figure()
plt.title("Algebraic Initialization (Least Squares)")
plt.imshow(img_coarse, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size))
plt.scatter(init_pos[:,0], init_pos[:,1], c='r', marker='x', label='Peaks')
plt.legend()
plt.show()

# ==========================================
#       TRAINING LOOP (OPTIMIZATION)
# ==========================================
trainer = Trainer(model, cfg)

print("Starting Gas Splatting training...")
loss_history = trainer.train(sim_data.p_rays, sim_data.u_rays, sim_data.y_true)
print(f"Loss: {loss_history[-1]:.6f}")

# ==========================================
#              VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(15, 5))
fig.suptitle(f"Initial Gaussians = {initial_gaussians}\nFinal Gaussians = {model.num_gaussians}\nBeams = {cfg.sim.num_beams}")

# 1. GT
plt.subplot(2, 3, 1)
plt.title(f"Ground Truth (Grid {cfg.sim.grid_res}x{cfg.sim.grid_res})")
plt.imshow(sim_data.img_gt, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size), cmap='viridis')

# Beams
for i in range(0, len(sim_data.beams)):
    (x0, y0), (x1, y1) = sim_data.beams[i]
    plt.plot([x0, x1], [y0, y1], 'w-', alpha=0.2, linewidth=0.5)
plt.colorbar(label="ppm")

# 2. Reconstruction
# a. Gaussians
img_pred_gaussian = render_gaussian_map(model, cfg.sim.map_size, DEVICE, grid_res=100)
pos = model.get_pos().detach().cpu().numpy()

plt.subplot(2, 3, 2)
plt.title(f"GS Reconstruction")
plt.imshow(img_pred_gaussian, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size), cmap='viridis')
plt.colorbar(label="ppm")
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5)

# b. Grid
img_pred = render_gaussian_map(model, cfg.sim.map_size, DEVICE, cfg.sim.grid_res)

plt.subplot(2, 3, 3)
plt.title(f"GS Reconstruction (Grid)")
plt.imshow(img_pred, origin='lower', extent=(0, cfg.sim.map_size, 0, cfg.sim.map_size), cmap='viridis')
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