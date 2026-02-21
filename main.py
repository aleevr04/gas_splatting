import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import InitParams, SimulationParams, TrainParams
from gs_model import GasSplattingModel
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
init_cfg = InitParams()
sim_cfg = SimulationParams()
train_cfg = TrainParams()

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

img_gt = generate_gas_distribution((sim_cfg.grid_res, sim_cfg.grid_res), num_blobs=sim_cfg.num_blobs, gauss_filter=sim_cfg.gauss_filter)

# Simulate integral measurements
cell_size = sim_cfg.grid_res / sim_cfg.grid_res
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

model = GasSplattingModel(init_cfg.initial_gaussians, sim_cfg.map_size).to(DEVICE)
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
print("Starting Gas Splatting traning...")

optimizer = optim.Adam([
    {'params': [model._pos], 'lr': train_cfg.pos_lr, 'name': 'pos'},
    {'params': [model._scale], 'lr': train_cfg.scale_lr, 'name': 'scale'},
    {'params': [model._rotation], 'lr': train_cfg.rotation_lr, 'name': 'rotation'},
    {'params': [model._concentration], 'lr': train_cfg.concentration_lr, 'name': 'concentration'},
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.lr_decay_step, gamma=train_cfg.lr_decay)

loss_history = []

# Barra de progreso
pbar = tqdm(range(train_cfg.iterations), desc="Training")
for it in pbar:
    optimizer.zero_grad()
    
    # Forward Pass (Compute gaussians integrals)
    y_pred = model(p_rays, u_rays)
    
    # Loss
    loss = torch.mean((y_pred - y_true)**2)
    
    # L1 Regularization (sparsity)
    l1_reg = 0.01 * torch.mean(model.get_concentration())
    total_loss = loss + l1_reg
    
    # Backward
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    current_loss = loss.item()
    loss_history.append(current_loss)

    if it % 100 == 0:
        pbar.set_postfix({'loss': f'{loss.item():.5f}'})
    
    if current_loss < train_cfg.target_loss:
        pbar.write(f"Training ended at iteration: {it}")
        break

pbar.close()

print(f"Final Loss: {loss.item():.6f}")

# ==========================================
#              VISUALIZATION
# ==========================================

fig = plt.figure(figsize=(15, 5))
fig.suptitle(f"Initial Gaussians = {init_cfg.initial_gaussians}\nBeams = {sim_cfg.num_beams}")

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
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5, label='Centros')
plt.colorbar(label="ppm")

# b. Grid
img_pred = render_gaussian_map(model, sim_cfg.map_size, DEVICE, sim_cfg.grid_res)

plt.subplot(2, 3, 3)
plt.title(f"GS Reconstruction (Grid)")
plt.imshow(img_pred, origin='lower', extent=(0, sim_cfg.map_size, 0, sim_cfg.map_size), cmap='viridis')
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5, label='Centros')
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