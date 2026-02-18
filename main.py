import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
#       CONFIGURACIÓN DEL ESCENARIO
# ==========================================
MAP_SIZE = 20.0   # 10x10 metros
NUM_BEAMS = 30   # Número de rayos TDLAS
N_TRAIN_GAUSSIANS = 5 # Número de gaussianas (por ahora fijo)
GRID_RES = 20    # Resolución del grid
COARSE_RES = 10 # Resolución de la reconstrucción inicial (baja para rapidez)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# --- Generar Geometría de Rayos (TDLAS) ---
# 1. Generar lista de tuplas [(start, end), ...]
raw_beams = generate_random_beams((MAP_SIZE, MAP_SIZE), NUM_BEAMS // 2)
raw_beams += generate_radial_beams((MAP_SIZE, MAP_SIZE), NUM_BEAMS // 2)

# 2. Convertir a Tensores para el Modelo GS (Optimización)
p_list = []
u_list = []
for (start, end) in raw_beams:
    p = np.array(start)
    u = np.array(end) - np.array(start) 
    p_list.append(p)
    u_list.append(u)

p_rays = torch.tensor(np.array(p_list), dtype=torch.float32).to(DEVICE)
u_rays = torch.tensor(np.array(u_list), dtype=torch.float32).to(DEVICE)

# ==========================================
#  GENERAR GROUND TRUTH (REALIDAD SIMULADA)
# ==========================================
print("Generando Ground Truth...")

# Simulación discreta
img_gt = generate_gas_distribution((GRID_RES, GRID_RES), num_blobs=5, gauss_filter=True)

# Calcular medidas usando tomo_utils (física de celdas)
cell_size = MAP_SIZE / GRID_RES
measurements_list = simulate_gas_integrals(img_gt, raw_beams, cell_size)

# Convertir lista de medidas a Tensor para comparar con el modelo
y_true = torch.tensor(measurements_list, dtype=torch.float32, device=DEVICE)

# ====================================================
#    INICIALIZACIÓN "INTELIGENTE" DE LOS PARÁMETROS
# ====================================================
print(f"Ejecutando inicialización inteligente (Grid {COARSE_RES}x{COARSE_RES})")

init_pos, init_concentration, init_std, img_coarse = lsqr_initialization(
    raw_beams, 
    y_true, 
    MAP_SIZE, 
    N_TRAIN_GAUSSIANS,
    coarse_res=COARSE_RES
)

model = GasSplattingModel(N_TRAIN_GAUSSIANS, MAP_SIZE).to(DEVICE)
model.initialize_parameters(
    init_pos.to(DEVICE), 
    init_concentration.to(DEVICE), 
    init_std
)
print("Modelo inicializado con estrategia algebraica.")

# Visualizar la "pista" inicial
plt.figure()
plt.title("Inicialización Algebraica (Least Squares)")
plt.imshow(img_coarse, origin='lower', extent=(0, MAP_SIZE, 0, MAP_SIZE))
plt.scatter(init_pos[:,0], init_pos[:,1], c='r', marker='x', label='Centros')
plt.legend()
plt.show()

# ==========================================
#   BUCLE DE ENTRENAMIENTO (OPTIMIZACIÓN)
# ==========================================
print("Iniciando entrenamiento de Gaussian Splatting...")

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

loss_history = []
TARGET_LOSS = 1e-5
iterations = 1000

# Barra de progreso
pbar = tqdm(range(iterations), desc="Optimizando")
for it in pbar:
    optimizer.zero_grad()
    
    # 1. Forward Pass (El modelo siempre predice usando la fórmula analítica rápida)
    y_pred = model(p_rays, u_rays)
    
    # 2. Loss
    loss = torch.mean((y_pred - y_true)**2)
    
    # Regularización L1 (sparsity)
    l1_reg = 0.01 * torch.mean(model.get_concentration())
    total_loss = loss + l1_reg
    
    # 3. Backward
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    current_loss = loss.item()
    loss_history.append(current_loss)

    if it % 100 == 0:
        pbar.set_postfix({'loss': f'{loss.item():.5f}'})
    
    if current_loss < TARGET_LOSS:
        pbar.write(f"Entrenamiento terminado en iteración: {it}")
        break

pbar.close()

print(f"Loss final: {loss.item():.6f}")

# ==========================================
#              VISUALIZACIÓN
# ==========================================

fig = plt.figure(figsize=(15, 5))
fig.suptitle(f"Gaussianas Iniciales = {N_TRAIN_GAUSSIANS}\nMediciones = {NUM_BEAMS}")

# 1. Mapa Real (GT)
plt.subplot(2, 3, 1)
plt.title(f"Ground Truth (Grid {GRID_RES}x{GRID_RES})")
plt.imshow(img_gt, origin='lower', extent=(0, MAP_SIZE, 0, MAP_SIZE), cmap='viridis')

# Pintamos rayos
for i in range(0, len(raw_beams)):
    (x0, y0), (x1, y1) = raw_beams[i]
    plt.plot([x0, x1], [y0, y1], 'w-', alpha=0.2, linewidth=0.5)
plt.colorbar(label="ppm")

# 2. Mapa Reconstruido
# a. Gaussianas
img_pred_gaussian = render_gaussian_map(model, MAP_SIZE, DEVICE, grid_res=100)
pos = model.get_pos().detach().cpu().numpy()

plt.subplot(2, 3, 2)
plt.title(f"Reconstrucción GS")
plt.imshow(img_pred_gaussian, origin='lower', extent=(0, MAP_SIZE, 0, MAP_SIZE), cmap='viridis')
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5, label='Centros')
plt.colorbar(label="ppm")

# b. Gaussianas Discretizadas
img_pred = render_gaussian_map(model, MAP_SIZE, DEVICE, GRID_RES)

plt.subplot(2, 3, 3)
plt.title(f"Reconstrucción GS Discretizada")
plt.imshow(img_pred, origin='lower', extent=(0, MAP_SIZE, 0, MAP_SIZE), cmap='viridis')
plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10, marker='x', alpha=0.5, label='Centros')
plt.colorbar(label="ppm")

# 3. Curva de aprendizaje
plt.subplot(2, 1, 2)
plt.title("Convergencia del Loss")
plt.plot(loss_history)
plt.xlabel("Iteraciones")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.show()