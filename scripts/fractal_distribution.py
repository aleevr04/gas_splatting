import os
import sys
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SimulationParams
from utils.sim_utils import generate_fractal_gas_distribution
from utils.plot_utils import set_publication_style

# Font global settings
set_publication_style()

# Parse args
parser = ArgumentParser(description="Generates a gas distribution using fractal noise. Only map-related options will have effect.")
parser.add_arguments(SimulationParams, dest="sim_params")
args = parser.parse_args()
sim_params: SimulationParams = args.sim_params

# Generate gas map
map_w, map_h = sim_params.map_size
grid_w = int(map_w / sim_params.cell_size)
grid_h = int(map_h / sim_params.cell_size)
plume_map = generate_fractal_gas_distribution(grid_size=(grid_h, grid_w))

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(plume_map, cmap='jet', origin='lower', extent=(0, map_w, 0, map_h))

# Color bar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('ppm', size=16)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout() 

# Save plot
save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'ground_truth.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[+] Fractal gas distribution saved in {save_path}")