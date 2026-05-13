import os
import sys
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.sim_utils import generate_fractal_gas_distribution
from utils.plot_utils import set_publication_style

# Font global settings
set_publication_style()

# Parse args
parser = ArgumentParser()
parser.add_arguments(Config, dest="cfg")
args = parser.parse_args()
cfg = args.cfg

# Generate gas map
map_w, map_h = cfg.sim.map_size
grid_w = int(map_w / cfg.sim.cell_size)
grid_h = int(map_h / cfg.sim.cell_size)
plume_map = generate_fractal_gas_distribution(grid_size=(grid_h, grid_w))

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(plume_map, cmap='jet', origin='lower')

# Color bar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('ppm', size=16)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout() 

# Save plot
save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'ground_truth.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()