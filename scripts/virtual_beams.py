import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sim_utils import generate_beams_from_obstacles

def main():
    occupancy_grid_dir = os.path.join(os.path.dirname(__file__), '..', 'ground_truth', 'occupancy_grids')
    occupancy_grid_file = os.path.join(occupancy_grid_dir, 'central_obstacle.csv')

    occupancy_grid = np.loadtxt(occupancy_grid_file, delimiter=',')
    
    cell_size = 0.1
    map_h = occupancy_grid.shape[0] * cell_size
    map_w = occupancy_grid.shape[1] * cell_size
    print(f"Map size: {map_w}x{map_h}")

    beams = generate_beams_from_obstacles(occupancy_grid, cell_size)
    print(f"Generated {len(beams)} beams")

    plt.imshow(occupancy_grid, extent=(0, map_w, 0, map_h))
    for i in range(len(beams)):
        (x0, y0), (x1, y1) = beams[i]
        plt.plot([x0, x1], [y0, y1], 'w-', alpha=0.3, linewidth=1.0)
    plt.show()

if __name__ == "__main__":
    main()