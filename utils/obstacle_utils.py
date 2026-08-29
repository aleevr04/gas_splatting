import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


SCENARIOS = (
    "none",
    "small_square",
    "large_square",
    "vertical_wall",
    "two_blocks",
    "l_shape",
    "u_shape",
    "circle",
    "boundary_wall",
)


def add_rectangle(
    obstacles: np.ndarray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> None:
    """Add a rectangle using coordinates normalized to the grid size."""
    grid_h, grid_w = obstacles.shape
    x0 = max(0, min(grid_w, int(x_min * grid_w)))
    x1 = max(0, min(grid_w, int(x_max * grid_w)))
    y0 = max(0, min(grid_h, int(y_min * grid_h)))
    y1 = max(0, min(grid_h, int(y_max * grid_h)))
    obstacles[y0:y1, x0:x1] = 1.0


def add_circle(
    obstacles: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> None:
    """Add a circle using normalized grid coordinates."""
    grid_h, grid_w = obstacles.shape
    y, x = np.indices(obstacles.shape)
    x = x / grid_w
    y = y / grid_h
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    obstacles[mask] = 1.0


def create_obstacle_scenario(name: str, grid_shape: tuple[int, int]) -> np.ndarray:
    """Create a reproducible binary obstacle mask for a named scenario."""
    obstacles = np.zeros(grid_shape, dtype=np.float32)

    if name == "none":
        return obstacles
    if name == "small_square":
        add_rectangle(obstacles, 0.40, 0.40, 0.60, 0.60)
    elif name == "large_square":
        add_rectangle(obstacles, 0.25, 0.25, 0.75, 0.75)
    elif name == "vertical_wall":
        add_rectangle(obstacles, 0.48, 0.10, 0.52, 0.90)
    elif name == "two_blocks":
        add_rectangle(obstacles, 0.15, 0.20, 0.35, 0.80)
        add_rectangle(obstacles, 0.65, 0.20, 0.85, 0.80)
    elif name == "l_shape":
        add_rectangle(obstacles, 0.20, 0.20, 0.30, 0.80)
        add_rectangle(obstacles, 0.20, 0.70, 0.70, 0.80)
    elif name == "u_shape":
        add_rectangle(obstacles, 0.20, 0.20, 0.30, 0.80)
        add_rectangle(obstacles, 0.70, 0.20, 0.80, 0.80)
        add_rectangle(obstacles, 0.20, 0.20, 0.80, 0.30)
    elif name == "circle":
        add_circle(obstacles, 0.50, 0.50, 0.25)
    elif name == "boundary_wall":
        add_rectangle(obstacles, 0.00, 0.35, 0.08, 0.65)
    else:
        raise ValueError(f"Unknown obstacle scenario: {name}")

    return obstacles


def obstacle_mask_to_geometry(
    obstacles: np.ndarray,
    cell_size: float,
):
    """Convert occupied grid cells into one unioned Shapely geometry."""
    polygons = []
    for row, col in zip(*np.where(obstacles > 0.5)):
        x_min, x_max = col * cell_size, (col + 1) * cell_size
        y_min, y_max = row * cell_size, (row + 1) * cell_size
        polygons.append(Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
        ]))
    return unary_union(polygons) if polygons else None
