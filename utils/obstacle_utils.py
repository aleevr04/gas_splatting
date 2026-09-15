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



def _steps_to_nearest_marked_cell(
    marked: np.ndarray,
    axis: int,
    unreachable: float,
) -> np.ndarray:
    """Number of cells to the nearest marked cell, either way along one axis.

    Works on cell indices rather than on distances. Replacing every unmarked
    cell by an index that loses any comparison, then taking a running minimum
    along the axis, leaves each cell holding the index of the closest marked
    cell at or after it; running a maximum the other way gives the closest one
    at or before it. Subtracting the cell's own index turns those into counts.

    Marked cells resolve to zero, and cells with no marked cell on a given side
    to `unreachable`, which must be at least as large as any distance the caller
    intends to keep.
    """
    length = marked.shape[axis]
    position = np.arange(length, dtype=np.float32)
    position = position.reshape((-1, 1) if axis == 0 else (1, -1))

    # Index of the closest marked cell at or after each cell, then at or before.
    # The fillers sit just outside the axis, so "none on this side" is explicit
    # rather than a large index that would be mistaken for a real one.
    after = np.flip(np.minimum.accumulate(
        np.flip(np.where(marked, position, length), axis), axis), axis)
    before = np.maximum.accumulate(np.where(marked, position, -1.0), axis)

    return np.minimum(
        np.where(after < length, after - position, unreachable),
        np.where(before >= 0.0, position - before, unreachable),
    )


def compute_wall_distance_field(
    obstacles: np.ndarray,
    cell_size: float,
    max_dist: float,
) -> np.ndarray:
    """Distance from every cell to the nearest obstacle along each axis (x and y).

    A signed distance field only exposes the closest surface, so a Gaussian in a
    corner can respect it while leaking through the perpendicular wall. Keeping
    one distance per axis constrains every surrounding wall independently.

    Each axis covers both of its directions: the penalty only ever uses the
    closer wall of the two, so they are merged here rather than at sample time,
    which halves both the stored field and the per-iteration sampling. There is
    no diagonal axis, because obstacles that are only visible diagonally are
    what the signed distance field alongside this one is for.

    The result is signed: inside an obstacle it holds minus the distance to the
    nearest free space. That ramp is what pushes a Gaussian that ended up inside
    an obstacle back out of it, so it must not be flattened.

    Args:
        obstacles: Binary occupancy grid (H, W).
        cell_size: Cell size in meters.
        max_dist: Distances are clipped here. Axes with no wall in sight
            saturate at this value, which keeps the field smooth enough for
            bilinear sampling.

    Returns:
        Array (2, H, W) holding the signed distance in meters along the
        horizontal and vertical axes, negative inside obstacles.
    """
    occupied = obstacles > 0.5
    free = ~occupied
    grid_h, grid_w = occupied.shape

    # Larger than any reachable step count, so "no wall ahead" saturates below
    unreachable = float(grid_h + grid_w)

    def signed_steps(axis: int) -> np.ndarray:
        """Steps to the nearest wall along one array axis, negative inside it."""
        # Outside, how far to the first obstacle. Inside, how far back out.
        # Cells are sampled at their center, hence the half step offset, which
        # keeps the ramp linear all the way across the wall face.
        return np.where(
            occupied,
            -(_steps_to_nearest_marked_cell(free, axis, unreachable) - 0.5),
            _steps_to_nearest_marked_cell(occupied, axis, unreachable) - 0.5,
        )

    # The x channel comes first so that the penalty can read the variance
    # projected on each axis straight off the diagonal of the covariance matrix
    distances = np.stack([
        signed_steps(axis=1),  # scan across columns: the x axis
        signed_steps(axis=0),  # scan down rows: the y axis
    ]) * cell_size

    return np.clip(distances, -max_dist, max_dist).astype(np.float32)
