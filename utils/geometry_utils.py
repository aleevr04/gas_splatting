import numpy as np
from shapely.geometry import LineString, Polygon

def xy2cell(pos_m: tuple, cell_size_m: float) -> tuple[int, int]:
    """Translate (x, y) coordinates in meters to (row, column) indices."""
    column = int(pos_m[0] // cell_size_m)
    row = int(pos_m[1] // cell_size_m)
    return row, column

def cell2xy(cell_rc: tuple, cell_size_m: float) -> tuple[float, float]:
    """Translate (row, column) cell coordinates to cell-center (x, y) meters."""
    x = cell_rc[1] * cell_size_m + cell_size_m / 2
    y = cell_rc[0] * cell_size_m + cell_size_m / 2
    return x, y

def iter_ray_cell_intersections(
    beam: tuple,
    grid_size: tuple[int, int],
    cell_size_m: float,
):
    """Yield grid cells crossed by a beam and their path lengths in meters."""
    (x0, y0), (x1, y1) = beam
    rows, cols = grid_size
    map_width = cols * cell_size_m
    map_height = rows * cell_size_m

    if not (
        0 <= x0 <= map_width
        and 0 <= y0 <= map_height
        and 0 <= x1 <= map_width
        and 0 <= y1 <= map_height
    ):
        return

    beam_line = LineString([(x0, y0), (x1, y1)])

    min_col = max(0, int(np.floor(min(x0, x1) / cell_size_m)))
    max_col = min(cols - 1, int(np.floor(max(x0, x1) / cell_size_m)))
    min_row = max(0, int(np.floor(min(y0, y1) / cell_size_m)))
    max_row = min(rows - 1, int(np.floor(max(y0, y1) / cell_size_m)))

    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            x_min = col * cell_size_m
            y_min = row * cell_size_m
            x_max = (col + 1) * cell_size_m
            y_max = (row + 1) * cell_size_m
            cell_polygon = Polygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ])
            intersection = beam_line.intersection(cell_polygon)

            if not intersection.is_empty and intersection.geom_type == 'LineString':
                yield row, col, intersection.length
