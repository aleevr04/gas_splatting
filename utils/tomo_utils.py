import numpy as np
import math
from shapely.geometry import LineString, Polygon
from tqdm import tqdm
from scipy.sparse import dok_matrix
from scipy.ndimage import gaussian_filter


#==================================
#       GEOMETRY FUNCITONS
#==================================
def xy2cell(pos_m: tuple, cell_size_m: float)->tuple:
    """
    Translates (x, y) coordinates in meters to (row, column) indices in a 2D array,
    Args:
        pos_m (tuple): The (x,y)-coordinate in meters.
        cell_size_m: The dimensions in m of a cell
    Returns:
        tuple: A tuple containing the (row, column) index as integers.
    """    
    column = int(pos_m[0] // cell_size_m)
    row = int(pos_m[1] // cell_size_m)
    return row, column

def cell2xy(cell_rc: tuple, cell_size_m: float)->tuple:
    """
    Translates (row, column)  cell coordinates to (x, y) coordinates in meters,
    Args:
        cell_rc(tuple): The (row,col)-coordinate in the 2D array.
        cell_size_m: The dimensions in m of a cell
    Returns:
        tuple: A tuple containing the (x,y) coordinates in m.
    """    
    x = cell_rc[1] * cell_size_m + cell_size_m/2
    y = cell_rc[0] * cell_size_m + cell_size_m/2
    return x,y


#==================================
#       GAS DISTRIBUTION FUNCTIONS
#==================================
def generate_gas_distribution(grid_size: tuple, num_blobs: int = 5, gauss_filter : bool = True, seed = None) -> np.ndarray:
    """
    Generates a grid gas distribution given the amount of gas sources (high concentration points).

    Args:
        grid_size (tuple): Grid dimensions (rows, columns).
        num_blobs (int): Amount of high concentration points. Default = 5.
        gauss_filter (bool): If True, a gaussian filter is applied to get a cloudy and more natural distribution. Default = True.
        seed (int): Random seed.

    Returns:
        np.ndarray: 2D concentration map"
    """
    if seed is not None:
        np.random.seed(seed)
    
    rows, cols = grid_size
    gas_map = np.zeros(grid_size)

    # 1. Colocar "semillas" de concentración aleatoria muy densas
    #    (Puntos de ruido blanco)
    for _ in range(num_blobs):
        r = np.random.randint(rows // 5, 4 * rows // 5)
        c = np.random.randint(cols // 5, 4 * cols // 5)
        gas_map[r, c] = np.random.uniform(5.0, 10.0)

    # 2. Aplicar un filtro gaussiano MUY fuerte para expandir estas semillas
    #    y fusionarlas en formas amorfas.
    #    sigma variable crea asimetría en las formas.
    if gauss_filter:
        sigma_r = rows / np.random.uniform(6, 12)
        sigma_c = cols / np.random.uniform(6, 12)
        gas_map = gaussian_filter(gas_map, sigma=(sigma_r, sigma_c))

    # 3. Añadir ruido de alta frecuencia para textura (imperfecciones)
    noise = np.random.rand(rows, cols)
    gas_map += noise * (gas_map.max() * 0.1)

    # 4. Normalizar y limpiar el fondo (Thresholding)
    #    Hacemos que las concentraciones bajas sean exactamente 0 para tener bordes definidos.
    threshold = gas_map.max() * 0.3
    gas_map[gas_map < threshold] = 0

    # 5. Re-normalizar al rango deseado (ej. max 1.0 ppm)
    if gas_map.max() > 0:
        gas_map = gas_map / gas_map.max()

    return gas_map

def gaussian_plume(x_coords, y_coords, source_x, source_y, sigma_x, sigma_y, amplitude=1.0):
    """
    Generates a 2D Gaussian plume over a grid of (x, y) coordinates.

    Args:
        x_coords (np.ndarray): 1D array of x-coordinates for the grid.
        y_coords (np.ndarray): 1D array of y-coordinates for the grid.
        source_x (float): The x-coordinate of the plume source.
        source_y (float): The y-coordinate of the plume source.
        sigma_x (float): The standard deviation in the x-direction (plume width).
        sigma_y (float): The standard deviation in the y-direction (plume length/spread).
        amplitude (float): The maximum concentration at the source.

    Returns:
        np.ndarray: A 2D NumPy array representing the Gaussian plume concentration
                    at each (x, y) coordinate.
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    exponent = -((X - source_x)**2 / (2 * sigma_x**2) + (Y - source_y)**2 / (2 * sigma_y**2))
    plume = amplitude * np.exp(exponent)
    return plume

def generate_gaussian_plume_array(array_shape, cell_size_m, source_location_meters, plume_sigmas_meters, amplitude=1.0):
    """
    Generates a 2D NumPy array representing a Gaussian plume based on array dimensions,
    cell resolution, source location in meters, and plume standard deviations in meters.

    Args:
        array_shape (tuple): A tuple (rows, cols) defining the shape of the output array.
        cell_size_m (float): The size of each cell in meters.
        source_location_meters (tuple): A tuple (x_meter, y_meter) representing the
                                       source location in the geometric coordinate system
                                       (bottom-left origin assumed for calculations).
        plume_sigmas_meters (tuple): A tuple (sigma_x_meter, sigma_y_meter) representing
                                     the standard deviations of the plume in meters.
        amplitude (float): The maximum concentration at the source.

    Returns:
        np.ndarray: A 2D NumPy array representing the Gaussian plume concentration.
    """
    rows, cols = array_shape
    x_coords_meter = np.arange(0, cols * cell_size_m, cell_size_m) + cell_size_m / 2
    y_coords_meter = np.arange(0, rows * cell_size_m, cell_size_m) + cell_size_m / 2

    source_x_meter, source_y_meter = source_location_meters
    sigma_x_meter, sigma_y_meter = plume_sigmas_meters

    plume_data = gaussian_plume(x_coords_meter, y_coords_meter,
                                 source_x_meter, source_y_meter,
                                 sigma_x_meter / cell_size_m, sigma_y_meter / cell_size_m,
                                 amplitude)

    return plume_data


#==================================
#       BEAM RAYTRACING FUNCITONS
#==================================
def generate_radial_beams(map_size_m: tuple, num_beams: int):
    """Generates beams starting from bottom-left-corner (0,0) and bottom-right corners (X,0)
    with endpoints distributed homogeneously in angle along the grid boundaries.
    map_size = (X,Y) m
    """
    beams = []
    map_x, map_y = map_size_m

    # Beams starting from corner (0,0)
    num_beams_left = int(num_beams // 2)
    if num_beams_left > 0:
        angles_left = np.linspace(0, np.pi/2, num_beams_left)
        for angle in angles_left:
            x0, y0 = 0.0, 0.0  # Starting point in meters
            if angle == 0:
                x1, y1 = map_x, 0.0
            elif angle == np.pi / 2:
                x1, y1 = 0.0, map_y
            else:
                if angle <= math.atan2(map_y, map_x):
                    # End is at the right boundary
                    x1 = map_x
                    y1 = map_x * np.tan(angle)
                else:
                    # End is at the upper boundary
                    x1 = map_y / np.tan(angle)
                    y1 = map_y
            beams.append(((x0, y0), (x1, y1)))

    # Beams starting from corner (map_x, 0)
    num_beams_right = num_beams - num_beams_left
    if num_beams_right > 0:
        angles_right = np.linspace(0, np.pi/2, num_beams_right)
        for angle in angles_right:
            x0, y0 = map_x, 0.0  # Starting point in meters
            if angle == 0:
                x1, y1 = 0.0, 0.0
            elif angle == np.pi / 2:
                x1, y1 = map_x, map_y
            else:
                if angle <= math.atan2(map_y, map_x):
                    # End is at the left boundary
                    x1 = 0.0
                    y1 = map_x * np.tan(angle)
                else:
                    # End is at the upper boundary
                    x1 = map_x - (map_y / np.tan(angle))
                    y1 = map_y
            beams.append(((x0, y0), (x1, y1)))
    
    return beams


def generate_random_beams(map_size_m: tuple, num_beams: int):
    """Generates random TDLAS beams from the perimeter of a map in meters,
    storing start & end points."""
    beams = []
    map_x, map_y = map_size_m

    for _ in range(num_beams):
        # Choose start point on one of the four edges
        start_edge = np.random.choice(['left', 'right', 'bottom', 'top'])
        if start_edge == 'left':
            x0, y0 = 0.0, np.random.uniform(0, map_y)
        elif start_edge == 'right':
            x0, y0 = map_x, np.random.uniform(0, map_y)
        elif start_edge == 'bottom':
            x0, y0 = np.random.uniform(0, map_x), 0.0
        else: # 'top'
            x0, y0 = np.random.uniform(0, map_x), map_y

        # Choose end point on one of the other three edges
        end_edges = [edge for edge in ['left', 'right', 'bottom', 'top'] if edge != start_edge]
        end_edge = np.random.choice(end_edges)
        if end_edge == 'left':
            x1, y1 = 0.0, np.random.uniform(0, map_y)
        elif end_edge == 'right':
            x1, y1 = map_x, np.random.uniform(0, map_y)
        elif end_edge == 'bottom':
            x1, y1 = np.random.uniform(0, map_x), 0.0
        else: # 'top'
            x1, y1 = np.random.uniform(0, map_x), map_y

        beams.append(((x0, y0), (x1, y1)))
        
    return beams


def generate_horizontal_vertical_beams(map_size_m: tuple, num_beams: int):
    """Generates half horizontal and half vertical beams, evenly distributed
    across a map in meters."""
    beams = []
    map_x, map_y = map_size_m
    
    h_beams = int(num_beams*map_y // (map_x+map_y))

    # Horizontal Beams (Scanning left to right)
    if h_beams > 0:
        y_positions = np.linspace(0, map_y, h_beams, endpoint=False)
        for y in y_positions:
            beams.append(((0.0, y), (map_x, y)))
            
    # Vertical Beams (Scanning bottom to top)
    remaining_beams = num_beams - h_beams
    if remaining_beams > 0:
        x_positions = np.linspace(0, map_x, remaining_beams, endpoint=False)
        for x in x_positions:
            beams.append(((x, 0.0), (x, map_y)))

    return beams



#==================================
#   BEAM GAS INTEGRAL FUNCITONS
#==================================
def simulate_gas_integrals(
    gas_concentration_map: np.ndarray,
    beams: list[tuple[tuple[float, float], tuple[float, float]]],
    cell_dimensions_meters: float
) -> list[float]:
    """
    Simulates a Tunable Diode Laser Absorption Spectroscopy (TDLAS)
    raytracing measurement with improved path length calculation within cells.
    Note that we assume the beam reflects at the end and returns to the original position,
    where the detector is, so the path is twice.

    Args:
        gas_concentration_map (np.ndarray): A 2D numpy array representing
            the ground truth gas concentration in each cell (in ppm).
        beams (list[tuple[tuple[float, float], tuple[float, float]]]): A list of beams,
            where each beam is defined by a tuple containing the start and
            end coordinates (x, y) of the beam in meters.
        cell_dimensions_meters (float): The physical dimensions of each cell
            in the map, in meters.

    Returns:
        list[float]: A list of integral gas concentrations (in ppm*m),
            one for each input beam.
    """
    integral_concentrations = []
    rows, cols = gas_concentration_map.shape
    map_width = cols * cell_dimensions_meters
    map_height = rows * cell_dimensions_meters

    for (x0, y0), (x1, y1) in tqdm(beams, desc="TDLAS_simulation"):
        # Ensure beam coordinates are within the map boundaries
        if not (0 <= x0 <= map_width and 0 <= y0 <= map_height and
                0 <= x1 <= map_width and 0 <= y1 <= map_height):
            print(f"Warning: Beam ({x0}, {y0}) - ({x1}, {y1}) is out of map boundaries. Skipping.")
            integral_concentrations.append(0.0)
            continue

        beam_line = LineString([(x0, y0), (x1, y1)])
        weighted_concentration = 0.0

        # Determine the range of cells the beam might interact with
        min_x_cell = int(np.floor(min(x0, x1) / cell_dimensions_meters))
        max_x_cell = int(np.floor(max(x0, x1) / cell_dimensions_meters))
        min_y_cell = int(np.floor(min(y0, y1) / cell_dimensions_meters))
        max_y_cell = int(np.floor(max(y0, y1) / cell_dimensions_meters))

        # Clamp cell indices to the map dimensions
        min_x_cell = max(0, min_x_cell)
        max_x_cell = min(cols - 1, max_x_cell)
        min_y_cell = max(0, min_y_cell)
        max_y_cell = min(rows - 1, max_y_cell)

        # Iterate through all cells within the bounding box of the beam
        for r in range(min_y_cell, max_y_cell + 1):
            for c in range(min_x_cell, max_x_cell + 1):
                # Define the polygon representing the current cell in meters
                x_min = c * cell_dimensions_meters
                x_max = (c + 1) * cell_dimensions_meters
                y_min = r * cell_dimensions_meters
                y_max = (r + 1) * cell_dimensions_meters
                
                cell_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

                # Calculate the intersection between the beam line and the cell polygon
                intersection = beam_line.intersection(cell_polygon)

                if not intersection.is_empty:
                    if intersection.geom_type == 'LineString':
                        path_length_in_cell = intersection.length
                        concentration = gas_concentration_map[r, c]
                        weighted_concentration += (concentration * path_length_in_cell) * 2     # the beam pass twice (go and back)
                    
        integral_concentrations.append(weighted_concentration)

    return integral_concentrations



#==================================
#   SYSTEM MATRIX (ECUATIONS)
#==================================
def create_system_matrix_sparse(
    grid_size: tuple,
    beams: list[tuple[tuple[float, float], tuple[float, float]]],    
    cell_dimensions_meters: float
) -> dok_matrix:
    """
    Creates the system matrix A for TDLAS tomography, where A * x = b.
    x is the flattened gas concentration map, and b is the list of
    integral gas concentrations.

    This version operates directly on meter-based coordinates.

    Args:
        grid_size (tuple): The size of the grid (rows, columns).
        beams (list[tuple[tuple[float, float], tuple[float, float]]]): A list of beams,
            where each beam is defined by a tuple containing the start and
            end coordinates (x, y) of the beam in meters.
        cell_dimensions_meters (float): The physical dimensions of each cell
            in the map, in meters.

    Returns:
        dok_matrix: A sparse matrix A (Dictionary of Keys format).
    """
    rows, cols = grid_size
    num_cells = rows * cols
    num_beams = len(beams)
    A = dok_matrix((num_beams, num_cells), dtype=float)

    for i, ((x0, y0), (x1, y1)) in tqdm(enumerate(beams), desc="Building System Matrix", total=num_beams):
        # Create a LineString for the beam using meter coordinates
        beam_line = LineString([(x0, y0), (x1, y1)])

        # Determine the bounding box of cells the beam might intersect
        min_c = int(np.floor(min(x0, x1) / cell_dimensions_meters))
        max_c = int(np.floor(max(x0, x1) / cell_dimensions_meters))
        min_r = int(np.floor(min(y0, y1) / cell_dimensions_meters))
        max_r = int(np.floor(max(y0, y1) / cell_dimensions_meters))

        # Clamp cell indices to the grid boundaries
        min_c = max(0, min_c)
        max_c = min(cols - 1, max_c)
        min_r = max(0, min_r)
        max_r = min(rows - 1, max_r)

        # Iterate through all cells within the bounding box
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                # Define the polygon for the current cell in meters
                x_min = c * cell_dimensions_meters
                y_min = r * cell_dimensions_meters
                x_max = (c + 1) * cell_dimensions_meters
                y_max = (r + 1) * cell_dimensions_meters

                cell_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

                # Calculate the intersection between the beam line and the cell
                intersection = beam_line.intersection(cell_polygon)

                if not intersection.is_empty and intersection.geom_type == 'LineString':
                    path_length_in_cell = intersection.length
                    cell_index = r * cols + c  # Flattened index of the cell
                    A[i, cell_index] = path_length_in_cell * 2   # the beam pass twice (go and back)

    return A