import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

#===============================
# SART - SIMULTANEOUS ALGEBRAIC RECONSTRUCTION TECHNIQUE
#===============================
def sart(system_matrix: sparse.csr_matrix, measurements: np.ndarray, grid_size: tuple , num_iterations: int = 50, initial_guess=None, relaxation_factor: float = 1.0):
    """
    Simultaneous Algebraic Reconstruction Technique (SART).
    
    Unlike ART (which updates the image ray by ray sequentially), 
    SART updates the image simultaneously by averaging the corrections of all 
    rays passing through a voxel. This drastically reduces the "salt and pepper" 
    noise typical of ART.
    
    The mathematical update in matrix form is:
    g^{(k+1)} = g^{(k)} + lambda * V^{-1} * A^T * W^{-1} * (p - A * g^{(k)})
    where:
      - W is the diagonal matrix with the sum of the rows of A (ray weights).
      - V is the diagonal matrix with the sum of the columns of A (voxel weights).

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (A).
        measurements (np.ndarray): The vector of measurements/projections (p).
        num_iterations (int): Number of iterations to perform.
        initial_guess (np.ndarray, optional): Initial guess for the image.
        relaxation_factor (float, optional): Relaxation parameter (lambda). 
                                             Typical values are between 0.1 and 1.0.

    Returns:
        np.ndarray: The reconstructed image reshaped as a square 2D array.
    """
    num_voxels = system_matrix.shape[1]
    
    if initial_guess is None:
        reconstruction = np.zeros(num_voxels, dtype=np.float32)
    else:
        reconstruction = initial_guess.astype(np.float32).copy()

    # 1. Precompute the row (W) and column (V) sums for SART normalization
    # A small epsilon (1e-8) is added to avoid division by zero in empty areas.
    eps = 1e-8
    
    # W_j = sum_i A_ji (Sum of weights along each ray)
    row_sums = np.array(system_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = eps
    
    # V_i = sum_j A_ji (Sum of weights along each voxel)
    col_sums = np.array(system_matrix.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = eps

    # 2. Iterative reconstruction loop
    for iteration in tqdm(range(num_iterations), desc="SART"):
        # Step A: Forward projection -> A * g
        predicted_projection = system_matrix.dot(reconstruction)
        
        # Step B: Calculate the projection error -> p - A * g
        error = measurements - predicted_projection
        
        # Step C: Normalize the error by row sums -> (p - A * g) / W
        # This corresponds to the normalized individual correction of each ray
        weighted_error = error / row_sums
        
        # Step D: Back-projection of the weighted error -> A^T * weighted_error
        # Distributes the error back to the voxels
        back_projection = system_matrix.transpose().dot(weighted_error)
        
        # Step E: Normalize by column sums and apply relaxation 
        # -> lambda * (A^T * weighted_error) / V
        update = relaxation_factor * (back_projection / col_sums)
        
        # Step F: Update the estimate
        reconstruction += update
        
        # Step G: Apply non-negativity constraint (physically necessary for gases)
        reconstruction[reconstruction < 0] = 0

    # Reshape the flattened 1D array into a 2D square matrix
    return reconstruction.reshape(grid_size)

#===============================
# RBF-COUPLED SART TOMOGRAPHY (G-CSRBF + SART)
# Reference: Gao, X. et al. (2022) "Radial Basis Function Coupled SART Method 
# for Dynamic LAS Tomography"
#===============================
def build_g_csrbf_matrix(grid_size: tuple, cell_size_m: float, R: float, 
                         beta: float, epsilon: float, center_step: int):
    """
    Builds the Phi matrix (sparse) of Gaussian Radial Basis Functions.
    Iterates only over the cells within the Bounding Box of radius R.
    """
    rows, cols = grid_size
    num_voxels = rows * cols
    
    center_rows = np.arange(0, rows, center_step)
    center_cols = np.arange(0, cols, center_step)
    num_centers = len(center_rows) * len(center_cols)
    
    row_indices = []
    col_indices = []
    data = []
    
    q_idx = 0
    r_pixels = int(np.ceil(R / cell_size_m))
    
    for cr in center_rows:
        for cc in center_cols:
            # Physical center of the RBF (meters)
            xq = cc * cell_size_m + cell_size_m / 2.0
            yq = cr * cell_size_m + cell_size_m / 2.0
            
            # Search limits (Bounding box)
            min_r = max(0, cr - r_pixels)
            max_r = min(rows - 1, cr + r_pixels)
            min_c = max(0, cc - r_pixels)
            max_c = min(cols - 1, cc + r_pixels)
            
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    xi = c * cell_size_m + cell_size_m / 2.0
                    yi = r * cell_size_m + cell_size_m / 2.0
                    
                    dist = np.hypot(xi - xq, yi - yq)
                    
                    if dist <= R:
                        # G-CSRBF (Equation 12)
                        val = ((1.0 - (dist / R)**2)**beta) * np.exp(-(epsilon * dist)**2)
                        pixel_idx = r * cols + c
                        
                        row_indices.append(pixel_idx)
                        col_indices.append(q_idx)
                        data.append(val)
                        
            q_idx += 1
            
    Phi = sparse.csr_matrix((data, (row_indices, col_indices)), 
                            shape=(num_voxels, num_centers), dtype=np.float32)
    return Phi

def rbf_sart(system_matrix: sparse.csr_matrix, 
             measurements: np.ndarray, 
             grid_size: tuple,
             cell_size_m: float,
             target_rbf_x: int = 15,
             overlap_factor: float = 2.5,
             beta: float = 1.5,
             epsilon_base: float = 2.0,
             num_iterations: int = 50, 
             relaxation_factor: float = 1.0):
    """
    RBF-SART Tomography (Gao et al. 2023).
    Dynamically scaled to be independent of the grid resolution.
    
    Args:
        system_matrix: Sensitivity matrix L (M rays x N cells).
        measurements: Absorbance vector p (M rays).
        grid_size: (rows, columns) of the image.
        cell_size_m: Size in meters of each cell.
        target_rbf_x: Approximate number of Gaussians across the width of the map.
        overlap_factor: How many times the radius covers the distance between two centers (2.0 to 3.0).
        beta: Shape of the compact support dome (1.0 to 1.5 recommended).
        epsilon_base: How much the Gaussian narrows before reaching the edge R (1.5 to 2.5).
        num_iterations: SART iterations.
        relaxation_factor: Relaxation factor lambda.
    """
    rows, cols = grid_size
    
    # ---------------------------------------------------------
    # 1. DYNAMIC CALCULATION OF PHYSICAL PARAMETERS
    # ---------------------------------------------------------
    # Calculate how many cells apart an RBF center should be placed
    center_step = max(1, cols // target_rbf_x)
    
    # Distance in meters between centers
    dist_centros_m = center_step * cell_size_m
    
    # The Radius (R) ensures that the bells always overlap in the same way
    R = overlap_factor * dist_centros_m
    
    # Epsilon is scaled with R so that the bell decay is consistent
    epsilon = epsilon_base / R

    # ---------------------------------------------------------
    # 2. MATRIX CONSTRUCTION
    # ---------------------------------------------------------
    Phi = build_g_csrbf_matrix(grid_size, cell_size_m, R, beta, epsilon, center_step)
    
    # Couple classical SART with the RBF domain (W = L * Phi)
    W = system_matrix.dot(Phi)
    num_centers = W.shape[1]
    
    # Precompute SART weights on the new W matrix
    eps_val = 1e-8
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = eps_val
    
    col_sums = np.array(W.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = eps_val

    # ---------------------------------------------------------
    # 3. SART ITERATION (OVER ALPHA COEFFICIENTS)
    # ---------------------------------------------------------
    alpha = np.zeros(num_centers, dtype=np.float32)
    
    for _ in tqdm(range(num_iterations), desc="RBF-SART Iterations"):
        predicted = W.dot(alpha)
        error = measurements - predicted
        
        weighted_error = error / row_sums
        back_proj = W.transpose().dot(weighted_error)
        
        alpha += relaxation_factor * (back_proj / col_sums)
        
        # Positivity constraint on coefficients (Guarantees image > 0)
        alpha[alpha < 0] = 0

    # ---------------------------------------------------------
    # 4. FINAL RECONSTRUCTION
    # ---------------------------------------------------------
    # Project coefficients to image space (Density = Phi * Alpha)
    reconstruction = Phi.dot(alpha)
    
    # Ensure non-negativity due to floating point rounding errors
    reconstruction[reconstruction < 0] = 0
    
    return reconstruction.reshape(rows, cols)

#===============================
# LFD - Low First Derivative (Smoothness)
#===============================
def lfd(system_matrix: sparse.csr_matrix, measurements: np.ndarray, grid_size: tuple, alpha: float) -> np.ndarray:
    """
    Performs tomographic reconstruction with smoothness regularization, penalizing
    large first spatial derivatives (differences between neighboring pixels).
    Uses lsqr to solve the augmented system.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (A).
        measurements (np.ndarray): The projection measurements (y).
        alpha (float): The regularization parameter controlling the strength of the
                       smoothness constraint.

    Returns:
        np.ndarray: The reconstructed image.
    """
    num_pixels = system_matrix.shape[1]
    image_shape = grid_size

    # Create difference operators for horizontal and vertical directions
    Dx = sparse.dok_matrix((image_shape[0] * (image_shape[1] - 1), num_pixels))
    Dy = sparse.dok_matrix(((image_shape[0] - 1) * image_shape[1], num_pixels))

    for i in range(image_shape[0]):
        for j in range(image_shape[1] - 1):
            idx = i * image_shape[1] + j
            Dx[i * (image_shape[1] - 1) + j, idx] = -1
            Dx[i * (image_shape[1] - 1) + j, idx + 1] = 1

    for i in range(image_shape[0] - 1):
        for j in range(image_shape[1]):
            idx = i * image_shape[1] + j
            Dy[i * image_shape[1] + j, idx] = -1
            Dy[i * image_shape[1] + j, idx + image_shape[1]] = 1

    Dx = Dx.tocsr()
    Dy = Dy.tocsr()

    # Construct the augmented system matrix
    augmented_A = sparse.vstack([system_matrix, alpha * Dx, alpha * Dy])

    # Construct the augmented measurement vector (zeros for the regularization parts)
    augmented_b = np.concatenate([measurements, np.zeros(Dx.shape[0] + Dy.shape[0])])

    # Solve the augmented system using lsqr
    reconstruction_flat = lsqr(augmented_A, augmented_b)[0]

    # Apply non-negativity constraint
    reconstruction_flat[reconstruction_flat < 0] = 0

    return reconstruction_flat.reshape(*grid_size)

#===============================
# LTD (LOW THIRD DERIVATIVE)
#===============================
def ltd(system_matrix: sparse.csr_matrix, measurements: np.ndarray, grid_size: tuple, alpha: float = 0.01) -> np.ndarray:
    """
    Implements the Low Third Derivative (LTD) method for tomographic reconstruction.
    This version corrects the D3_col matrix construction and provides improved explanations.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (projection matrix)
                                            of shape (num_measurements, num_voxels).
        measurements (np.ndarray): The measurement vector of shape (num_measurements).
        grid_size (tuple): The [rows, cols] of the image to reconstruct.
        alpha (float, optional): The regularization parameter. Defaults to 0.01.

    Returns:
        np.ndarray: The reconstructed image of shape (rows, cols).
    """
    rows, cols = grid_size
    num_voxels = rows * cols

    # Third derivative component for the rows (horizontal)
    # 
    # This matrix penalizes sharp variations in the horizontal direction (along the rows).
    # It is built with diagonal blocks, one for each row of the grid.
    def create_d3_1d(length):
        if length < 4:
            return sparse.csr_matrix((0, length))
        return sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(length - 3, length), dtype=float)

    row_reg_blocks = [create_d3_1d(cols) for _ in range(rows)]
    D3_row = sparse.block_diag(row_reg_blocks)

    # Third derivative component for the columns (vertical)
    # 
    # This matrix penalizes sharp variations in the vertical direction (along the columns).
    # The correct construction is more complex. A D3 operator is created in 1D space
    # and then 'replicated' to apply it along the columns of the flattened vector.
    D3_col = sparse.lil_matrix((cols * (rows - 3), num_voxels))
    for j in range(cols):
        for i in range(rows - 3):
            idx = i * cols + j
            # Stencil [-1, 3, -3, 1] applied along column j
            D3_col[i * cols + j, idx] = -1
            D3_col[i * cols + j, idx + cols] = 3
            D3_col[i * cols + j, idx + 2 * cols] = -3
            D3_col[i * cols + j, idx + 3 * cols] = 1
    
    D3_col = D3_col.tocsr() # Convert to CSR for better performance

    # Construction of the augmented system of equations
    # Regularization is added as new 'equations' with weight alpha.
    augmented_matrix = sparse.vstack([system_matrix, alpha * D3_row, alpha * D3_col])
    
    # The augmented vector has zeros for the regularization equations,
    # since we seek to minimize the third derivative to a value close to zero.
    n_row_reg = D3_row.shape[0]
    n_col_reg = D3_col.shape[0]
    augmented_measurements = np.hstack([measurements, np.zeros(n_row_reg), np.zeros(n_col_reg)])

    # Solve the least squares system with the iterative lsqr method.
    reconstructed_image_flat, istop, itn, normr = lsqr(augmented_matrix, augmented_measurements, iter_lim=500)[:4]

    # Apply a non-negativity constraint to ensure the solution is physically
    # plausible (attenuation values cannot be negative).
    reconstructed_image_flat[reconstructed_image_flat < 0] = 0

    reconstructed_image = reconstructed_image_flat.reshape(rows, cols)

    return reconstructed_image


def ltd_weighted(system_matrix: sparse.csr_matrix, measurements: np.ndarray, grid_size: tuple, alpha: float = 0.01, weights: np.ndarray = None) -> np.ndarray:
    """
    Implements the Low Third Derivative (LTD) method with weighted measurements.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix.
        measurements (np.ndarray): The measurement vector.
        grid_size (tuple): The [rows, cols] of the image.
        alpha (float, optional): The regularization parameter. Defaults to 0.01.
        weights (np.ndarray, optional): A vector of weights for each measurement.
                                        If None, all measurements are equally weighted.

    Returns:
        np.ndarray: The reconstructed image.
    """
    rows, cols = grid_size
    num_voxels = rows * cols

    # --- 1. Construction of the regularization matrix (unchanged) ---
    def create_d3_1d(length):
        if length < 4:
            return sparse.csr_matrix((0, length))
        return sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(length - 3, length), dtype=float)

    row_reg_blocks = [create_d3_1d(cols) for _ in range(rows)]
    D3_row = sparse.block_diag(row_reg_blocks)
    
    D3_col = sparse.lil_matrix((cols * (rows - 3), num_voxels))
    for j in range(cols):
        for i in range(rows - 3):
            idx = i * cols + j
            D3_col[i * cols + j, idx] = -1
            D3_col[i * cols + j, idx + cols] = 3
            D3_col[i * cols + j, idx + 2 * cols] = -3
            D3_col[i * cols + j, idx + 3 * cols] = 1
    D3_col = D3_col.tocsr()

    # --- 2. Preparation of weighted matrices and measurement vector ---
    if weights is None:
        # If no weights are provided, use a vector of ones.
        weights = np.ones(measurements.shape[0])
    
    # Creates a diagonal matrix from the weights vector.
    W = sparse.diags(weights)

    # Weights the system matrix and the measurement vector.
    weighted_system_matrix = W @ system_matrix
    weighted_measurements = W @ measurements
    
    # --- 3. Construction of the augmented system of equations (with weighted matrix and measurements) ---
    augmented_matrix = sparse.vstack([weighted_system_matrix, alpha * D3_row, alpha * D3_col])
    
    n_row_reg = D3_row.shape[0]
    n_col_reg = D3_col.shape[0]
    augmented_measurements = np.hstack([weighted_measurements, np.zeros(n_row_reg), np.zeros(n_col_reg)])

    # --- 4. System resolution and post-processing ---
    reconstructed_image_flat, istop, itn, normr = lsqr(augmented_matrix, augmented_measurements, iter_lim=500)[:4]

    reconstructed_image_flat[reconstructed_image_flat < 0] = 0

    reconstructed_image = reconstructed_image_flat.reshape(rows, cols)

    return reconstructed_image