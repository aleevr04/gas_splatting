import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from tqdm import tqdm


#===============================
# A. ART - ITERATIVE METHOD FOR ILL POSED PROBLEMS WITH NO PRIOR INFORMATION
#===============================
def art(system_matrix, measurements, num_iterations, initial_guess=None, relaxation_factor=1.0):
    """
        The Algebraic Reconstruction Technique (ART) is a method used in tomographic reconstruction, 
        based on iteratively solving a system of linear equations derived from projection data. 
        It is particularly useful when the problem is sparse or underdetermined.

        Iterative Correction: Instead of solving all equations simultaneously, ART updates the solution 
        one equation at a time, adjusting the estimate incrementally.
    """
    num_voxels = system_matrix.shape[1]
    if initial_guess is None:
        reconstruction = np.zeros(num_voxels)
    else:
        reconstruction = initial_guess.copy()

    for iteration in tqdm(range(num_iterations), desc="ART"):
        for i in range(len(measurements)):
            row = system_matrix.getrow(i)
            if row.nnz > 0:
                projection = measurements[i]
                predicted_projection = row.dot(reconstruction)
                error = projection - predicted_projection
                update = relaxation_factor * error * row.toarray().flatten() / (row.power(2).sum() + 1e-6)
                reconstruction += update

                # Enforce non-negativity
                reconstruction[reconstruction < 0] = 0

    return reconstruction



#===============================
# B. TIKHONOV - L2 norm regularization, penalizing solutions with high gas concentrations (stability)
#===============================
def tikhonov_iterative(system_matrix: sparse.csr_matrix, measurements: np.ndarray, alpha: float, num_iterations: int = 100, initial_guess=None):
    """
        Performs tomographic reconstruction using Tikhonov regularization (L2 regularization).
        It iteratively minimizes the squared error between the measurements and the projected
        reconstruction, while also penalizing the squared magnitude of the reconstructed image
        to promote stability and reduce noise. The 'alpha' parameter controls the strength of
        the regularization. An iterative gradient descent approach is used here (Landweber).
    """
    num_voxels = system_matrix.shape[1]
    if initial_guess is None:
        reconstruction = np.zeros(num_voxels)
    else:
        reconstruction = initial_guess.copy()

    # For direct solution (non-iterative), we can use:
    # ATA = system_matrix.transpose().dot(system_matrix)
    # L = sparse.identity(num_voxels) * alpha_reg
    # rhs = system_matrix.transpose().dot(measurements)
    # reconstruction = sparse.linalg.spsolve(ATA + L, rhs)

    # Implementing an iterative version (e.g., Landweber iteration with regularization)
    if num_iterations is None:
        num_iterations = 100
    learning_rate = 0.001

    for iteration in tqdm(range(num_iterations), desc="Tikhonov"):
        residual = measurements - system_matrix.dot(reconstruction)
        gradient = system_matrix.transpose().dot(residual) - alpha * reconstruction
        # system_matrix.transpose().dot(residual): is the standard back-projection term, indicating how to update the reconstruction to better fit the measurements. 
        # - alpha * reconstruction: is the Tikhonov regularization term. This term acts as a penalty on the magnitude (L2 norm) of the solution.
        reconstruction += learning_rate * gradient
        reconstruction[reconstruction < 0] = 0  # Enforce non-negativity

    return reconstruction



def tikhonov_direct(system_matrix: sparse.csr_matrix, measurements: np.ndarray, alpha: float) -> np.ndarray:
    """
    Performs tomographic reconstruction using direct Tikhonov regularization (L2 regularization)
    by solving the regularized normal equations.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (A).
        measurements (np.ndarray): The projection measurements (y).
        alpha (float): The Tikhonov regularization parameter.

    Returns:
        np.ndarray: The reconstructed image.
    """
    num_voxels = system_matrix.shape[1]

    ATA = system_matrix.transpose().dot(system_matrix)
    L = sparse.identity(num_voxels) * alpha
    rhs = system_matrix.transpose().dot(measurements)
    reconstruction_flat = sparse.linalg.spsolve(ATA + L, rhs)

    return reconstruction_flat



#===============================
# C. LFD - Low First Derivative (Smoothness)
#===============================
def lfd(system_matrix: sparse.csr_matrix, measurements: np.ndarray, grid_size: tuple, alpha: float) -> np.ndarray:
    """
    Performs tomographic reconstruction with smoothness regularization, penalizing
    large first spatial derivatives (differences between neighboring pixels).
    Uses lsqr to solve the augmented system.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (A).
        measurements (np.ndarray): The projection measurements (y).
        beta (float): The regularization parameter controlling the strength of the
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

    return reconstruction_flat.reshape(grid_size[0], grid_size[1])



#===============================
# D. LSD - Low Second Derivative (Smoothness)
#===============================
def lsd(
    system_matrix: sparse.csr_matrix,
    measurements: np.ndarray,
    grid_size: tuple,
    alpha: float
) -> np.ndarray:
    """
    Performs tomographic reconstruction with second-order smoothness regularization,
    penalizing large second spatial derivatives in both horizontal and vertical
    directions.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (A).
        measurements (np.ndarray): The projection measurements (y).
        grid_size (tuple): The shape of the grid (rows, columns).
        alpha (float): The regularization parameter controlling the strength of the
                       second-order smoothness constraint.

    Returns:
        np.ndarray: The reconstructed flat array of gas concentrations.
    """
    rows, cols = grid_size
    num_pixels = system_matrix.shape[1]

    # Create second-order difference operators (Laplacian-like)
    Dx2 = sparse.dok_matrix((num_pixels, num_pixels))
    Dy2 = sparse.dok_matrix((num_pixels, num_pixels))

    # Iterate through each pixel to build the difference operators
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c  # Flattened index of the pixel

            # Horizontal second derivative approximation
            if c > 0 and c < cols - 1:
                # Central difference: f''(x) = f(x-1) - 2f(x) + f(x+1)
                Dx2[idx, idx - 1] = 1
                Dx2[idx, idx] = -2
                Dx2[idx, idx + 1] = 1
            elif c == 0 and cols > 1:
                # Forward difference at the left edge
                Dx2[idx, idx] = -1
                Dx2[idx, idx + 1] = 1
            elif c == cols - 1 and cols > 1:
                # Backward difference at the right edge
                Dx2[idx, idx] = 1
                Dx2[idx, idx - 1] = -1

            # Vertical second derivative approximation
            if r > 0 and r < rows - 1:
                # Central difference: f''(y) = f(y-1) - 2f(y) + f(y+1)
                Dy2[idx, idx - cols] = 1
                Dy2[idx, idx] = -2
                Dy2[idx, idx + cols] = 1
            elif r == 0 and rows > 1:
                # Forward difference at the top edge
                Dy2[idx, idx] = -1
                Dy2[idx, idx + cols] = 1
            elif r == rows - 1 and rows > 1:
                # Backward difference at the bottom edge
                Dy2[idx, idx] = 1
                Dy2[idx, idx - cols] = -1

    Dx2 = Dx2.tocsr()
    Dy2 = Dy2.tocsr()

    # Construct the augmented system matrix for Tikhonov regularization
    augmented_A = sparse.vstack([system_matrix, alpha * Dx2, alpha * Dy2], format='csr')

    # Construct the augmented measurement vector (zeros for the regularization parts)
    augmented_b = np.concatenate([measurements, np.zeros(Dx2.shape[0] + Dy2.shape[0])])

    # Solve the augmented system using lsqr
    reconstruction_flat = lsqr(augmented_A, augmented_b)[0]

    # Apply non-negativity constraint
    reconstruction_flat[reconstruction_flat < 0] = 0

    return reconstruction_flat.reshape(rows, cols)


#===============================
# E. LTD (LOW THIRD DERIVATIVE)
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

    # Componente de la tercera derivada para las filas (horizontal)
    # 
    # Esta matriz penaliza las variaciones bruscas en la dirección horizontal (a lo largo de las filas).
    # Se construye con bloques diagonales, uno para cada fila del grid.
    def create_d3_1d(length):
        if length < 4:
            return sparse.csr_matrix((0, length))
        return sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(length - 3, length), dtype=float)

    row_reg_blocks = [create_d3_1d(cols) for _ in range(rows)]
    D3_row = sparse.block_diag(row_reg_blocks)

    # Componente de la tercera derivada para las columnas (vertical)
    # 
    # Esta matriz penaliza las variaciones bruscas en la dirección vertical (a lo largo de las columnas).
    # La construcción correcta es más compleja. Se crea un operador D3 en el espacio 1D
    # y luego se 'replica' para aplicarlo a lo largo de las columnas del vector aplanado.
    D3_col = sparse.lil_matrix((cols * (rows - 3), num_voxels))
    for j in range(cols):
        for i in range(rows - 3):
            idx = i * cols + j
            # Stencil [-1, 3, -3, 1] aplicado a lo largo de la columna j
            D3_col[i * cols + j, idx] = -1
            D3_col[i * cols + j, idx + cols] = 3
            D3_col[i * cols + j, idx + 2 * cols] = -3
            D3_col[i * cols + j, idx + 3 * cols] = 1
    
    D3_col = D3_col.tocsr() # Convertir a CSR para un mejor rendimiento

    # Construcción del sistema de ecuaciones aumentado
    # Se añade la regularización como nuevas "ecuaciones" con peso alpha.
    augmented_matrix = sparse.vstack([system_matrix, alpha * D3_row, alpha * D3_col])
    
    # El vector aumentado tiene ceros para las ecuaciones de regularización,
    # ya que buscamos minimizar la tercera derivada a un valor cercano a cero.
    n_row_reg = D3_row.shape[0]
    n_col_reg = D3_col.shape[0]
    augmented_measurements = np.hstack([measurements, np.zeros(n_row_reg), np.zeros(n_col_reg)])

    # Resolver el sistema de mínimos cuadrados con el método iterativo lsqr.
    reconstructed_image_flat, istop, itn, normr = lsqr(augmented_matrix, augmented_measurements, iter_lim=500)[:4]

    # Aplicar una restricción de no negatividad para asegurar que la solución sea físicamente
    # plausible (los valores de atenuación no pueden ser negativos).
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

    # --- 1. Construcción de la matriz de regularización (sin cambios) ---
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

    # --- 2. Preparación de las matrices ponderadas y el vector de medidas ---
    if weights is None:
        # Si no se proporcionan pesos, usa un vector de unos.
        weights = np.ones(measurements.shape[0])
    
    # Crea una matriz diagonal a partir del vector de pesos.
    W = sparse.diags(weights)

    # Pondera la matriz del sistema y el vector de mediciones.
    weighted_system_matrix = W @ system_matrix
    weighted_measurements = W @ measurements
    
    # --- 3. Construcción del sistema de ecuaciones aumentado (con la matriz y medidas ponderadas) ---
    augmented_matrix = sparse.vstack([weighted_system_matrix, alpha * D3_row, alpha * D3_col])
    
    n_row_reg = D3_row.shape[0]
    n_col_reg = D3_col.shape[0]
    augmented_measurements = np.hstack([weighted_measurements, np.zeros(n_row_reg), np.zeros(n_col_reg)])

    # --- 4. Resolución del sistema y post-procesamiento ---
    reconstructed_image_flat, istop, itn, normr = lsqr(augmented_matrix, augmented_measurements, iter_lim=500)[:4]

    reconstructed_image_flat[reconstructed_image_flat < 0] = 0

    reconstructed_image = reconstructed_image_flat.reshape(rows, cols)

    return reconstructed_image


#===============================
# F. PDE Difusion Advection Regularization
#===============================
def domain_knowledge_regularized_landweber_iterative(system_matrix: sparse.csr_matrix, measurements: np.ndarray, alpha: float = 0.01, rho: float = 0.001, num_iterations: int = 100, image_shape=None, flow_field=None, diffusion_coefficient=None, initial_guess=None):
    """
    Performs tomographic reconstruction using the iterative Landweber method
    with domain knowledge regularization based on a gas dispersion PDE.

    Args:
        system_matrix (sparse.csr_matrix): The system matrix (A).
        measurements (np.ndarray): The projection measurements (y).
        alpha (float, optional): The relaxation parameter (omega) for the
                                 Landweber update. Defaults to 0.01.
        rho (float, optional): The domain knowledge regularization parameter.
                               Defaults to 0.001.
        num_iterations (int, optional): The number of iterations to perform. Defaults to 100.
        image_shape (tuple, optional): The shape of the image (rows, cols).
                                       Required for discretizing the PDE.
        flow_field (np.ndarray, optional): The velocity field for advection.
        diffusion_coefficient (float, optional): The diffusion coefficient.
        initial_guess (np.ndarray, optional): Initial guess for the reconstruction.
                                              Defaults to a zero array.

    Returns:
        np.ndarray: The reconstructed image (flattened).
    """
    num_voxels = system_matrix.shape[1]
    if image_shape is None:
        side = int(np.sqrt(num_voxels))
        image_shape = (side, side)

    if initial_guess is None:
        reconstruction = np.zeros(num_voxels, dtype=np.float32)
    else:
        reconstruction = initial_guess.astype(np.float32).copy()

    # 1. Discretize the Gas Dispersion PDE
    def get_discretized_pde_operator(shape, flow, diff_coeff):
        """
        Placeholder function to generate the sparse matrix operator for the
        discretized gas dispersion PDE based on the provided parameters.
        This needs to be implemented based on the specific PDE and
        discretization scheme from the Wiedemann et al. paper.
        """
        # Example: A simple diffusion operator (Laplacian) - REPLACE WITH ACTUAL PDE
        rows, cols = shape
        num_pixels = rows * cols
        L = csr_matrix((num_pixels, num_pixels))
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                L[idx, idx] -= 4 * diff_coeff  # Center

                if i > 0: L[idx, idx - cols] += diff_coeff  # Up
                if i < rows - 1: L[idx, idx + cols] += diff_coeff  # Down
                if j > 0: L[idx, idx - 1] += diff_coeff  # Left
                if j < cols - 1: L[idx, idx + 1] += diff_coeff  # Right
        return L

    PDE_operator = get_discretized_pde_operator(image_shape, flow_field, diffusion_coefficient)

    for iteration in tqdm(range(num_iterations), desc="PDE Reg. Landweber"):
        projection_estimate = system_matrix.dot(reconstruction)
        residual = measurements - projection_estimate
        back_projection = system_matrix.transpose().dot(residual)

        # 2. Calculate the PDE residual
        pde_residual = PDE_operator.dot(reconstruction)

        # 3. Formulate the regularization term (e.g., gradient of ||PDE(x)||^2)
        regularization_term = rho * PDE_operator.transpose().dot(pde_residual)

        # 4. Update the reconstruction
        reconstruction = reconstruction + alpha * (back_projection - regularization_term)

    return reconstruction