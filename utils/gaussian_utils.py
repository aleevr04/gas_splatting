import torch

def inverse_softplus(x, beta=1):
    return torch.log(torch.exp(beta * x) - 1) / beta

def inverse_sigmoid(x, norm_factor=1.0):
    return torch.logit(x / norm_factor)

def compute_predicted_projections(pos, cov_inv, concentration, p_rays, u_rays):
    """
    Computes predicted projections, given gaussians' parameters and rays information.
    
    Args:
        pos (Tensor (K,2)): Gaussians' positions
        cov_inv (Tensor (K,2,2)): Gaussians' covariance inverse matrix
        concentration (Tensor (K,)): Gaussians' concentrations
        p_rays (Tensor (N,2)): Rays' points (any point in rays' trajectory)
        u_rays (Tensor (N,2)): Rays' direction vectors

    Returns:
        proj_pred (Tensor (N,)): Predicted projections
    """
    # Obtenemos N y K para usar en reshapes si fuera necesario
    K = pos.shape[0]
    N = p_rays.shape[0]

    # Normalizar dirección de rayos
    # (N, 2) -> (N, 2)
    u = u_rays / torch.norm(u_rays, dim=1, keepdim=True)
    
    # ---------------------------------------------------------
    # PREPARACIÓN DE FORMAS (RESHAPING)
    # Queremos que todo tenga forma compatible para operar (N, K, ...)
    # ---------------------------------------------------------

    # Expandimos inv_sigma para que sea compatible con N rayos
    # De (K, 2, 2) -> (1, K, 2, 2) -> PyTorch hará broadcast a (N, K, 2, 2)
    Sigma_inv_exp = cov_inv.unsqueeze(0) 

    # Convertimos u en vectores columna y expandimos para K gaussianas
    # De (N, 2) -> (N, 1, 2, 1)
    u_vec = u.view(N, 1, 2, 1)

    # Convertimos u en vectores fila (transpuesta) para multiplicar por la izquierda
    # De (N, 2) -> (N, 1, 1, 2)
    u_vec_T = u_vec.transpose(-1, -2)

    # Calculamos d = p - mu
    # p (N, 1, 2) - mu (1, K, 2) = d (N, K, 2)
    diff = p_rays.unsqueeze(1) - pos.unsqueeze(0)
    
    # Convertimos d en vector columna (N, K, 2, 1) y fila (N, K, 1, 2)
    d_vec = diff.unsqueeze(-1)
    d_vec_T = d_vec.transpose(-1, -2)

    # ---------------------------------------------------------
    # CÁLCULO DE A, B, C
    # Operación: vector_fila @ matriz @ vector_columna
    # ---------------------------------------------------------

    # Término intermedio común: Sigma^-1 * u
    # (1, K, 2, 2) @ (N, 1, 2, 1) -> Broadcasting -> (N, K, 2, 1)
    # Esto es: Para cada rayo y gaussiana, aplicamos la matriz a u
    Sig_u = torch.matmul(Sigma_inv_exp, u_vec)

    # Término intermedio común: Sigma^-1 * d
    # (1, K, 2, 2) @ (N, K, 2, 1) -> Broadcasting -> (N, K, 2, 1)
    Sig_d = torch.matmul(Sigma_inv_exp, d_vec)

    # --- Cálculo de A ---
    # A = 0.5 * u^T * (Sigma^-1 * u)
    # (N, 1, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1)
    # Hacemos squeeze para quitar las dimensiones 1 finales -> (N, K)
    A = 0.5 * torch.matmul(u_vec_T, Sig_u).squeeze(-1).squeeze(-1)

    # --- Cálculo de B ---
    # B = u^T * (Sigma^-1 * d)
    # (N, 1, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1)
    B = torch.matmul(u_vec_T, Sig_d).squeeze(-1).squeeze(-1)

    # --- Cálculo de C ---
    # C = 0.5 * d^T * (Sigma^-1 * d)
    # (N, K, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1)
    C = 0.5 * torch.matmul(d_vec_T, Sig_d).squeeze(-1).squeeze(-1)

    # ---------------------------------------------------------
    # INTEGRAL
    # ---------------------------------------------------------
    
    term_exp = (B**2) / (4 * A) - C
    
    # Calculamos la integral (matriz N x K)
    integral_matrix = torch.sqrt(torch.pi / A) * torch.exp(term_exp)

    proj_pred = torch.matmul(integral_matrix, concentration)

    return proj_pred