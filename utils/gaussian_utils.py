import torch

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

def inverse_sigmoid(x, norm_factor=1.0):
    return torch.logit(x / norm_factor)

def compute_integral(pos, cov_inv, concentration, beams):
    """
    Computes predicted projections, given gaussians' parameters and rays information.
    
    Args:
        pos (Tensor (K,2)): Gaussians' positions (mu)
        cov_inv (Tensor (K,2,2)): Gaussians' covariance inverse matrix (Sigma^-1)
        concentration (Tensor (K,)): Gaussians' concentrations (rho)
        beams (Tensor (N, 2, 2)): Rays defined by pairs of [start_point, end_point]

    Returns:
        y_pred (Tensor (N,)): Predicted projections
    """

    p_rays = beams[:, 0, :] # Starting points (N, 2)

    u_rays = beams[:, 1, :] - p_rays # u = E - S
    # Normalize rays' vectors
    u = u_rays / torch.norm(u_rays, dim=1, keepdim=True)

    # d = p - mu
    # p (N, 1, 2) - mu (1, K, 2) = d (N, K, 2)
    diff = p_rays.unsqueeze(1) - pos.unsqueeze(0)

    # A = 0.5 * u^T * Sigma^-1 * u
    A = 0.5 * torch.einsum('ni,kij,nj->nk', u, cov_inv, u)
    A = torch.clamp(A, min=1e-8)

    # B = u^T * Sigma^-1 * d
    B = torch.einsum('ni,kij,nkj->nk', u, cov_inv, diff)

    # C = 0.5 * d^T * Sigma^-1 * d
    C = 0.5 * torch.einsum('nki,kij,nkj->nk', diff, cov_inv, diff)
    
    term_exp = (B**2) / (4 * A) - C
    
    integral_matrix = torch.sqrt(torch.pi / A) * torch.exp(term_exp) # (N, K)

    y_pred = torch.matmul(integral_matrix, concentration) # (N,)

    return y_pred

def compute_definite_integral(pos, cov_inv, concentration, beams):
    """
    Computes predicted projections using the definite integral over line segments,
    given gaussians' parameters and rays defined by a single 'beams' tensor.
    
    Args:
        pos (Tensor (K,2)): Gaussians' positions (mu)
        cov_inv (Tensor (K,2,2)): Gaussians' covariance inverse matrix (Sigma^-1)
        concentration (Tensor (K,)): Gaussians' concentrations (rho)
        beams (Tensor (N, 2, 2)): Rays defined by pairs of [start_point, end_point]

    Returns:
        y_pred (Tensor (N,)): Predicted projections
    """

    # Extract starting (S) and ending (E) points 
    start_rays = beams[:, 0, :] # (N, 2)
    end_rays = beams[:, 1, :]   # (N, 2)

    # Ray vector v = E - S
    v_rays = end_rays - start_rays
    ray_lengths = torch.norm(v_rays, dim=1) # (N,)

    # d = S - mu
    # start_rays (N, 1, 2) - pos (1, K, 2) = d (N, K, 2)
    diff = start_rays.unsqueeze(1) - pos.unsqueeze(0)
    
    # A = 0.5 * v^T * Sigma^-1 * v
    A = 0.5 * torch.einsum('ni,kij,nj->nk', v_rays, cov_inv, v_rays)
    A = torch.clamp(A, min=1e-8)

    # B = v^T * Sigma^-1 * d
    B = torch.einsum('ni,kij,nkj->nk', v_rays, cov_inv, diff)

    # C = 0.5 * d^T * Sigma^-1 * d
    C = 0.5 * torch.einsum('nki,kij,nkj->nk', diff, cov_inv, diff)

    # exp(B^2 / 4A - C)
    term_exp = torch.exp((B**2) / (4 * A) - C)
    
    # Integration limits after variable change u = sqrt(A) * (t + B/(2*A))
    sqrt_A = torch.sqrt(A)
    u0 = B / (2 * sqrt_A)
    u1 = sqrt_A + u0
    
    # erf(u1) - erf(u0)
    erf_term = torch.erf(u1) - torch.erf(u0)
    
    # Integral from t=0 to t=1
    # sqrt(pi / 4A) = sqrt(pi) / (2 * sqrt_A)
    integral_t = (torch.sqrt(torch.tensor(torch.pi)) / (2 * sqrt_A)) * term_exp * erf_term # (N, K)

    # Multiply by beam length
    integral_matrix = integral_t * ray_lengths.unsqueeze(1) # (N, K) * (N, 1) -> (N, K)

    # Multiply by Gaussian concentration (rho)
    y_pred = torch.matmul(integral_matrix, concentration) # (N,)

    return y_pred