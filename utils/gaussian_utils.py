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

    K = pos.shape[0]
    N = beams.shape[0]

    p_rays = beams[:, 0, :] # Starting points (N, 2)

    u_rays = beams[:, 1, :] - p_rays # u = E - S
    # Normalize rays' vectors
    u = u_rays / torch.norm(u_rays, dim=1, keepdim=True)

    u_vec = u.view(N, 1, 2, 1) # (N, 2) -> (N, 1, 2, 1)
    u_vec_T = u_vec.transpose(-1, -2) # (N, 2) -> (N, 1, 1, 2)

    # d = p - mu
    # p (N, 1, 2) - mu (1, K, 2) = d (N, K, 2)
    diff = p_rays.unsqueeze(1) - pos.unsqueeze(0)
    
    d_vec = diff.unsqueeze(-1) # (N, K, 2, 1)
    d_vec_T = d_vec.transpose(-1, -2) # (N, K, 1, 2)
    
    Sigma_inv_exp = cov_inv.unsqueeze(0) # (K, 2, 2) -> (1, K, 2, 2)

    # Sigma^-1 * u
    # (1, K, 2, 2) @ (N, 1, 2, 1) -> (N, K, 2, 1)
    Sig_u = torch.matmul(Sigma_inv_exp, u_vec)

    # Sigma^-1 * d
    # (1, K, 2, 2) @ (N, K, 2, 1) -> (N, K, 2, 1)
    Sig_d = torch.matmul(Sigma_inv_exp, d_vec)

    # A = 0.5 * u^T * (Sigma^-1 * u)
    # (N, 1, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1)
    A = 0.5 * torch.matmul(u_vec_T, Sig_u).squeeze(-1).squeeze(-1)

    # B = u^T * (Sigma^-1 * d)
    # (N, 1, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1)
    B = torch.matmul(u_vec_T, Sig_d).squeeze(-1).squeeze(-1)

    # C = 0.5 * d^T * (Sigma^-1 * d)
    # (N, K, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1)
    C = 0.5 * torch.matmul(d_vec_T, Sig_d).squeeze(-1).squeeze(-1)
    
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
    K = pos.shape[0]
    N = beams.shape[0]

    # Extract starting (S) and ending (E) points 
    start_rays = beams[:, 0, :] # (N, 2)
    end_rays = beams[:, 1, :]   # (N, 2)

    # Ray vector v = E - S
    v_rays = end_rays - start_rays
    ray_lengths = torch.norm(v_rays, dim=1) # (N,)

    v_vec = v_rays.view(N, 1, 2, 1) # (N, 2) -> (N, 1, 2, 1)
    v_vec_T = v_vec.transpose(-1, -2) # (N, 1, 1, 2)

    # d = S - mu
    # start_rays (N, 1, 2) - pos (1, K, 2) = d (N, K, 2)
    diff = start_rays.unsqueeze(1) - pos.unsqueeze(0)
    
    d_vec = diff.unsqueeze(-1) # (N, K, 2, 1)
    d_vec_T = d_vec.transpose(-1, -2) # (N, K, 1, 2)
    
    Sigma_inv_exp = cov_inv.unsqueeze(0) # (K, 2, 2) -> (1, K, 2, 2)

    # Sigma^-1 * v
    # (1, K, 2, 2) @ (N, 1, 2, 1) -> (N, K, 2, 1)
    Sig_v = torch.matmul(Sigma_inv_exp, v_vec)

    # Sigma^-1 * d
    # (1, K, 2, 2) @ (N, K, 2, 1) -> (N, K, 2, 1)
    Sig_d = torch.matmul(Sigma_inv_exp, d_vec)

    # A = 0.5 * v^T * (Sigma^-1 * v)
    # (N, 1, 1, 2) @ (N, K, 2, 1) -> (N, K, 1, 1) -> (N, K)
    A = 0.5 * torch.matmul(v_vec_T, Sig_v).squeeze(-1).squeeze(-1)
    A = torch.clamp(A, min=1e-8)

    # B = v^T * (Sigma^-1 * d)
    B = torch.matmul(v_vec_T, Sig_d).squeeze(-1).squeeze(-1) # (N, K)

    # C = 0.5 * d^T * (Sigma^-1 * d)
    C = 0.5 * torch.matmul(d_vec_T, Sig_d).squeeze(-1).squeeze(-1) # (N, K)
    
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