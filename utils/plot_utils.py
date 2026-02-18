import torch

from gs_model import GasSplattingModel

def render_gaussian_map(gaussians: GasSplattingModel, map_size: float, device: torch.device, grid_res=100):
    """
    Turns gaussians into a 2D image (numpy matrix)
    """

    # Crear grid de coordenadas
    x = torch.linspace(0, map_size, grid_res, device=device)
    y = torch.linspace(0, map_size, grid_res, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_pos = torch.stack([X, Y], dim=-1) # (H, W, 2)

    final_img = torch.zeros((grid_res, grid_res), device=device)

    with torch.no_grad():
        pos = gaussians.get_pos()
        cov_inv = gaussians.get_covariance_inverse()
        concentration = gaussians.get_concentration()

        # Sumar contribución de cada gaussiana
        for k in range(gaussians.num_gaussians):
            mu = pos[k]
            sig_inv = cov_inv[k]
            c = concentration[k]
            
            d = grid_pos - mu
            d = d.unsqueeze(-1) 
            
            # Distancia de Mahalanobis: d^T * Sigma^-1 * d
            # Ajustamos dimensiones para matmul eficiente
            sig_inv_exp = sig_inv.view(1, 1, 2, 2)
            dist = torch.matmul(d.transpose(-1, -2), torch.matmul(sig_inv_exp, d)).squeeze()
            
            final_img += c * torch.exp(-0.5 * dist)

    # Retornar como numpy array (para tomo_utils)
    return final_img.detach().cpu().numpy()