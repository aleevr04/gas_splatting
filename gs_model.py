import torch
import torch.nn as nn

from utils.gaussian_utils import (
    compute_predicted_projections,
    inverse_sigmoid,
    inverse_softplus
)

class GasSplattingModel(nn.Module):
    def __init__(self, num_gaussians, map_size):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.map_size = map_size
        
        # --- Parámetros entrenables ---
        self._pos = nn.Parameter(torch.rand(num_gaussians, 2))
        self._concentration = nn.Parameter(torch.rand(num_gaussians))
        self._scale = nn.Parameter(torch.zeros(num_gaussians, 2))
        self._rotation = nn.Parameter(torch.rand(num_gaussians) * 2*torch.pi)

    def initialize_parameters(self, pos, concentration, std):
        """
        Overwrite random parameters with an informed initialization.

        Args:
            pos (Tensor (K, 2)): Initial positions (x, y) in map coordinates.
            concentration (Tensor (K,) or float): Initial concentration/weight per gaussian.
            std (float or Tensor): Initial standard deviation (scalar, (K,) or (K,2)).
        """
        with torch.no_grad():
            # --- Positions: store as logit(normalized_pos) so get_pos() -> sigmoid(_pos)*map_size
            pos_t = torch.as_tensor(pos, dtype=self._pos.dtype, device=self._pos.device)
            if pos_t.shape != (self.num_gaussians, 2):
                raise ValueError(f"pos must have shape ({self.num_gaussians}, 2), got {tuple(pos_t.shape)}")
            
            self._pos.data.copy_(inverse_sigmoid(pos_t, self.map_size))

            # --- Concentration: inverse of softplus
            c_t = torch.as_tensor(concentration, dtype=self._concentration.dtype, device=self._concentration.device)
            if c_t.dim() == 0:
                c_t = c_t.expand(self.num_gaussians)
            elif c_t.numel() == self.num_gaussians and c_t.dim() == 1:
                pass
            else:
                raise ValueError(f"concentration must be scalar or shape ({self.num_gaussians},), got {tuple(c_t.shape)}")

            self._concentration.data.copy_(inverse_softplus(c_t))

            # --- Scale: we store log(scale) in _scale so get_scale() = exp(_scale)
            std_t = torch.as_tensor(std, dtype=self._scale.dtype, device=self._scale.device)
            if std_t.dim() == 0:
                scales = std_t * torch.ones((self.num_gaussians, 2), dtype=self._scale.dtype, device=self._scale.device)
            elif std_t.shape == (self.num_gaussians,):
                scales = std_t.unsqueeze(1).repeat(1, 2)
            elif std_t.shape == (self.num_gaussians, 2):
                scales = std_t
            else:
                raise ValueError(f"std must be scalar, ({self.num_gaussians},) or ({self.num_gaussians},2), got {tuple(std_t.shape)}")
            
            self._scale.data.copy_(torch.log(scales))

            # --- Rotation: initialize to zero (no rotation) by default
            self._rotation.data.zero_()

    def get_pos(self):
        return torch.sigmoid(self._pos) * self.map_size
    
    def get_scale(self):
        return torch.exp(self._scale)

    def get_concentration(self):
        return nn.functional.softplus(self._concentration)

    def get_rotation_matrix(self):
        thetas = self._rotation
        cos = torch.cos(thetas).unsqueeze(-1)  # (N, 1)
        sin = torch.sin(thetas).unsqueeze(-1)  # (N, 1)

        row1 = torch.cat([cos, -sin], dim=1)  # (N, 2)
        row2 = torch.cat([sin, cos], dim=1)   # (N, 2)

        R = torch.stack([row1, row2], dim=1)  # (N, 2, 2)

        return R
    
    def get_scale_square_inverse(self):
        scale_sq_inv = 1.0 / (self.get_scale()**2 + 1e-7)
        return torch.diag_embed(scale_sq_inv) # (N, 2, 2)

    def get_covariance_inverse(self):
        R = self.get_rotation_matrix()
        S_sq_inv = self.get_scale_square_inverse()
        
        # Sigma^-1 = R * S^-2 * R^T
        covariance_inverse = torch.bmm(R, torch.bmm(S_sq_inv, R.transpose(1, 2)))
        
        return covariance_inverse

    def forward(self, p_rays, u_rays):
        pos = self.get_pos()
        covariance_inverse = self.get_covariance_inverse()
        concentration = self.get_concentration()
        
        return compute_predicted_projections(pos, covariance_inverse, concentration, p_rays, u_rays)
