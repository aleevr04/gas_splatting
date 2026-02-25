import torch
import torch.nn as nn
import math

from config import DensificationParams
from utils.gaussian_utils import (
    compute_predicted_projections,
    inverse_sigmoid,
    inverse_softplus
)

class GasSplattingModel(nn.Module):
    def __init__(self, num_gaussians, map_size, densify_cfg: DensificationParams):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.map_size = map_size
        self.densify_cfg = densify_cfg

        self.pos_grad_accum = torch.zeros((num_gaussians, 1))
        self.denom = torch.zeros((num_gaussians, 1))
        
        # --- Model parameters ---
        self._pos = nn.Parameter(torch.rand(num_gaussians, 2) * map_size)
        self._concentration = nn.Parameter(torch.rand(num_gaussians))
        self._scale = nn.Parameter(torch.zeros(num_gaussians, 2))
        self._rotation = nn.Parameter(torch.rand(num_gaussians) * 2*torch.pi)

    def initialize_gaussians(self, pos, concentration, std):
        """
        Overwrite random model parameters with an informed initialization.

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

    # -------- DENSIFICATION ----------

    def _prune_optimizer(self, optimizer: torch.optim.Optimizer, mask):
        optimizable_tensors = {}
        keep_mask = ~mask

        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # Update optimizer's internal state
                stored_state["exp_avg"] = stored_state["exp_avg"][keep_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][keep_mask]

                # Remove old state
                del optimizer.state[group["params"][0]]
                
                # New param
                group["params"][0] = nn.Parameter((group["params"][0][keep_mask].requires_grad_(True)))

                # Set new param's state
                optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][keep_mask].requires_grad_(True))
            
            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _cat_tensors_to_optimizer(self, optimizer: torch.optim.Optimizer, tensors_dict):
        optimizable_tensors = {}

        for group in optimizer.param_groups:
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # Update optimizer's internal state
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                # Remove old state
                del optimizer.state[group["params"][0]]

                # New param
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )

                # Set param's new state
                optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
            
            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors      

    def prune(self, optimizer: torch.optim.Optimizer, mask):
        optimizable_tensors = self._prune_optimizer(optimizer, mask)

        self._pos = optimizable_tensors["pos"]
        self._concentration = optimizable_tensors["concentration"]
        self._scale = optimizable_tensors["scale"]
        self._rotation = optimizable_tensors["rotation"]

    def clone(self, optimizer: torch.optim.Optimizer, mask):
        new_pos = self._pos[mask]
        new_concentration = inverse_softplus(self.get_concentration()[mask] * 0.5)
        new_scale = self._scale[mask]
        new_rotation = self._rotation[mask]

        # Original gaussians now have half the concentration
        self._concentration[mask] = new_concentration

        tensors_dict = {
            "pos": new_pos,
            "concentration": new_concentration,
            "scale": new_scale,
            "rotation": new_rotation
        }

        # Add new gaussians' parameters
        optimizable_tensors = self._cat_tensors_to_optimizer(optimizer, tensors_dict)

        self._pos = optimizable_tensors["pos"]
        self._concentration = optimizable_tensors["concentration"]
        self._scale = optimizable_tensors["scale"]
        self._rotation = optimizable_tensors["rotation"]

    def split(self, optimizer: torch.optim.Optimizer, mask, N=2):
        # Generate new positions based on original gaussian functions
        stds = self.get_scale()[mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 2))
        samples = torch.normal(mean=means, std=stds)
        
        # Transform to global coordinate system
        rots = self.get_rotation_matrix()[mask].repeat(N, 1, 1)
        new_pos = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_pos()[mask].repeat(N, 1)

        # Avoid invalid positions and transform to parameter's space
        new_pos = torch.clamp(new_pos, min=1e-5, max=self.map_size - 1e-5)
        new_pos = inverse_sigmoid(new_pos, self.map_size)

        # Divide original concentration by N
        new_concentration = inverse_softplus(
            self.get_concentration()[mask].repeat(N) * (1 / N)
        )

        # Divide original scale by a factor of 0.8 * N
        new_scale = self._scale[mask].repeat(N, 1) - math.log(0.8 * N)

        new_rotation = self._rotation[mask].repeat(N)

        tensors_dict = {
            "pos": new_pos,
            "concentration": new_concentration,
            "scale": new_scale,
            "rotation": new_rotation
        }

        # Add new gaussians' parameters
        self._cat_tensors_to_optimizer(optimizer, tensors_dict)

        # Prune original gaussians
        prune_mask = torch.cat(
            (
                mask, 
                torch.zeros(N * mask.sum(), dtype=torch.bool)
            )
        )
        self.prune(optimizer, prune_mask)

    def densify_and_prune(self, optimizer: torch.optim.Optimizer):
        # Gaussians with high gradient
        grads = (self.pos_grad_accum / self.denom).squeeze() # Average pos gradient
        grads[grads.isnan()] = 0.0
        grad_mask = grads > self.densify_cfg.gradient_threshold

        # Gaussians with small scale
        small_scale_mask = torch.max(self.get_scale(), dim=1).values < self.densify_cfg.scale_threshold

        # --- Clone ---
        clone_mask = torch.logical_and(grad_mask, small_scale_mask)
        self.clone(optimizer, clone_mask)

        # --- Split ---
        split_mask = torch.logical_and(grad_mask, ~small_scale_mask)

        # Number of gaussians may have changed
        num_cloned = int(clone_mask.sum().item())
        if num_cloned > 0:
            padding = torch.zeros(num_cloned, dtype=torch.bool)
            split_mask = torch.cat([split_mask, padding])

        self.split(optimizer, split_mask)

        # --- Prune ---
        if self.num_gaussians > 1:
            prune_mask = (self.get_concentration() < self.densify_cfg.prune_threshold).squeeze()
            self.prune(optimizer, prune_mask)

        # Update current gaussians count
        self.num_gaussians = self._pos.shape[0]

        # Reset gradient accumulator
        self.pos_grad_accum = torch.zeros((self.num_gaussians, 1), device=self._pos.device)
        self.denom = torch.zeros((self.num_gaussians, 1), device=self._pos.device)

    def update_accum_gradient(self):
        grad_norm = torch.linalg.vector_norm(self._pos.grad, dim=-1, keepdim=True)
        mask = grad_norm > 1e-6

        self.pos_grad_accum[mask] += grad_norm[mask]
        self.denom[mask] += 1