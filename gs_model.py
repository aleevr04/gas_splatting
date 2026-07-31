import numpy as np
import torch
import torch.nn as nn
import math

from config import Config
from utils.gaussian_utils import (
    compute_definite_integral,
    inverse_sigmoid,
    inverse_softplus
)

class GasSplattingModel(nn.Module):
    def __init__(self, initial_gaussians, cfg: Config):
        super().__init__()
        self.initial_gaussians = initial_gaussians
        self.num_gaussians = initial_gaussians

        self.map_size = torch.tensor(cfg.sim.map_size, device=cfg.device)
        self.densify_cfg = cfg.densify

        self.pos_grad_accum = torch.zeros((initial_gaussians, 1), device=cfg.device)
        self.denom = torch.zeros((initial_gaussians, 1), device=cfg.device)

        self.splits = 0
        self.clones = 0

        # --- Model parameters ---
        self._pos = nn.Parameter(torch.rand(initial_gaussians, 2) * self.map_size)
        self._concentration = nn.Parameter(torch.rand(initial_gaussians))
        self._scale = nn.Parameter(torch.zeros(initial_gaussians, 2))
        self._rotation = nn.Parameter(torch.rand(initial_gaussians) * 2*torch.pi)

    def initialize_gaussians(self, pos: torch.Tensor, concentration: torch.Tensor, std: torch.Tensor):
        """
        Overwrite random model parameters with an informed initialization.

        Args:
            pos (Tensor (K, 2)): Initial positions (x, y) in map coordinates.
            concentration (Tensor (0,) or (K,)): Initial concentration/weight per gaussian.
            std (Tensor (0,), (K,) or (K, 2)): Initial standard deviation.
        """

        with torch.no_grad():
            # Positions: store as logit(normalized_pos) so get_pos() -> sigmoid(_pos)*map_size
            if pos.shape != (self.num_gaussians, 2):
                raise ValueError(f"pos must have shape ({self.num_gaussians}, 2), got {tuple(pos.shape)}")

            self._pos.data.copy_(inverse_sigmoid(pos, self.map_size))

            # Concentration: inverse of softplus
            if concentration.dim() == 0:
                concentration = concentration.expand(self.num_gaussians)
            elif concentration.numel() == self.num_gaussians and concentration.dim() == 1:
                pass
            else:
                raise ValueError(f"concentration must be scalar or shape ({self.num_gaussians},), got {tuple(concentration.shape)}")

            self._concentration.data.copy_(inverse_softplus(concentration))

            # Scale: we store log(scale) in _scale so get_scale() = exp(_scale)
            if std.dim() == 0:
                scales = std * torch.ones((self.num_gaussians, 2), dtype=self._scale.dtype, device=self._scale.device)
            elif std.shape == (self.num_gaussians,):
                scales = std.unsqueeze(1).repeat(1, 2)
            elif std.shape == (self.num_gaussians, 2):
                scales = std
            else:
                raise ValueError(f"std must be scalar, ({self.num_gaussians},) or ({self.num_gaussians},2), got {tuple(std.shape)}")

            self._scale.data.copy_(torch.log(scales))

            # Rotation: initialize to zero (no rotation) by default
            self._rotation.data.zero_()

    def get_pos(self):
        """
        Computes the positions of the Gaussians in map coordinates.
        Returns:
            pos: Tensor of shape (K, 2) representing the positions of the Gaussians in map coordinates.
        """
        return torch.sigmoid(self._pos) * self.map_size

    def get_scale(self):
        """
        Computes the scale (standard deviation) of the Gaussians.
        Returns:
            scale: Tensor of shape (K, 2) representing the scale of the Gaussians in map coordinates.
        """
        return torch.exp(self._scale)

    def get_concentration(self):
        """
        Computes the concentration (weight) of the Gaussians.
        Returns:
            concentration: Tensor of shape (K,) representing the concentration of the Gaussians."""
        return nn.functional.softplus(self._concentration)

    def get_rotation_matrix(self):
        """
        Computes the rotation matrix R for each Gaussian based on its rotation angle.
        Returns:
            R: Tensor of shape (K, 2, 2) representing the rotation matrices"""
        thetas = self._rotation
        cos = torch.cos(thetas).unsqueeze(-1)  # (K, 1)
        sin = torch.sin(thetas).unsqueeze(-1)  # (K, 1)

        row1 = torch.cat([cos, -sin], dim=1)  # (K, 2)
        row2 = torch.cat([sin, cos], dim=1)   # (K, 2)

        R = torch.stack([row1, row2], dim=1)  # (K, 2, 2)

        return R

    def get_scale_square_inverse(self):
        scale_sq_inv = 1.0 / (self.get_scale()**2 + 1e-7)
        return torch.diag_embed(scale_sq_inv) # (K, 2, 2)

    def get_covariance_inverse(self):
        """
        Computes the inverse covariance matrix Sigma^-1 = R * S^-2 * R^T
        Returns:
            covariance_inverse: Tensor of shape (K, 2, 2) representing the inverse covariance matrices of the Gaussians.
        """
        R = self.get_rotation_matrix()
        S_sq_inv = self.get_scale_square_inverse()

        # Sigma^-1 = R * S^-2 * R^T
        covariance_inverse = torch.bmm(R, torch.bmm(S_sq_inv, R.transpose(1, 2)))

        return covariance_inverse

    def get_covariance(self):
        """Computes the covariance matrix Sigma = R * S^2 * R^T
        Returns:
            covariance: Tensor of shape (K, 2, 2) representing the covariance matrices of the Gaussians.
        """
        R = self.get_rotation_matrix()
        
        # S^2
        scales_sq = self.get_scale()**2
        S_sq = torch.diag_embed(scales_sq) # (K, 2, 2)

        # Sigma = R * S^2 * R^T
        covariance = torch.bmm(R, torch.bmm(S_sq, R.transpose(1, 2)))

        return covariance

    def forward(self, beams):
        pos = self.get_pos()
        covariance_inverse = self.get_covariance_inverse()
        concentration = self.get_concentration()

        return compute_definite_integral(pos, covariance_inverse, concentration, beams)
    
    def inject_gaussians(self, optimizer: torch.optim.Optimizer, pos: torch.Tensor, scale: float, concentration: float = 0.01):
        """
        Injects new Gaussians into the model with specified parameters.
        """
        N = pos.shape[0]
        if N == 0: return

        # Initialize new parameters for the injected Gaussians
        new_pos = inverse_sigmoid(pos, self.map_size)
        new_scale = torch.log(torch.full((N, 2), scale, device=self._scale.device))
        new_concentration = inverse_softplus(torch.full((N,), concentration, device=self._concentration.device))
        new_rotation = torch.zeros(N, device=self._rotation.device)

        tensors_dict = {
            "pos": new_pos,
            "concentration": new_concentration,
            "scale": new_scale,
            "rotation": new_rotation
        }

        optimizable_tensors = self._cat_tensors_to_optimizer(optimizer, tensors_dict)

        self._pos = optimizable_tensors["pos"]
        self._concentration = optimizable_tensors["concentration"]
        self._scale = optimizable_tensors["scale"]
        self._rotation = optimizable_tensors["rotation"]

        self.num_gaussians = self._pos.shape[0]
        self.pos_grad_accum = torch.cat([self.pos_grad_accum, torch.zeros((N, 1), device=self._pos.device)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((N, 1), device=self._pos.device)], dim=0)
    
    def render_map(self, cell_size: float) -> np.ndarray:
        """
        Turns Gaussians into a 2D image (numpy matrix)
        """
        # Extract dimensions from the map_size tensor
        map_w = self.map_size[0].item()
        map_h = self.map_size[1].item()
        
        # Dynamically infer the device from model parameters
        device = self._pos.device

        w_cells = int(map_w / cell_size)
        h_cells = int(map_h / cell_size)

        # Grid setup
        x = torch.linspace(0, map_w, w_cells, device=device)
        y = torch.linspace(0, map_h, h_cells, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        grid_pos = torch.stack([X, Y], dim=-1) # (H, W, 2)

        # PyTorch expects (H, W) -> (h_cells, w_cells)
        final_img = torch.zeros((h_cells, w_cells), device=device)

        with torch.no_grad():
            pos = self.get_pos()
            cov_inv = self.get_covariance_inverse()
            concentration = self.get_concentration()

            # Sum each Gaussian contribution
            for k in range(self.num_gaussians):
                mu = pos[k]
                sig_inv = cov_inv[k]
                c = concentration[k]
                
                # Evaluate Gaussian at each cell
                d = grid_pos - mu
                d = d.unsqueeze(-1) 
                
                sig_inv_exp = sig_inv.view(1, 1, 2, 2)
                dist = torch.matmul(d.transpose(-1, -2), torch.matmul(sig_inv_exp, d)).squeeze()
                
                final_img += c * torch.exp(-0.5 * dist)

        return final_img.detach().cpu().numpy()

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

    def split_original(self, optimizer: torch.optim.Optimizer, mask, N=2):
        """
        Splits the original Gaussians into N new Gaussians.
        New Gaussians' positions are sampled from the original Gaussian.
        Scale is reduced by 0.8 * N.
        Concentration is divided by N.
        """
        stds = self.get_scale()[mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 2), device=stds.device)
        samples = torch.normal(mean=means, std=stds)

        # Transform to global coordinate system
        rots = self.get_rotation_matrix()[mask].repeat(N, 1, 1)
        new_pos = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_pos()[mask].repeat(N, 1)

        # Avoid invalid positions and transform to parameter's space
        new_pos = torch.max(new_pos, torch.tensor(1e-5, device=new_pos.device))
        new_pos = torch.min(new_pos, self.map_size - 1e-5)
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
                torch.zeros(N * mask.sum(), dtype=torch.bool, device=mask.device)
            )
        )
        self.prune(optimizer, prune_mask)

    def split_long_axis(self, optimizer: torch.optim.Optimizer, mask):
        """
        Places 2 new gaussians symmetrically along the longest axis.
        Concentration and scale are computed dynamically to preserve variation and mass.
        """
        N = 2
        
        pos = self.get_pos()[mask]
        scales = self.get_scale()[mask]
        rots = self.get_rotation_matrix()[mask]
        concs = self.get_concentration()[mask]
        
        # Determine max and min axis scales
        s_max, max_axis_idx = torch.max(scales, dim=1)
        s_min, min_axis_idx = torch.min(scales, dim=1)
        s_min = torch.clamp(s_min, min=1e-5) # Prevent division by zero
        
        c = torch.full_like(s_max, fill_value=0.5) 
        
        idx = torch.arange(scales.size(0))
        
        # Shift new positions using the calculated c along the longest axis
        shift_local = torch.zeros_like(scales)
        shift_local[idx, max_axis_idx] = c * s_max
        
        # Transform local shifts to the global coordinate system
        shift_global = torch.bmm(rots, shift_local.unsqueeze(-1)).squeeze(-1)
        
        new_pos = torch.cat([
            pos + shift_global,
            pos - shift_global
        ], dim=0)
        
        new_pos = torch.max(new_pos, torch.tensor(1e-5, device=new_pos.device))
        new_pos = torch.min(new_pos, self.map_size - 1e-5)
        new_pos = inverse_sigmoid(new_pos, self.map_size)
        
        # Mass Conservation (Scales). Minor scale stays untouched
        new_scales = scales.clone()
        c_tensor_sqrt = torch.sqrt(1 - c**2)
        new_scales[idx, max_axis_idx] *= c_tensor_sqrt
        
        new_scale = torch.log(torch.cat([new_scales, new_scales], dim=0))
        
        # Mass Conservation (Concentration)
        new_concs = concs / (2.0 * c_tensor_sqrt)
        new_concentration = inverse_softplus(torch.cat([new_concs, new_concs], dim=0))
        
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
                torch.zeros(N * mask.sum(), dtype=torch.bool, device=mask.device)
            )
        )
        self.prune(optimizer, prune_mask)

    def densify_and_prune(self, optimizer: torch.optim.Optimizer):
        """
        Densifies the model by splitting/cloning Gaussians with high gradient and pruning low concentration Gaussians.
        Returns:
            dict: A dictionary containing the number of splits, clones, and prunes performed.
        """
        # Gaussians with high gradient
        grads = (self.pos_grad_accum / self.denom).squeeze(1) # Average pos gradient
        grads[grads.isnan()] = 0.0
        grad_mask = grads > self.densify_cfg.gradient_threshold

        use_original_dens = hasattr(self.densify_cfg, 'original_dens') and self.densify_cfg.original_dens

        num_clones = 0
        num_splits = 0

        if use_original_dens:
            # --- Original Densification ---
            # Gaussians with small scale
            adjusted_scales = self.get_scale() / torch.max(self.map_size)
            small_scale_mask = torch.max(adjusted_scales, dim=1).values < self.densify_cfg.scale_threshold

            # Clone
            clone_mask = torch.logical_and(grad_mask, small_scale_mask)
            num_clones = int(clone_mask.sum().item())
            
            if num_clones > 0:
                self.clone(optimizer, clone_mask)
                self.clones += num_clones

            # Split
            split_mask = torch.logical_and(grad_mask, ~small_scale_mask)

            # Number of gaussians may have changed due to previous clone operation
            if num_clones > 0:
                padding = torch.zeros(num_clones, dtype=torch.bool, device=split_mask.device)
                split_mask = torch.cat([split_mask, padding])

            num_splits = int(split_mask.sum().item())
            
            if num_splits > 0:
                self.split_original(optimizer, split_mask)
                self.splits += num_splits
        else:
            # --- Proposed Densification ---
            # We don't use small_scale_mask. Every Gaussian with high gradient
            # undergoes the unified operation
            num_splits = int(grad_mask.sum().item())
            
            if num_splits > 0:
                self.split_long_axis(optimizer, grad_mask)
                self.splits += num_splits

        # --- Prune ---
        num_prunes = 0
        if self.num_gaussians > 1:
            prune_mask = (self.get_concentration() < self.densify_cfg.prune_threshold).view(-1)
            num_prunes = int(prune_mask.sum().item())
            self.prune(optimizer, prune_mask)

        # Update current gaussians count
        self.num_gaussians = self._pos.shape[0]

        # Reset gradient accumulator
        self.pos_grad_accum = torch.zeros((self.num_gaussians, 1), device=self._pos.device)
        self.denom = torch.zeros((self.num_gaussians, 1), device=self._pos.device)

        return {'splits': num_splits, 'clones': num_clones, 'prunes': num_prunes}

    def update_accum_gradient(self):
        if self._pos.grad is not None:
            adjusted_grads = self._pos.grad / self.map_size
            grad_norm = torch.linalg.vector_norm(adjusted_grads, dim=-1, keepdim=True)
            mask = grad_norm > 1e-6

            self.pos_grad_accum[mask] += grad_norm[mask]
            self.denom[mask] += 1