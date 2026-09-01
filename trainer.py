import os
import time
import torch
import numpy as np
import scipy.ndimage as ndimage
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import EnvironmentContext, MeasurementBatch
from utils.densification_utils import extract_candidate_positions


@dataclass
class BatchResults:
    final_loss: float = 0.0
    batch_time: float = 0.0

@dataclass
class TrainingResults:
    loss_history: List[float] = field(default_factory=list)
    densify_history: Dict[int, dict] = field(default_factory=dict)
    rmse_history: Dict[int, float] = field(default_factory=dict)
    training_time: float = 0.0


def get_exp_lr_func(lr_init, lr_final, max_steps):
    def lr_func(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        
        t = np.clip(step / max_steps, 0, 1)
        return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

    return lr_func

class Trainer:
    def __init__(self, model: GasSplattingModel, cfg: Config, environment: EnvironmentContext | None = None, max_buffer_size: int | None = None):
        self.model = model
        self.ground_truth = environment.ground_truth.gas_map if environment and environment.ground_truth else None
        self.cfg = cfg
        if self.cfg.train.live_vis:
            from visualization.live_visualizer import LiveVisualizer
            self.visualizer = LiveVisualizer(cfg)
        else:
            self.visualizer = None
        if self.visualizer and self.ground_truth is not None:
            self.visualizer.set_ground_truth(self.ground_truth)
        self.results = TrainingResults()
        self.global_iteration = 0

        # --- Process obstacles to compute SDF and gradients ---
        self.sdf_tensor = None
        self.sdf_grad_tensor = None
        
        if environment and environment.obstacles is not None:
            obstacles = environment.obstacles
            free_space = (obstacles <= 0.5)
            occupied_space = (obstacles > 0.5)
            
            dist_outside = np.asarray(ndimage.distance_transform_edt(free_space))
            dist_inside = np.asarray(ndimage.distance_transform_edt(occupied_space))
            
            # SDF field in meters
            sdf_physical = (dist_outside - dist_inside) * cfg.env.cell_size
            
            # Compute spatial gradients (normals) using numpy
            # np.gradient returns (gradient_y, gradient_x) for a 2D array
            grad_y, grad_x = np.gradient(sdf_physical)
            
            # Stack into (2, H, W)
            sdf_grad_physical = np.stack([grad_x, grad_y], axis=0)
            
            # Store SDF as (1, 1, H, W)
            self.sdf_tensor = torch.tensor(
                sdf_physical, 
                dtype=torch.float32, 
                device=self.cfg.device
            ).unsqueeze(0).unsqueeze(0)
            
            # Store Gradients as (1, 2, H, W)
            self.sdf_grad_tensor = torch.tensor(
                sdf_grad_physical,
                dtype=torch.float32,
                device=self.cfg.device
            ).unsqueeze(0)
        # ----------------------------------------------------
        
        # Data Buffers
        self.max_buffer_size = max_buffer_size or float('inf')
        dev = self.cfg.device
        self.buffer_beams = torch.empty((0, 2, 2), device=dev)
        self.buffer_measurements = torch.empty((0,), device=dev)
        self.buffer_weights = torch.empty((0,), device=dev)

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam([
            {'params': [model._pos], 'lr': self.cfg.train.pos_lr, 'name': 'pos'},
            {'params': [model._scale], 'lr': self.cfg.train.scale_lr, 'name': 'scale'},
            {'params': [model._rotation], 'lr': self.cfg.train.rotation_lr, 'name': 'rotation'},
            {'params': [model._concentration], 'lr': self.cfg.train.concentration_lr, 'name': 'concentration'},
        ])

        # Setup Learning Rates Schedulers
        self.pos_lr_func = get_exp_lr_func(
            lr_init=cfg.train.pos_lr,
            lr_final=0.1*cfg.train.pos_lr,
            max_steps=cfg.train.iterations
        )
        self.scale_lr_func = get_exp_lr_func(
            lr_init=cfg.train.scale_lr,
            lr_final=0.1*cfg.train.scale_lr,
            max_steps=cfg.train.iterations
        )
        self.rotation_lr_func = get_exp_lr_func(
            lr_init=cfg.train.rotation_lr,
            lr_final=0.1*cfg.train.rotation_lr,
            max_steps=cfg.train.iterations
        )
        self.concentration_lr_func = get_exp_lr_func(
            lr_init=cfg.train.concentration_lr,
            lr_final=0.1*cfg.train.concentration_lr,
            max_steps=cfg.train.iterations
        )

    def is_densify_it(self, iteration: int):
        return (iteration > self.cfg.densify.densify_from and 
                iteration < self.cfg.densify.densify_until and
                iteration % self.cfg.densify.densify_interval == 0)
    
    def update_learning_rates(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pos":
                param_group["lr"] = self.pos_lr_func(iteration)
            elif param_group["name"] == "scale":
                param_group["lr"] = self.scale_lr_func(iteration)
            elif param_group["name"] == "rotation":
                param_group["lr"] = self.rotation_lr_func(iteration)
            elif param_group["name"] == "concentration":
                param_group["lr"] = self.concentration_lr_func(iteration)

    def update_buffers(self, batch_data: MeasurementBatch):
        """Merges new incoming data into the training buffers, applying weight decay to older data."""
        # New measurements have maximum importance
        new_weights = torch.ones_like(batch_data.measurements)

        # Merge new data
        self.buffer_beams = torch.cat([self.buffer_beams, batch_data.beams], dim=0)
        self.buffer_measurements = torch.cat([self.buffer_measurements, batch_data.measurements], dim=0)
        self.buffer_weights = torch.cat([self.buffer_weights, new_weights], dim=0)

        # Truncate buffers for memory safety
        if self.buffer_measurements.shape[0] > self.max_buffer_size:
            self.buffer_beams = self.buffer_beams[-self.max_buffer_size:]
            self.buffer_measurements = self.buffer_measurements[-self.max_buffer_size:]
            self.buffer_weights = self.buffer_weights[-self.max_buffer_size:]

    def optimization_step(self, iteration: int):
        """Executes a single optimization step (Forward + Backward + Step)."""
        self.optimizer.zero_grad()
        
        y_pred = self.model(self.buffer_beams)

        # Weigthed measurements loss
        data_loss = torch.mean(self.buffer_weights * torch.abs(y_pred - self.buffer_measurements))
        total_loss = data_loss

        # Directional SDF repulsion loss
        if self.sdf_tensor is not None and self.sdf_grad_tensor is not None:
            pos = self.model.get_pos()                      # (K, 2)
            Sigma = self.model.get_covariance()             # (K, 2, 2)
            concentrations = self.model.get_concentration() # (K,)

            # Normalize positions to [-1, 1] for grid sampling
            map_w, map_h = self.model.map_size[0], self.model.map_size[1]
            norm_x = (pos[:, 0] / map_w) * 2.0 - 1.0
            norm_y = (pos[:, 1] / map_h) * 2.0 - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2) # (1, 1, K, 2)
            
            # Sample SDF distance: (1, 1, 1, K) -> (K,)
            sdf_at_pos = F.grid_sample(
                self.sdf_tensor, grid, 
                mode='bilinear', padding_mode='border', align_corners=False
            ).view(-1)
            
            # Sample SDF gradient: (1, 2, 1, K) -> (K, 2)
            grad_at_pos = F.grid_sample(
                self.sdf_grad_tensor, grid, 
                mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(0).squeeze(1).transpose(0, 1)
            
            # Normalize to get the unit normal vector 'n'
            # Add small epsilon to avoid division by zero if gradient is flat
            n = F.normalize(grad_at_pos, p=2, dim=-1, eps=1e-6) # (K, 2)
            
            # Calculate directional variance: n^T * Sigma * n
            n_unsq_left = n.unsqueeze(1)  # (K, 1, 2)
            n_unsq_right = n.unsqueeze(-1) # (K, 2, 1)
            
            # (K, 1, 2) @ (K, 2, 2) @ (K, 2, 1) -> (K, 1, 1) -> (K,)
            directional_variance = torch.bmm(n_unsq_left, torch.bmm(Sigma, n_unsq_right)).view(-1)
            
            # Standard deviation in the direction of the wall
            sigma_n = torch.sqrt(torch.clamp(directional_variance, min=1e-7))
            
            # Dynamic margin: 2 std deviations along the normal
            margin = 2.0 * sigma_n
            
            # Calculate penalty
            violations = F.relu(margin - sdf_at_pos)
            sdf_penalty = torch.sum(violations * concentrations)
            
            obstacle_lambda = getattr(self.cfg.train, 'obstacle_lambda', 0.1)
            total_loss = total_loss + (obstacle_lambda * sdf_penalty)

        total_loss.backward()
        self.model.update_accum_gradient()
        self.update_learning_rates(iteration)
        self.optimizer.step()
        
        return total_loss.item()

    def inject_gaussians(self) -> int:
        """Injects new Gaussians based on high-error beams using Continuous Spatial NMS."""
        y_pred = self.model(self.buffer_beams)
        residuals = self.buffer_measurements - y_pred
        
        # Dynamic scale
        map_w, map_h = self.model.map_size[0].item(), self.model.map_size[1].item()
        init_scale = min(map_w, map_h) * 0.1
        
        final_pos = extract_candidate_positions(
            beams=self.buffer_beams,
            importance_scores=residuals,
            min_dist=init_scale,
            device=self.cfg.device
        )
        
        num_injected = final_pos.shape[0]
        
        if num_injected > 0:
            self.model.inject_gaussians(
                optimizer=self.optimizer, 
                pos=final_pos, 
                scale=init_scale
            )
            
        return num_injected

    def train(self, batch_data: MeasurementBatch) -> BatchResults:
        self.update_buffers(batch_data)

        # Early stopping init
        ema_loss = None
        best_ema_loss = float('inf')
        patience_counter = 0

        # Training timer
        batch_time = 0.0

        current_loss = 0.0
        pbar = tqdm(range(self.cfg.train.iterations), desc="Training", dynamic_ncols=True, disable=self.cfg.quiet)        
        for iteration in pbar:
            t_start = time.time()
            
            current_loss = self.optimization_step(iteration)
            self.results.loss_history.append(current_loss)

            # --- Early stopping ---
            # Update the smoothed EMA loss
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss = (self.cfg.train.ema_alpha * current_loss) + ((1 - self.cfg.train.ema_alpha) * ema_loss)

            if iteration > self.cfg.densify.densify_until:
                # Check for significant improvement
                if ema_loss < best_ema_loss - self.cfg.train.early_stopping_min_delta:
                    best_ema_loss = ema_loss
                    patience_counter = 0  # Reset patience if we hit a new best
                else:
                    patience_counter += 1 # Increment if it stalled or got worse
                    
                # Halt if patience runs out
                if patience_counter >= self.cfg.train.early_stopping_patience:
                    batch_time += (time.time() - t_start) # Add iteration time before breaking
                    if not self.cfg.quiet:
                        tqdm.write(f"Early stopping triggered at iteration {iteration} (EMA Loss stalled at {ema_loss:.5f})")
                    break 
            # ----------------------------

            # --- Densification ---
            is_densifying = self.is_densify_it(iteration)
            if is_densifying:
                with torch.no_grad():
                    stats = self.model.densify_and_prune(self.optimizer)
                    stats['injections'] = self.inject_gaussians()

                self.results.densify_history[self.global_iteration] = stats
            # -------------------------

            batch_time += (time.time() - t_start)

            if iteration % 100 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.5f}', 'gaussians': self.model.num_gaussians})
            
            # --- Real time visualization ---
            if self.visualizer and (iteration % 20 == 0 or is_densifying):
                self.visualizer.update(
                    iteration=self.global_iteration, 
                    loss_history=self.results.loss_history, 
                    model=self.model
                )
            # ---------------------------------
            
            # --- Model evaluation ---
            if self.cfg.train.do_eval and iteration % self.cfg.train.eval_interval == 0:
                current_map = self.model.render_map(cell_size=self.cfg.env.cell_size)
                rmse = np.sqrt(np.mean((current_map - self.ground_truth)**2))
                self.results.rmse_history[self.global_iteration] = rmse
            # ----------------------------

            self.global_iteration += 1

        pbar.close()

        self.results.training_time += batch_time

        return BatchResults(final_loss=current_loss, batch_time=batch_time)

    def finish(self, gif_path: str = "plots/training.gif") -> TrainingResults:
        """Handles post-training cleanup and returns the global training history."""
        if self.visualizer:
            os.makedirs("plots", exist_ok=True)
            self.visualizer.save_gif(gif_path)

        del self.buffer_beams
        del self.buffer_measurements
        del self.buffer_weights

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.results