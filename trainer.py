import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import SimulationData
from utils.live_vis import LiveVisualizer


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
    def __init__(self, model: GasSplattingModel, cfg: Config):
        self.model = model
        self.cfg = cfg
        
        self.visualizer =  LiveVisualizer(cfg) if self.cfg.train.live_vis else None

        self.optimizer: optim.Optimizer = optim.Adam([
            {'params': [model._pos], 'lr': self.cfg.train.pos_lr, 'name': 'pos'},
            {'params': [model._scale], 'lr': self.cfg.train.scale_lr, 'name': 'scale'},
            {'params': [model._rotation], 'lr': self.cfg.train.rotation_lr, 'name': 'rotation'},
            {'params': [model._concentration], 'lr': self.cfg.train.concentration_lr, 'name': 'concentration'},
        ])

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

    def is_densify_it(self, iteration):
        dfrom, duntil, dinterval = (
            self.cfg.densify.densify_from,
            self.cfg.densify.densify_until,
            self.cfg.densify.densify_interval
        )

        return (iteration >= dfrom and iteration <= duntil and (iteration - dfrom) % dinterval == 0)
    
    def update_learning_rates(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pos":
                param_group["lr"] = self.pos_lr_func(iteration)
            elif param_group["name"] == "scale":
                param_group["lr"] = self.scale_lr_func(iteration)
            elif param_group["name"] == "rotation":
                param_group["lr"] = self.rotation_lr_func(iteration)
            elif param_group["name"] == "concentration":
                param_group["lr"] = self.concentration_lr_func(iteration)

    def train(self, sim_data: SimulationData):
        results = TrainingResults()

        if self.visualizer: self.visualizer.set_ground_truth(sim_data.img_gt)

        # Early stopping init
        ema_loss = None
        best_ema_loss = float('inf')
        patience_counter = 0

        # Training timer
        total_train_time = 0.0

        pbar = tqdm(range(self.cfg.train.iterations), desc="Training", dynamic_ncols=True)        
        for it in pbar:
            t_start = time.time()
            self.optimizer.zero_grad()
            
            y_pred = self.model(sim_data.beams)

            l1_loss = F.l1_loss(y_pred, sim_data.measurements)

            total_loss = l1_loss
            
            total_loss.backward()
            self.model.update_accum_gradient()
            self.update_learning_rates(it)
            self.optimizer.step()
            
            current_loss = total_loss.item()
            results.loss_history.append(current_loss)

            # --- Early stopping ---
            # Update the smoothed EMA loss
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss = (self.cfg.train.ema_alpha * current_loss) + ((1 - self.cfg.train.ema_alpha) * ema_loss)
            
            # Check that we have passed the last densification iteration
            if it > self.cfg.densify.densify_until:
                # Check for significant improvement
                if ema_loss < best_ema_loss - self.cfg.train.early_stopping_min_delta:
                    best_ema_loss = ema_loss
                    patience_counter = 0  # Reset patience if we hit a new best
                else:
                    patience_counter += 1 # Increment if it stalled or got worse
                    
                # Halt if patience runs out
                if patience_counter >= self.cfg.train.early_stopping_patience:
                    total_train_time += (time.time() - t_start) # Add iteration time before breaking
                    tqdm.write(f"Early stopping triggered at iteration {it} (EMA Loss stalled at {ema_loss:.5f})")
                    break 
            # ----------------------------

            # --- Densification ---
            is_densifying = self.is_densify_it(it)
            if is_densifying:
                with torch.no_grad():
                    stats = self.model.densify_and_prune(self.optimizer)
                results.densify_history[it] = stats
            # -------------------------

            total_train_time += (time.time() - t_start)

            if it % 100 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.5f}'})
            
            # --- Real time visualization ---
            if self.visualizer and (it % 20 == 0 or is_densifying):
                self.visualizer.update(
                    iteration=it, 
                    loss_history=results.loss_history, 
                    model=self.model
                )
            # ---------------------------------
            
            # --- Model evaluation ---
            if self.cfg.train.do_eval and it % self.cfg.train.eval_interval == 0:
                current_map = self.model.render_map(cell_size=self.cfg.sim.cell_size)
                rmse = np.sqrt(np.mean((current_map - sim_data.img_gt)**2))
                
                # Store it inside the class instance
                results.rmse_history[it] = rmse
            # ----------------------------

        pbar.close()
        
        if self.visualizer:
            # Save Training GIF
            os.makedirs("plots", exist_ok=True)
            self.visualizer.save_gif()

        results.training_time = total_train_time

        return results
