import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import SimulationData


@dataclass
class TrainingResults:
    loss_history: List[float] = field(default_factory=list)
    densify_history: Dict[int, dict] = field(default_factory=dict)
    rmse_history: Dict[int, float] = field(default_factory=dict)


class LiveVisualizer:
    def __init__(self, map_size):
        # Interactive mode
        plt.ion()
        self.fig, (self.ax_loss, self.ax_map) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("Real-time Gas Splatting Training")

        # Loss History
        self.ax_loss.set_title("Loss History")
        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.set_ylabel("Total Loss")
        self.ax_loss.set_yscale('log')
        self.loss_line, = self.ax_loss.plot([], [], 'b-', alpha=0.8)

        # Gaussians
        self.ax_map.set_title("Gaussians' positions")
        self.ax_map.set_xlim(0, map_size[0])
        self.ax_map.set_ylim(0, map_size[1])
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, linestyle='--', alpha=0.5)
        
        # Point size depends on gaussian concentration
        self.scatter = self.ax_map.scatter([], [], c='red', alpha=0.6, edgecolors='k')
        
        # Informative text
        self.text_info = self.ax_map.text(0.05, 0.95, '', transform=self.ax_map.transAxes, 
                                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show(block=False)

    def update(self, iteration, loss_history, pos_tensor, concentration_tensor):
        # update loss
        x_data = list(range(len(loss_history)))
        self.loss_line.set_data(x_data, loss_history)
        
        # Readjust loss axes
        self.ax_loss.set_xlim(0, max(100, iteration + 10))
        valid_losses = [l for l in loss_history if not np.isnan(l) and not np.isinf(l)]
        if valid_losses:
            self.ax_loss.set_ylim(min(valid_losses) * 0.5, max(valid_losses) * 1.5)

        # Get gaussians position and concentration
        pos = pos_tensor.detach().cpu().numpy()
        conc = concentration_tensor.detach().cpu().numpy()
        
        # Update coordinates
        self.scatter.set_offsets(pos)
        
        # Update sizes
        sizes = np.clip(conc * 50, 10, 500) # Evitar puntos invisibles o gigantes
        self.scatter.set_sizes(sizes)
        
        # Update text
        self.text_info.set_text(f"Iteration: {iteration}\nGaussians: {len(pos)}")

        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


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
        
        self.visualizer =  LiveVisualizer(model.map_size) if self.cfg.train.live_vis else None

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

        pbar = tqdm(range(self.cfg.train.iterations), desc="Training", dynamic_ncols=True)        
        for it in pbar:
            self.optimizer.zero_grad()
            
            y_pred = self.model(sim_data.beams)

            l1_loss = F.l1_loss(y_pred, sim_data.y_true)
            
            total_loss = l1_loss
            
            total_loss.backward()
            self.model.update_accum_gradient()
            self.update_learning_rates(it)
            self.optimizer.step()
            
            current_loss = total_loss.item()
            results.loss_history.append(current_loss)

            if it % 100 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.5f}'})
            
            # Real time visualization
            if self.visualizer and it % 20 == 0:
                with torch.no_grad():
                    self.visualizer.update(
                        iteration=it, 
                        loss_history=results.loss_history, 
                        pos_tensor=self.model.get_pos(), 
                        concentration_tensor=self.model.get_concentration()
                    )
            
            # Model evaluation
            if self.cfg.train.do_eval and it % self.cfg.train.eval_interval == 0:
                with torch.no_grad():
                    from utils.plot_utils import render_gaussian_map 
                    current_map = render_gaussian_map(
                        self.model, 
                        self.cfg.sim.map_size, 
                        self.cfg.device, 
                        cell_size=self.cfg.sim.cell_size
                    )
                    rmse = np.sqrt(np.mean((current_map - sim_data.img_gt)**2))
                    
                    # Store it inside the class instance
                    results.rmse_history[it] = rmse

            # Densification
            if self.is_densify_it(it):
                with torch.no_grad():
                    stats = self.model.densify_and_prune(self.optimizer)
                    results.densify_history[it] = stats

                    # Forzamos un refresco justo después de densificar para ver el cambio instantáneo
                    if self.visualizer:
                        self.visualizer.update(
                            iteration=it, 
                            loss_history=results.loss_history, 
                            pos_tensor=self.model.get_pos(), 
                            concentration_tensor=self.model.get_concentration()
                        )
            
            if current_loss < self.cfg.train.target_loss:
                pbar.write(f"Training ended at iteration: {it}")
                break

        pbar.close()
        
        # Deactivate interactive mode
        if self.visualizer:
            plt.ioff()

        return results
