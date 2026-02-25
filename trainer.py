import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import TrainParams, DensificationParams
from gs_model import GasSplattingModel


class LiveVisualizer:
    def __init__(self, map_size):
        self.map_size = map_size
        
        # Interactive mode
        plt.ion()
        self.fig, (self.ax_loss, self.ax_map) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("Real-time Gas Splatting Training")

        # Loss History
        self.ax_loss.set_title("Loss History")
        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.set_ylabel("MSE")
        self.ax_loss.set_yscale('log')
        self.loss_line, = self.ax_loss.plot([], [], 'b-', alpha=0.8)

        # Gaussians
        self.ax_map.set_title("Gaussians' positions")
        self.ax_map.set_xlim(0, map_size)
        self.ax_map.set_ylim(0, map_size)
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


class Trainer:
    def __init__(self, model: GasSplattingModel, train_cfg: TrainParams, densify_cfg: DensificationParams):
        self.model = model
        self.train_cfg = train_cfg
        self.densify_cfg = densify_cfg
        
        self.visualizer = None if self.train_cfg.no_live_vis else LiveVisualizer(model.map_size)

        self.optimizer: optim.Optimizer = optim.Adam([
            {'params': [model._pos], 'lr': train_cfg.pos_lr, 'name': 'pos'},
            {'params': [model._scale], 'lr': train_cfg.scale_lr, 'name': 'scale'},
            {'params': [model._rotation], 'lr': train_cfg.rotation_lr, 'name': 'rotation'},
            {'params': [model._concentration], 'lr': train_cfg.concentration_lr, 'name': 'concentration'},
        ])
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_cfg.lr_decay_step,
            gamma=train_cfg.lr_decay
        )

    def is_densify_it(self, iteration):
        dfrom, duntil, dinterval = (
            self.densify_cfg.densify_from,
            self.densify_cfg.densify_until,
            self.densify_cfg.densify_interval
        )

        return (iteration >= dfrom and iteration <= duntil and (iteration - dfrom) % dinterval == 0)

    def train(self, p_rays, u_rays, y_true):
        loss_history = []
        pbar = tqdm(range(self.train_cfg.iterations), desc="Training", dynamic_ncols=True)
        
        for it in pbar:
            self.optimizer.zero_grad()
            
            y_pred = self.model(p_rays, u_rays)
            loss = torch.mean((y_pred - y_true)**2)
            
            total_loss = loss + self.train_cfg.l1_reg * torch.mean(self.model.get_concentration())
            
            total_loss.backward()
            self.model.update_accum_gradient()
            self.optimizer.step()
            self.scheduler.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)

            if it % 100 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.5f}'})
            
            # Real time visualization
            if self.visualizer and it % 20 == 0:
                with torch.no_grad():
                    self.visualizer.update(
                        iteration=it, 
                        loss_history=loss_history, 
                        pos_tensor=self.model.get_pos(), 
                        concentration_tensor=self.model.get_concentration()
                    )
            
            # Densification
            if self.is_densify_it(it):
                with torch.no_grad():
                    self.model.densify_and_prune(self.optimizer)
                    
                    # Forzamos un refresco justo después de densificar para ver el cambio instantáneo
                    if self.visualizer:
                        self.visualizer.update(
                            iteration=it, 
                            loss_history=loss_history, 
                            pos_tensor=self.model.get_pos(), 
                            concentration_tensor=self.model.get_concentration()
                        )
            
            if current_loss < self.train_cfg.target_loss:
                pbar.write(f"Training ended at iteration: {it}")
                break

        pbar.close()
        
        # Deactivate interactive mode
        if self.visualizer:
            plt.ioff()
        
        print(f"Final Loss: {loss.item():.6f}")

        return loss_history
