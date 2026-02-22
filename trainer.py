import torch
import torch.optim as optim
from tqdm import tqdm

from config import TrainParams, DensificationParams
from gs_model import GasSplattingModel

class Trainer:
    def __init__(self, model: GasSplattingModel, train_cfg: TrainParams, densify_cfg: DensificationParams):
        self.model = model
        self.train_cfg = train_cfg
        self.densify_cfg = densify_cfg

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
        return (iteration >= self.densify_cfg.densify_from
            and iteration <= self.densify_cfg.densify_until 
            and iteration % self.densify_cfg.densify_interval == 0)

    def train(self, p_rays, u_rays, y_true):
        """
            Training loop.

            Args:
                p_rays (Tensor (N,2)): Rays' points
                u_rays (Tensor (N,2)): Rays' direction vectors
                y_true (Tensor (N,)): Real measurements

            Returns:
                loss_history (list): Loss value in each iteration
        """
        loss_history = []

        # Progress bar
        pbar = tqdm(range(self.train_cfg.iterations), desc="Training")
        for it in pbar:
            self.optimizer.zero_grad()
            
            # Forward Pass (Compute gaussians integrals)
            y_pred = self.model(p_rays, u_rays)
            
            # Loss
            loss = torch.mean((y_pred - y_true)**2)
            
            # L1 Regularization (sparsity)
            total_loss = loss + self.train_cfg.l1_reg * torch.mean(self.model.get_concentration())
            
            # Backward
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)

            if it % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
            
            if self.is_densify_it(it):
                self.model.densify_and_prune(self.optimizer)
            
            if current_loss < self.train_cfg.target_loss:
                pbar.write(f"Training ended at iteration: {it}")
                break

        pbar.close()

        print(f"Final Loss: {loss.item():.6f}\nTotal gaussians: {self.model.num_gaussians}")

        return loss_history
