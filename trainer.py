import os
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import imageio.v3 as iio
from pyqtgraph.Qt import QtWidgets, QtCore
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, cast

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import SimulationData


@dataclass
class TrainingResults:
    loss_history: List[float] = field(default_factory=list)
    densify_history: Dict[int, dict] = field(default_factory=dict)
    rmse_history: Dict[int, float] = field(default_factory=dict)


class LiveVisualizer:
    def __init__(self, cfg: Config):
        self.map_size = cfg.sim.map_size
        self.cell_size = cfg.sim.cell_size
        self.history = []

        # Initialize Qt
        self.app = cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())
        if self.app is None:
            # If no QApplication exists, create one
            self.app = QtWidgets.QApplication([])

        # --- GLOBAL STYLES (Modern Bright Theme) ---
        pg.setConfigOption('background', '#f3f4f6') # Soft light gray background
        pg.setConfigOption('foreground', '#374151') # Dark slate for axes and labels
        pg.setConfigOptions(antialias=True) # Antialiaing

        font = self.app.font()
        font.setPointSize(12) 
        self.app.setFont(font)

        # --- WINDOW ---
        # This is the widget that holds everything
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Gas Splatting")
        
        # Compute window size based on map size
        screen_rect = self.app.primaryScreen().availableGeometry()
        target_height = int(screen_rect.height() * 0.80)
        map_aspect_ratio = cfg.sim.map_size[0] / cfg.sim.map_size[1]
        top_row_height = target_height * 0.6
        ideal_width = int(3 * (top_row_height * map_aspect_ratio))
        
        min_width, min_height = 800, 600
        final_width = max(ideal_width, min_width)
        final_height = max(target_height, min_height)

        self.win.setMinimumSize(min_width, min_height)
        self.win.resize(final_width, final_height)

        # Center the window on the screen
        x_pos = (screen_rect.width() - final_width) // 2
        y_pos = (screen_rect.height() - final_height) // 2
        self.win.move(x_pos, y_pos)

        # ---------------------------------------------------
        # ROW 0, COL 0: Gaussians' positions (Scatter)
        # ---------------------------------------------------
        self.p_map = self.win.addPlot(row=0, col=0)
        self.p_map.setTitle("Gaussians' positions", size='16pt', color='#111827')
        self.p_map.setXRange(0, cfg.sim.map_size[0], padding=0)
        self.p_map.setYRange(0, cfg.sim.map_size[1], padding=0)
        self.p_map.setAspectLocked(True)
        self.p_map.showGrid(x=True, y=True, alpha=0.15) # Softer grid for bright theme
        
        self.scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(239, 68, 68, 200) # Vibrant coral/red
        )
        self.p_map.addItem(self.scatter)

        # Updated text overlay for light theme
        self.text_item = pg.TextItem(text="", color='#111827', fill=pg.mkBrush(255, 255, 255, 200))
        text_font = self.text_item.textItem.font()
        text_font.setPointSize(13)
        text_font.setBold(True)
        self.text_item.setFont(text_font)
        self.p_map.addItem(self.text_item)
        self.text_item.setPos(cfg.sim.map_size[0] * 0.05, cfg.sim.map_size[1] * 0.95)

        # ---------------------------------------------------
        # ROW 0, COL 1: Estimated Gas Map (ImageItem)
        # ---------------------------------------------------
        self.p_render = self.win.addPlot(row=0, col=1)
        self.p_render.setTitle("Estimated Gas Map", size='16pt', color='#111827')
        self.p_render.setXRange(0, cfg.sim.map_size[0])
        self.p_render.setYRange(0, cfg.sim.map_size[1])
        self.p_render.setAspectLocked(True)
        
        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(pg.colormap.get('plasma'))
        self.p_render.addItem(self.img_item)

        self.img_item.setRect(QtCore.QRectF(0, 0, cfg.sim.map_size[0], cfg.sim.map_size[1]))

        # ---------------------------------------------------
        # ROW 0, COL 2:  Ground Truth (ImageItem)
        # ---------------------------------------------------
        self.p_gt = self.win.addPlot(row=0, col=2)
        self.p_gt.setTitle("Ground Truth", size='16pt', color='#111827')
        self.p_gt.setXRange(0, cfg.sim.map_size[0])
        self.p_gt.setYRange(0, cfg.sim.map_size[1])
        self.p_gt.setAspectLocked(True)
        
        self.img_gt_item = pg.ImageItem()
        self.img_gt_item.setColorMap(pg.colormap.get('plasma'))
        self.p_gt.addItem(self.img_gt_item)
        self.img_gt_item.setRect(QtCore.QRectF(0, 0, cfg.sim.map_size[0], cfg.sim.map_size[1]))

        # ---------------------------------------------------
        # ROW 1, COL 0,1,2: Loss History (Spans all columns)
        # ---------------------------------------------------
        self.p_loss = self.win.addPlot(row=1, col=0, colspan=3)
        self.p_loss.setTitle("Loss History", size='16pt', color='#111827')
        self.p_loss.setLabel('bottom', "Iteration")
        self.p_loss.setLabel('left', "Total Loss")
        self.p_loss.setLogMode(x=False, y=True)
        self.p_loss.showGrid(x=True, y=True, alpha=0.15) 
        
        # Modern vivid blue for the loss curve
        self.loss_curve = self.p_loss.plot(pen=pg.mkPen('#3b82f6', width=3.0))

    def set_ground_truth(self, ground_truth: np.ndarray):
        if ground_truth is not None:
            self.img_gt_item.setImage(ground_truth.T, autoLevels=True)
            self.img_gt_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))

    def update(self, iteration, loss_history, model):
        # Update Loss
        valid_losses = [l for l in loss_history if not np.isnan(l) and not np.isinf(l)]
        x_data = np.arange(len(valid_losses))
        if valid_losses:
            self.loss_curve.setData(x_data, valid_losses)

        # Extract tensors
        pos = model.get_pos().detach().cpu().numpy()
        conc = model.get_concentration().detach().cpu().numpy()
        sizes = np.clip(conc * 50, 5, 30)

        # Update Gaussians Scatter
        self.scatter.setData(pos[:, 0], pos[:, 1], size=sizes)
        self.text_item.setText(f"Iter: {iteration} | Gaussians: {len(pos)}")

        # Update Rendered Map
        map_data = model.render_map(cell_size=self.cell_size).T
        self.img_item.setImage(map_data, autoLevels=True)
        self.img_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))

        self.app.processEvents()

        # Save state for GIF
        self.history.append({
            'it': iteration,
            'loss_x': x_data,
            'loss_y': valid_losses,
            'pos': pos.copy(),
            'sizes': sizes.copy(),
            'map': map_data.copy()
        })

    def save_gif(self, filepath="plots/training_evolution.gif"):
        if not self.history:
            print("[GIF] No frames stored in memory.")
            return
            
        print(f"[GIF] Generating GIF from {len(self.history)} frames...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        exporter = pyqtgraph.exporters.ImageExporter(self.win.scene())
        temp_dir = "plots/temp_pg_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_paths = []

        for i, state in enumerate(self.history):
            self.loss_curve.setData(state['loss_x'], state['loss_y'])
            self.scatter.setData(state['pos'][:, 0], state['pos'][:, 1], size=state['sizes'])
            self.text_item.setText(f"Iter: {state['it']} | Gaussians: {len(state['pos'])}")
            
            if state['map'] is not None:
                self.img_item.setImage(state['map'], autoLevels=True)
            
            self.app.processEvents()
            
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            exporter.export(frame_path)
            frame_paths.append(frame_path)

        frames = [iio.imread(p) for p in frame_paths]
        if frames:
            iio.imwrite(filepath, frames, duration=100, loop=0)
            print(f"[+] Training GIF saved in: {filepath}")
            
        shutil.rmtree(temp_dir, ignore_errors=True)
        self.win.close()


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

        pbar = tqdm(range(self.cfg.train.iterations), desc="Training", dynamic_ncols=True)        
        for it in pbar:
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
                    tqdm.write(f"Early stopping triggered at iteration {it} (EMA Loss stalled at {ema_loss:.5f})")
                    break 
            # ----------------------------

            if it % 100 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.5f}'})
            
            # --- Real time visualization ---
            if self.visualizer and it % 20 == 0:
                current_gas_map = self.model.render_map(cell_size=self.cfg.sim.cell_size)

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

            # --- Densification ---
            if self.is_densify_it(it):
                stats = self.model.densify_and_prune(self.optimizer)
                results.densify_history[it] = stats

                # Force a visualizer update to see the change
                if self.visualizer:
                    current_gas_map = self.model.render_map(cell_size=self.cfg.sim.cell_size)

                    self.visualizer.update(
                        iteration=it, 
                        loss_history=results.loss_history,
                        model=self.model
                    )
            # -------------------------

        pbar.close()
        
        if self.visualizer:
            # Save Training GIF
            os.makedirs("plots", exist_ok=True)
            self.visualizer.save_gif()

        return results
