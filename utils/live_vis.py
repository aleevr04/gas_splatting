import os
import shutil
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import imageio.v3 as iio
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from typing import cast

from config import Config

# --- CENTRALIZED THEME ---
THEME = {
    'bg': '#f3f4f6',                # Soft light gray background
    'fg': '#374151',                # Dark slate for axes/labels
    'title': '#1f2937',             # Darker gray for titles
    'axis_text': '#6b7280',         # Soft gray for axis numbers
    'scatter_brush': (99, 102, 241, 180), # Indigo with transparency
    'scatter_pen': (255, 255, 255, 200),  # White outline
    'loss_line': '#4f46e5',         # Indigo matching line (swapped from #3b82f6 to match the scatter plot!)
    'loss_fill': (79, 70, 229, 30),   # Soft matching Indigo fill underneath the curve
    'text_overlay': '#111827',
    'grid_alpha': 0.15,
    'font_family': 'Segoe UI',
    'title_size': '18pt',
    'colormap': 'turbo'
}

class LiveVisualizer:
    def __init__(self, cfg: Config):
        self.map_size = cfg.sim.map_size
        self.cell_size = cfg.sim.cell_size
        self.history = []

        self._init_qt()
        self._init_window(cfg)
        self._setup_plots(cfg)

    def _init_qt(self):
        """Initializes Qt and applies global theme settings."""
        self.app = cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        pg.setConfigOption('background', THEME['bg'])
        pg.setConfigOption('foreground', THEME['fg'])
        pg.setConfigOptions(antialias=True)

        self.base_font = QtGui.QFont(THEME['font_family'], 13)
        self.base_font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
        self.app.setFont(self.base_font)

    def _init_window(self, cfg: Config):
        """Calculates and centers the main window."""
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Gas Splatting")
        
        screen_rect = self.app.primaryScreen().availableGeometry()
        
        target_width = int(screen_rect.width() * 0.80)
        single_map_width = target_width / 3.0
        
        map_aspect_ratio = cfg.sim.map_size[0] / cfg.sim.map_size[1]
        map_height = single_map_width / map_aspect_ratio
        
        loss_plot_height = 350  # Fixed pixel height allocation for the bottom row
        ui_margins = 50         # Buffer for titles, axes numbers, and window borders
        ideal_height = int(map_height + loss_plot_height + ui_margins)
        
        final_width = max(target_width, 1000)
        final_height = max(ideal_height, 650)

        self.win.setMinimumSize(1000, 650)
        self.win.resize(final_width, final_height)
        self.win.move((screen_rect.width() - final_width) // 2, (screen_rect.height() - final_height) // 2)

    def _setup_plots(self, cfg: Config):
        """Builds the internal widgets and applies the theme."""
        # Helper for HTML titles
        def make_title(text):
            return f'<b style="font-family: {THEME["font_family"]}, sans-serif;">{text}</b>'

        self.top_layout = self.win.addLayout(row=0, col=0)
        self.bottom_layout = self.win.addLayout(row=1, col=0)

        # 1. Gaussians Scatter (Col 0)
        self.p_gaussians = self.top_layout.addPlot(row=0, col=0)
        self.p_gaussians.setTitle(make_title("Gaussians' positions"), size=THEME['title_size'], color=THEME['title'])
        self.p_gaussians.showGrid(x=True, y=True, alpha=THEME['grid_alpha'])

        axis_font = QtGui.QFont(THEME['font_family'], 10)
        for axis in ['bottom', 'left']:
            self.p_gaussians.getAxis(axis).setTickFont(axis_font)
            self.p_gaussians.getAxis(axis).setTextPen(THEME['axis_text'])
        
        self.scatter = pg.ScatterPlotItem(
            brush=pg.mkBrush(color=THEME['scatter_brush']),
            pen=pg.mkPen(color=THEME['scatter_pen'], width=1)
        )
        self.p_gaussians.addItem(self.scatter)

        # Text Overlay
        self.text_item = pg.TextItem(text="", color=THEME['text_overlay'], fill=pg.mkBrush(255, 255, 255, 200))
        text_font = QtGui.QFont(THEME['font_family'], 13, QtGui.QFont.Weight.Bold)
        self.text_item.setFont(text_font)
        self.p_gaussians.addItem(self.text_item)
        self.text_item.setPos(cfg.sim.map_size[0] * 0.10, cfg.sim.map_size[1] * 0.90)

        # 2. Estimated Map (Col 1)
        self.p_map = self.top_layout.addPlot(row=0, col=1)
        self.p_map.setTitle(make_title("Estimated Map"), size=THEME['title_size'], color=THEME['title'])
        self.p_map.hideAxis('left')
        self.p_map.hideAxis('bottom')
        
        self.img_map_item = pg.ImageItem()
        self.p_map.addItem(self.img_map_item)

        # 3. Ground Truth (Col 2)
        self.p_gt = self.top_layout.addPlot(row=0, col=2)
        self.p_gt.setTitle(make_title("Ground Truth"), size=THEME['title_size'], color=THEME['title'])
        self.p_gt.hideAxis('left')
        self.p_gt.hideAxis('bottom')
        
        self.img_gt_item = pg.ImageItem()
        self.p_gt.addItem(self.img_gt_item)

        # 4. Colorbar (Col 3)
        cmap = pg.colormap.get(THEME['colormap'])
        self.img_map_item.setColorMap(cmap)
        self.img_gt_item.setColorMap(cmap)

        self.cbar = pg.ColorBarItem(interactive=False, colorMap=cmap, width=15)
        self.cbar.setMaximumHeight(360)
        self.cbar.setImageItem(self.img_gt_item)
        self.cbar.getAxis('right').setTickFont(QtGui.QFont(THEME['font_family'], 11))
        self.top_layout.addItem(self.cbar, row=0, col=3)
        self.top_layout.layout.setAlignment(self.cbar, QtCore.Qt.AlignmentFlag.AlignVCenter)

        # 5. Loss Curve (Row 1)
        self.p_loss = self.bottom_layout.addPlot(row=0, col=0)
        self.p_loss.setTitle(make_title("Loss History"), size=THEME['title_size'], color=THEME['title'])
        self.p_loss.setLabel('bottom', "Iteration")
        self.p_loss.setLogMode(x=False, y=True)
        self.p_loss.showGrid(x=True, y=True, alpha=THEME['grid_alpha'])
        self.p_loss.getAxis('left').enableAutoSIPrefix(False) # Force Y-axis to always show the real value

        axis_font = QtGui.QFont(THEME['font_family'], 10)
        for axis in ['bottom', 'left']:
            self.p_loss.getAxis(axis).setTickFont(axis_font)
            self.p_loss.getAxis(axis).setTextPen(THEME['axis_text'])
        
        self.loss_curve = self.p_loss.plot(
            pen=pg.mkPen(color=THEME['loss_line'], width=2.5), 
            fillLevel=None, 
            fillBrush=pg.mkBrush(THEME['loss_fill'])
        )

        for p in [self.p_gaussians, self.p_map, self.p_gt]:
            p.setXRange(0, cfg.sim.map_size[0], padding=0.0)
            p.setYRange(0, cfg.sim.map_size[1], padding=0.0)
            p.setAspectLocked(True)

    def set_ground_truth(self, ground_truth: np.ndarray):
        """Sets the static ground truth image and stores its levels to lock the color scale."""
        if ground_truth is not None:
            self.img_gt_item.setImage(ground_truth.T, autoLevels=True)
            self.img_gt_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))
            self.gt_levels = (ground_truth.min(), ground_truth.max())

    def update(self, iteration, loss_history, model):
        """Extracts the latest parameters from the model, updates the UI components, and saves the frame state."""
        # Update Loss
        valid_losses = [l for l in loss_history if not np.isnan(l) and not np.isinf(l)]
        x_data = np.arange(len(valid_losses))
        if valid_losses:
            min_loss = min(valid_losses)
            min_loss_log = np.log10(min_loss)
            safe_bottom = min_loss_log - 0.1 
            
            self.loss_curve.setFillLevel(safe_bottom)
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

        if hasattr(self, 'gt_levels'):
            self.img_map_item.setImage(map_data, autoLevels=False)
            self.img_map_item.setLevels(self.gt_levels)
        else:
            self.img_map_item.setImage(map_data, autoLevels=True)
        self.img_map_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))

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
        """Replays the stored training history to export a GIF animation of the process."""
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
                self.img_map_item.setImage(state['map'], autoLevels=True)
            
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