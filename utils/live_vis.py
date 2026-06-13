import os
import shutil
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import imageio.v3 as iio
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from typing import List, cast

from config import Config
from gs_model import GasSplattingModel

THEME = {
    'bg': '#f3f4f6',                # Soft light gray background
    'fg': '#374151',                # Dark slate for axes/labels
    'font_family': 'Segoe UI',
    'title': '#1f2937',             # Darker gray for titles
    'title_size': '18pt',
    'axis_text': '#6b7280',         # Soft gray for axis numbers
    'grid_alpha': 0.10,
    'colormap': 'turbo',
    # Scatter Style
    'scatter_brush': (99, 102, 241, 180), # Indigo with transparency
    'scatter_pen': (255, 255, 255, 200),  # White outline
    'scatter_size_scale': 80,
    'scatter_size_min': 10,
    'scatter_size_max': 60,
    # Loss Curve
    'loss_line': '#4f46e5',         # Indigo matching line
    'loss_fill': (79, 70, 229, 30),   # Soft matching Indigo fill underneath the curve
    # Layout / sizing
    'window_width_ratio': 0.5,
    'min_window_width': 900,
    'min_window_height': 600,
    'margin': 20,
    'col_min_width': 200
}

class CleanLogAxis(pg.AxisItem):
    """Custom Y-axis that prevents overlapping text on log scales by only labeling 1, 2, and 5."""
    def tickStrings(self, values, scale, spacing):
        strings = super().tickStrings(values, scale, spacing)
        
        cleaned = []
        for v, s in zip(values, strings):
            # Convert internal values back to real numbers.
            real_val = 10 ** v
            
            # Format to scientific notation (e.g., '8.000e-02') to safely grab the leading digit
            leading_digit = f"{real_val:.3e}"[0]
            
            # Only keep the text if it's a 1, 2, or 5
            if leading_digit in ['1', '2', '5']:
                cleaned.append(s)
            else:
                cleaned.append("")  # Return an empty string to hide the text but keep the line!
                
        return cleaned

class LiveVisualizer:
    def __init__(self, cfg: Config):
        self.map_size = cfg.sim.map_size
        self.cell_size = cfg.sim.cell_size
        self.history = []

        self._init_qt()
        self._init_window()
        self._setup_plots()

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

    def _init_window(self):
        """Calculates and centers the main window."""
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Gas Splatting")
        
        screen_rect = self.app.primaryScreen().availableGeometry()

        target_width = int(screen_rect.width() * THEME['window_width_ratio'])
        single_map_width = target_width / 3.0

        map_aspect_ratio = self.map_size[0] / self.map_size[1]
        map_height = single_map_width / map_aspect_ratio

        ideal_height = int(2 * map_height)

        final_width = max(target_width, THEME['min_window_width'])
        final_height = max(ideal_height, THEME['min_window_height'])
        max_height = int(screen_rect.height() * 0.9)
        final_height = min(final_height, max_height)

        self.win.setFixedSize(final_width, final_height)
        self.win.move((screen_rect.width() - final_width) // 2, (screen_rect.height() - final_height) // 2)

    def _setup_plots(self):
        """Builds the internal widgets and applies the theme."""
        # Helper for HTML titles
        def make_title(text):
            return f'<b style="font-family: {THEME["font_family"]}, sans-serif;">{text}</b>'

        # Header Label (Row 0, spans all columns)
        self.header_label = self.win.addLabel(row=0, col=0, colspan=3)

        # Gaussians Scatter (Row 1, Col 0)
        self.p_gaussians = self.win.addPlot(row=1, col=0)
        self.p_gaussians.setTitle(make_title("Gaussians' positions"), size=THEME['title_size'], color=THEME['title'])
        self.p_gaussians.hideAxis('left')
        self.p_gaussians.hideAxis('bottom')
        
        self.scatter = pg.ScatterPlotItem(
            brush=pg.mkBrush(color=THEME['scatter_brush']),
            pen=pg.mkPen(color=THEME['scatter_pen'], width=1)
        )
        self.p_gaussians.addItem(self.scatter)

        # Estimated Map (Row 1, Col 1)
        self.p_map = self.win.addPlot(row=1, col=1)
        self.p_map.setTitle(make_title("Estimated Map"), size=THEME['title_size'], color=THEME['title'])
        self.p_map.hideAxis('left')
        self.p_map.hideAxis('bottom')
        
        self.img_map_item = pg.ImageItem()
        self.p_map.addItem(self.img_map_item)

        # Ground Truth (Row 1, Col 2)
        self.p_gt = self.win.addPlot(row=1, col=2)
        self.p_gt.setTitle(make_title("Ground Truth"), size=THEME['title_size'], color=THEME['title'])
        self.p_gt.hideAxis('left')
        self.p_gt.hideAxis('bottom')
        
        self.img_gt_item = pg.ImageItem()
        self.p_gt.addItem(self.img_gt_item)

        # Colormap for maps (shared scale)
        cmap = pg.colormap.get(THEME['colormap'])
        self.img_map_item.setColorMap(cmap)
        self.img_gt_item.setColorMap(cmap)

        # Loss Curve (Row 2) - span across all columns
        custom_y_axis = CleanLogAxis(orientation='left')
        self.p_loss = self.win.addPlot(row=2, col=0, colspan=3, axisItems={'left': custom_y_axis})
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
            fillBrush=pg.mkBrush(THEME['loss_fill'])
        )
        self.p_loss.setContentsMargins(0, 0, THEME['margin'], 0)

        for p in [self.p_gaussians, self.p_map, self.p_gt]:
            # Force the exact physical dimensions
            p.setXRange(0, self.map_size[0], padding=0.0)
            p.setYRange(0, self.map_size[1], padding=0.0)
            p.setAspectLocked(True)

            # Left and right margins
            margin = THEME['margin']
            p.setContentsMargins(margin, 0, margin, 0)
            
            # Prevent the plots from independently resizing themselves 
            # when dynamic data (Gaussians/Estimated Map) gets close to the edges.
            p.getViewBox().disableAutoRange()
            
            # Disable user interaction completely
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons() # hide auto-scale button

        main_layout = self.win.ci.layout

        if main_layout is not None:
            # compute and set column minimum widths
            total_w = max(0, int(self.win.size().width()))
            col_w = max(THEME['col_min_width'], int(total_w / 3) - 2 * THEME['margin'])
            for col in range(3):
                main_layout.setColumnMinimumWidth(col, col_w)

    def set_ground_truth(self, ground_truth: np.ndarray):
        """Sets the static ground truth image and stores its levels to lock the color scale."""
        if ground_truth is not None:
            self.img_gt_item.setImage(ground_truth.T, autoLevels=True)
            self.img_gt_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))
            # Save levels and apply them to both images so both maps use identical color scale
            self.gt_levels = (float(ground_truth.min()), float(ground_truth.max()))
            self.img_gt_item.setLevels(self.gt_levels)
            self.img_map_item.setLevels(self.gt_levels)

    def update(self, iteration: int, loss_history: List[float], model: GasSplattingModel):
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
        sizes = np.clip(conc * THEME['scatter_size_scale'], THEME['scatter_size_min'], THEME['scatter_size_max'])

        # Update Gaussians Scatter
        self.scatter.setData(pos[:, 0], pos[:, 1], size=sizes)

        # Update Header Label
        self.header_label.setText(
            f"Iter: {iteration} | Gaussians: {len(pos)}", 
            color=THEME['title'], 
            size='16pt', 
            bold=True
        )

        # Update Rendered Map
        map_data = model.render_map(cell_size=self.cell_size).T
        if hasattr(self, 'gt_levels'):
            self.img_map_item.setImage(map_data, levels=self.gt_levels)
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
            self.header_label.setText(
                f"Iter: {state['it']} | Gaussians: {len(state['pos'])}", 
                color=THEME['title'], 
                size='16pt', 
                bold=True
            )
            
            if state['map'] is not None:
                # Use fixed levels when available so GIF frames share the same color scale
                if hasattr(self, 'gt_levels'):
                    self.img_map_item.setImage(state['map'], levels=self.gt_levels)
                else:
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