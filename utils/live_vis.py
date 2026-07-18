import os
import shutil
import numpy as np
import pyqtgraph as pg
import subprocess
import queue
import threading
import imageio_ffmpeg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from typing import List, cast

from config import Config
from gs_model import GasSplattingModel

THEME = {
    # Application & Layout Colors
    'bg': '#f3f4f6',          # Main application background (soft gray)
    'fg': '#374151',          # Primary foreground for default text and labels (dark slate)
    'header_bg': '#ffffff',   # Background color specifically for the top HTML banner
    'border': '#e5e7eb',      # Color for dividers, borders, and layout separators

    # Typography
    'font_family': 'Segoe UI',
    'title': '#1f2937',         # Color for the plot titles
    'title_size': '18pt',         # Font size for plot titles
    'text_light': '#8b92a0',    # Muted text color for the static labels inside the banner
    'axis_text': '#6b7280',     # Color for the numerical tick values on the graphs
    
    # Plotting & Visuals
    'colormap': 'turbo',
    'grid_alpha': 0.10,   # Opacity (0.0 to 1.0) of the background grid in the loss plot

    # Accents & Data Points
    'accent': '#4f46e5',                # Primary brand color (Indigo) for banner numbers AND the loss curve line
    'accent_fill': (79, 70, 229, 30),     # Highly transparent version of the accent color for shading under the loss curve
    'scatter_brush': (99, 102, 241, 180), # Lighter indigo with transparency for the inner fill of Gaussian points
    'scatter_pen': (255, 255, 255, 200),  # Slightly transparent white for the outline border of Gaussian points
    
    # Gaussian Sizing Physics
    'scatter_size_scale': 80,
    'scatter_size_min': 10,
    'scatter_size_max': 60,
    
    # --- Window Dimensions ---
    'window_width_ratio': 0.5,
    'min_window_width': 900,
    'min_window_height': 600,
    'margin': 20,           # Margins around the plots and inside the layout
    'col_min_width': 200    # Minimum width for each of the 3 plot columns
}

class CleanLogAxis(pg.AxisItem):
    """Custom Y-axis that prevents overlapping text on log scales by only labeling 1, 2, and 5."""
    def tickStrings(self, values, scale, spacing):
        strings = super().tickStrings(values, scale, spacing)
        
        cleaned = []
        for v, s in zip(values, strings):
            real_val = 10 ** v
            leading_digit = f"{real_val:.3e}"[0]
            if leading_digit in ['1', '2', '5']:
                cleaned.append(s)
            else:
                cleaned.append("") 
                
        return cleaned

class LiveVisualizer:
    def __init__(self, cfg: Config):
        self.map_size = cfg.sim.map_size
        self.cell_size = cfg.sim.cell_size

        # Asynchronous frame queue for GIF generation
        self.frame_queue = queue.Queue()
        self.frame_count = 0
        self.temp_dir = "plots/temp_pg_frames"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Frame writer thread to handle saving frames without blocking the main UI
        self.writer_thread = threading.Thread(target=self._frame_writer, daemon=True)
        self.writer_thread.start()

        self._init_qt()
        self._init_window()
        self.app.processEvents()
        self._setup_plots()

    def _frame_writer(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            
            frame_path, qimage = item
            # Save the QImage to disk as PNG
            qimage.save(frame_path, "PNG")
            self.frame_queue.task_done()

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
        """Calculates and centers the main window, now using a central QWidget."""
        # Wrap everything in a main QWidget
        self.main_win = QtWidgets.QWidget()
        self.main_win.setWindowTitle("Real-time Gas Splatting")
        self.layout = QtWidgets.QVBoxLayout(self.main_win)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header Banner
        self.header_banner = QtWidgets.QLabel()
        self.header_banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.header_banner.setStyleSheet(f"""
            QLabel {{
                background-color: {THEME['header_bg']};
                font-family: '{THEME['font_family']}';
                font-size: 14pt;
                padding: 18px;
                border-bottom: 2px solid {THEME['border']};
            }}
        """)
        self.layout.addWidget(self.header_banner, stretch=0)

        # Pyqtgraph Canvas
        self.canvas = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.canvas, stretch=1)
        
        # Sizing Logic
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

        self.main_win.resize(final_width, final_height)
        self.main_win.move((screen_rect.width() - final_width) // 2, (screen_rect.height() - final_height) // 2)
        self.main_win.show()

    def _setup_plots(self):
        """Builds the internal widgets and applies the theme."""
        def make_title(text):
            return f'<b style="font-family: {THEME["font_family"]}, sans-serif;">{text}</b>'

        # Gaussians Scatter (Row 0, Col 0)
        self.p_gaussians = self.canvas.addPlot(row=0, col=0)
        self.p_gaussians.setTitle(make_title("Gaussians' positions"), size=THEME['title_size'], color=THEME['title'])
        self.p_gaussians.hideAxis('left')
        self.p_gaussians.hideAxis('bottom')
        
        self.scatter = pg.ScatterPlotItem(
            brush=pg.mkBrush(color=THEME['scatter_brush']),
            pen=pg.mkPen(color=THEME['scatter_pen'], width=1)
        )
        self.p_gaussians.addItem(self.scatter)

        # Estimated Map (Row 0, Col 1)
        self.p_map = self.canvas.addPlot(row=0, col=1)
        self.p_map.setTitle(make_title("Estimated Map"), size=THEME['title_size'], color=THEME['title'])
        self.p_map.hideAxis('left')
        self.p_map.hideAxis('bottom')
        
        self.img_map_item = pg.ImageItem()
        self.p_map.addItem(self.img_map_item)

        # Ground Truth (Row 0, Col 2)
        self.p_gt = self.canvas.addPlot(row=0, col=2)
        self.p_gt.setTitle(make_title("Ground Truth"), size=THEME['title_size'], color=THEME['title'])
        self.p_gt.hideAxis('left')
        self.p_gt.hideAxis('bottom')
        
        self.img_gt_item = pg.ImageItem()
        self.p_gt.addItem(self.img_gt_item)

        # Colormap for maps
        cmap = pg.colormap.get(THEME['colormap'])
        self.img_map_item.setColorMap(cmap)
        self.img_gt_item.setColorMap(cmap)

        # Loss Curve (Row 1, Cols 0-2)
        custom_y_axis = CleanLogAxis(orientation='left')
        self.p_loss = self.canvas.addPlot(row=1, col=0, colspan=3, axisItems={'left': custom_y_axis})
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
            pen=pg.mkPen(color=THEME['accent'], width=2.5),
            fillBrush=pg.mkBrush(THEME['accent_fill'])
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

        main_layout = self.canvas.ci.layout

        if main_layout is not None:
            total_w = max(0, int(self.canvas.size().width()))
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

        # Update Header Banner
        accent = THEME['accent']
        text_color = THEME['text_light']
        spacer = '&nbsp;' * 10
        
        formatted_text = (
            f'<span style="color: {text_color};">Iteration:</span> '
            f'<b style="color: {accent};">{iteration}</b>{spacer}'
            f'<span style="color: {text_color};">Gaussians:</span> '
            f'<b style="color: {accent};">{len(pos)}</b>'
        )
        self.header_banner.setText(formatted_text)

        # Update Rendered Map
        map_data = model.render_map(cell_size=self.cell_size).T
        if hasattr(self, 'gt_levels'):
            self.img_map_item.setImage(map_data, levels=self.gt_levels)
        else:
            self.img_map_item.setImage(map_data, autoLevels=True)
        self.img_map_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))

        self.app.processEvents()

        # Frame Capture for GIF Generation
        qimage = self.main_win.grab().toImage()
        frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_count:05d}.png")
        
        # Send the frame to the writer thread via the queue
        self.frame_queue.put((frame_path, qimage))
        self.frame_count += 1

    def save_gif(self, filepath="plots/training_evolution.gif"):
        """Waits for all frames to be written, then generates a GIF from the captured frames using ffmpeg."""
        if self.frame_count == 0:
            print("Could not generate GIF. No frames captured.")
            return
            
        print("Waiting for frame capture to complete...")
        # Send stop signal to the writer thread and wait for it to finish
        self.frame_queue.put(None)
        self.writer_thread.join()

        print(f"Generating GIF from {self.frame_count} frames...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create GIF from saved frames using ffmpeg
        input_pattern = f"{self.temp_dir}/frame_%05d.png"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run([
            ffmpeg_exe, '-y', '-framerate', '10', 
            '-loglevel', 'error',
            '-i', input_pattern,
            '-filter_complex', 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
            '-loop', '0', 
            filepath
        ], check=True)
        print(f"[+] Training GIF saved in: {filepath}")
            
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.main_win.close()