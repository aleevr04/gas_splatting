import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from typing import cast

THEME = {
    # Main Application Colors
    'bg': '#f3f4f6',                
    'fg': '#374151',                
    'header_bg': '#ffffff',
    'border': '#e5e7eb',
    
    # Typography
    'font_family': 'Segoe UI',
    'title': '#1f2937',             
    'title_size': '16pt',
    'text_light': '#8b92a0',        # For subtle banner labels
    
    # Accents & Sliders
    'accent': '#4f46e5',            # Primary Indigo
    'accent_dark': '#3730a3',       # Slider handle border
    'slider_groove': '#e5e7eb',
    'slider_border': '#d1d5db',
    'slider_add': '#cbd5e1',        # Unfilled slider track
    
    # Plotting Visuals
    'colormap': 'turbo',            
    'scatter_brush': (99, 102, 241, 0),        
    'scatter_pen_orig': (255, 255, 255, 255),  
    'scatter_pen_split': (0, 255, 255, 255),   
    'scatter_size': 12,
    'crosshair': (255, 255, 255, 60),          
    'contour': (255, 255, 255, 100),           
    
    # Layout & Dimensions
    'window_size_ratio': 0.5,                 
    'min_window_width': 600,
    'min_window_height': 600,
    'margin': 10,
}

class SplitVisualizer:
    def __init__(self):
        # Initial mathematical parameters
        self.init_sx = 4.0
        self.init_sy = 1.0
        self.init_theta = 30.0
        self.init_rho = 1.0
        self.init_c = 0.5
        self.mu_orig = np.array([0.0, 0.0])

        # Grid configuration
        self.grid_min, self.grid_max = -20, 20
        self.res = 100 
        
        self.x_lin = np.linspace(self.grid_min, self.grid_max, self.res)
        self.y_lin = np.linspace(self.grid_min, self.grid_max, self.res)
        self.X, self.Y = np.meshgrid(self.x_lin, self.y_lin, indexing='ij')

        # Application initialization
        self._init_qt()
        self._init_window()
        self._setup_plots()
        self._setup_controls()
        
        self.update_visuals()

    def _init_qt(self):
        self.app = cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        pg.setConfigOption('background', THEME['bg'])
        pg.setConfigOption('foreground', THEME['fg'])
        pg.setConfigOptions(antialias=True)

        self.base_font = QtGui.QFont(THEME['font_family'], 11)
        self.base_font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
        self.app.setFont(self.base_font)

    def _init_window(self):
        self.main_win = QtWidgets.QWidget()
        self.main_win.setWindowTitle("Interactive Split Visualizer")
        
        self.layout = QtWidgets.QVBoxLayout(self.main_win)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win, stretch=3)

        self.controls_widget = QtWidgets.QWidget()
        self.controls_widget.setObjectName("ControlsWidget")
        
        self.controls_layout = QtWidgets.QGridLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(30, 20, 30, 20)
        self.layout.addWidget(self.controls_widget, stretch=1)

        # Apply standardized theme colors to the controls via QSS
        self.controls_widget.setStyleSheet(f"""
            QWidget#ControlsWidget {{
                background-color: {THEME['bg']};
                border-top: 2px solid {THEME['border']};
            }}
            QLabel {{ 
                color: {THEME['fg']}; 
                font-weight: bold; 
            }}
            QSlider::groove:horizontal {{ 
                border: 1px solid {THEME['slider_border']}; 
                background: {THEME['slider_groove']}; 
                height: 8px; 
                border-radius: 4px; 
            }}
            QSlider::sub-page:horizontal {{ 
                background: {THEME['accent']}; 
                border-radius: 4px; 
            }}
            QSlider::add-page:horizontal {{ 
                background: {THEME['slider_add']}; 
                border-radius: 4px; 
            }}
            QSlider::handle:horizontal {{ 
                background: {THEME['accent']}; 
                border: 2px solid {THEME['accent_dark']}; 
                width: 16px; 
                height: 16px; 
                margin: -4px 0; 
                border-radius: 8px; 
            }}
        """)

        # Responsive window sizing
        screen_rect = self.app.primaryScreen().availableGeometry()
        target_width = int(screen_rect.width() * THEME['window_size_ratio'])
        final_width = max(target_width, THEME['min_window_width'])
        final_height = max(int(final_width * 0.75), THEME['min_window_height'])

        self.main_win.resize(final_width, final_height)
        self.main_win.move((screen_rect.width() - final_width) // 2, (screen_rect.height() - final_height) // 2)
        self.main_win.show()

    def make_title(self, text):
        return f'<b style="font-family: {THEME["font_family"]}, sans-serif;">{text}</b>'

    def _setup_plots(self): 
        # Header Banner Setup
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
        self.layout.insertWidget(0, self.header_banner, stretch=0)
        
        # Original Plot
        self.p_orig = self.win.addPlot(row=1, col=0)
        self.p_orig.setTitle(self.make_title("Original Gaussian"), size=THEME['title_size'], color=THEME['title'])
        self.img_orig = pg.ImageItem()
        self.p_orig.addItem(self.img_orig)
        self.scatter_orig = pg.ScatterPlotItem(size=THEME['scatter_size'], pen=pg.mkPen(color=THEME['scatter_pen_orig'], width=2), symbol='x')
        self.p_orig.addItem(self.scatter_orig)

        # Split Plot
        self.p_split = self.win.addPlot(row=1, col=1)
        self.p_split.setTitle(self.make_title("Split Gaussians"), size=THEME['title_size'], color=THEME['title'])
        self.img_split = pg.ImageItem()
        self.p_split.addItem(self.img_split)
        self.scatter_split = pg.ScatterPlotItem(size=THEME['scatter_size'], pen=pg.mkPen(color=THEME['scatter_pen_split'], width=2), symbol='x')
        self.p_split.addItem(self.scatter_split)

        # Setup Contours (Isocurves)
        self.iso_orig = [pg.IsocurveItem(pen=pg.mkPen(color=THEME['contour'], width=1.5)) for _ in range(3)]
        self.iso_split = [pg.IsocurveItem(pen=pg.mkPen(color=THEME['contour'], width=1.5)) for _ in range(3)]
        
        for iso_o, iso_s in zip(self.iso_orig, self.iso_split):
            iso_o.setParentItem(self.img_orig) 
            iso_s.setParentItem(self.img_split)

        # Apply common settings to both plots
        cmap = pg.colormap.get(THEME['colormap'])
        
        for p, img in zip([self.p_orig, self.p_split], [self.img_orig, self.img_split]):
            img.setColorMap(cmap)
                
            p.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=THEME['crosshair'], style=QtCore.Qt.PenStyle.DashLine)))
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color=THEME['crosshair'], style=QtCore.Qt.PenStyle.DashLine)))
            
            p.setAspectLocked(True)
            p.setXRange(self.grid_min, self.grid_max, padding=0.0)
            p.setYRange(self.grid_min, self.grid_max, padding=0.0)
            p.hideAxis('left')
            p.hideAxis('bottom')
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.setContentsMargins(THEME['margin'], THEME['margin'], THEME['margin'], THEME['margin'])

    def create_slider(self, row, label_text, min_val, max_val, init_val, scale_factor):
        lbl_name = QtWidgets.QLabel(label_text)
        lbl_name.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.controls_layout.addWidget(lbl_name, row, 0)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val * scale_factor))
        slider.setMaximum(int(max_val * scale_factor))
        slider.setValue(int(init_val * scale_factor))
        self.controls_layout.addWidget(slider, row, 1)

        lbl_val = QtWidgets.QLabel(str(init_val))
        lbl_val.setMinimumWidth(50)
        self.controls_layout.addWidget(lbl_val, row, 2)

        slider.valueChanged.connect(self.update_visuals)
        return slider, lbl_val, scale_factor

    def _setup_controls(self):
        self.controls_layout.setColumnStretch(1, 1)
        self.controls_layout.setHorizontalSpacing(20)

        self.slider_c, self.val_c, self.scale_c = self.create_slider(0, "Shift (c):", 0.01, 0.95, self.init_c, 100)
        self.slider_sx, self.val_sx, self.scale_sx = self.create_slider(1, "Scale X:", 1.0, 8.0, self.init_sx, 10)
        self.slider_sy, self.val_sy, self.scale_sy = self.create_slider(2, "Scale Y:", 0.5, 5.0, self.init_sy, 10)
        self.slider_th, self.val_th, self.scale_th = self.create_slider(3, "Rotation:", 0.0, 180.0, self.init_theta, 1)

    def compute_gaussian(self, mu, sx, sy, theta_deg, rho):
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        X_shift = self.X - mu[0]
        Y_shift = self.Y - mu[1]

        X_rot = X_shift * cos_t + Y_shift * sin_t
        Y_rot = -X_shift * sin_t + Y_shift * cos_t

        sx = max(sx, 1e-5)
        sy = max(sy, 1e-5)
        
        exponent = -0.5 * ((X_rot / sx)**2 + (Y_rot / sy)**2)
        return rho * np.exp(exponent)

    def calculate_gaussians(self, sx, sy, theta_deg, rho, c):
        theta = np.radians(theta_deg)
        u = np.array([np.cos(theta), np.sin(theta)])
        
        shift_dist = c * sx
        mu_1 = self.mu_orig + shift_dist * u
        mu_2 = self.mu_orig - shift_dist * u
        
        sx_new = sx * np.sqrt(1 - c**2)
        rho_new = rho / (2.0 * np.sqrt(1 - c**2))
        
        Z_orig = self.compute_gaussian(self.mu_orig, sx, sy, theta_deg, rho)
        Z_1 = self.compute_gaussian(mu_1, sx_new, sy, theta_deg, rho_new)
        Z_2 = self.compute_gaussian(mu_2, sx_new, sy, theta_deg, rho_new)
        
        Z_combined = Z_1 + Z_2
        return Z_orig, Z_combined, mu_1, mu_2, sx_new, rho_new

    def update_visuals(self):
        # Fetch current slider values
        c = self.slider_c.value() / self.scale_c
        sx = self.slider_sx.value() / self.scale_sx
        sy = self.slider_sy.value() / self.scale_sy
        th = self.slider_th.value() / self.scale_th

        # Update UI labels
        self.val_c.setText(f"{c:.2f}")
        self.val_sx.setText(f"{sx:.2f}")
        self.val_sy.setText(f"{sy:.2f}")
        self.val_th.setText(f"{th:.1f}")

        # Compute math
        Z_orig, Z_combined, mu_1, mu_2, sx_new, rho_new = self.calculate_gaussians(sx, sy, th, self.init_rho, c)
        rmse = np.sqrt(np.mean((Z_orig - Z_combined)**2))

        Z_orig_img = Z_orig.astype(np.float32)
        Z_combined_img = Z_combined.astype(np.float32)

        max_val = float(max(Z_orig_img.max(), Z_combined_img.max()))
        if max_val <= 0: 
            max_val = 1.0

        # Define boundary rect
        grid_rect = QtCore.QRectF(self.grid_min, self.grid_min, self.grid_max - self.grid_min, self.grid_max - self.grid_min)

        # Update Heatmaps
        self.img_orig.setImage(Z_orig_img, autoLevels=False)
        self.img_orig.setRect(grid_rect)
        self.img_orig.setLevels([0, max_val])

        self.img_split.setImage(Z_combined_img, autoLevels=False)
        self.img_split.setRect(grid_rect)
        self.img_split.setLevels([0, max_val])

        # Update Contours (15%, 50%, and 85% thresholds)
        levels = [max_val * 0.15, max_val * 0.50, max_val * 0.85]
        for i, lvl in enumerate(levels):
            self.iso_orig[i].setData(Z_orig_img)
            self.iso_orig[i].setLevel(lvl)
            self.iso_split[i].setData(Z_combined_img)
            self.iso_split[i].setLevel(lvl)

        # Update Scatter points
        self.scatter_orig.setData([self.mu_orig[0]], [self.mu_orig[1]])
        self.scatter_split.setData([mu_1[0], mu_2[0]], [mu_1[1], mu_2[1]])

        # Format and update header banner
        accent = THEME['accent']
        text_color = THEME['text_light']
        spacer = '&nbsp;' * 10
        
        formatted_text = (
            f'<span style="color: {text_color};">New Concentration:</span> '
            f'<b style="color: {accent};">{rho_new:.3f}</b>{spacer}'
            f'<span style="color: {text_color};">New X Scale:</span> '
            f'<b style="color: {accent};">{sx_new:.3f}</b>{spacer}'
            f'<span style="color: {text_color};">RMSE:</span> '
            f'<b style="color: {accent};">{rmse:.4f}</b>'
        )
        self.header_banner.setText(formatted_text)

    def run(self):
        sys.exit(self.app.exec())

if __name__ == '__main__':
    vis = SplitVisualizer()
    vis.run()