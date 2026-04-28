import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import multivariate_normal

# ==========================================
# 1. INITIAL PARAMETERS
# ==========================================
init_sx = 4.0
init_sy = 1.0
init_theta = 30.0
init_rho = 1.0
init_c = 0.5
mu_orig = np.array([0.0, 0.0])

# Grid setup (coarser grid = faster real-time rendering)
x, y = np.mgrid[-10:10:0.1, -10:10:0.1]
pos = np.dstack((x, y))

# ==========================================
# 2. CALCULATION FUNCTION
# ==========================================
def calculate_gaussians(sx, sy, theta_deg, rho, c):
    theta = np.radians(theta_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Original
    Sigma_orig = R @ np.array([[sx**2, 0], [0, sy**2]]) @ R.T
    norm_factor_orig = 2 * np.pi * sx * sy
    rv_orig = multivariate_normal(mu_orig, Sigma_orig)
    Z_orig = rho * rv_orig.pdf(pos) * norm_factor_orig
    
    # Split
    shift_dist = c * sx
    mu_1 = mu_orig + shift_dist * u
    mu_2 = mu_orig - shift_dist * u
    
    sx_new = sx * np.sqrt(1 - c**2)
    Sigma_new = R @ np.array([[sx_new**2, 0], [0, sy**2]]) @ R.T
    
    rho_new = rho / (2.0 * np.sqrt(1 - c**2))
    norm_factor_new = 2 * np.pi * sx_new * sy
    
    rv_new_1 = multivariate_normal(mu_1, Sigma_new)
    rv_new_2 = multivariate_normal(mu_2, Sigma_new)
    
    Z_combined = (rho_new * rv_new_1.pdf(pos) * norm_factor_new) + \
                 (rho_new * rv_new_2.pdf(pos) * norm_factor_new)
                 
    return Z_orig, Z_combined, mu_1, mu_2, sx_new, rho_new

# ==========================================
# 3. PLOT SETUP
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.35) # Make room for sliders

ax_orig, ax_split = axes
for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

Z_orig, Z_combined, mu_1, mu_2, sx_new, rho_new = calculate_gaussians(init_sx, init_sy, init_theta, init_rho, init_c)

# Use imshow for fast rendering
im_orig = ax_orig.imshow(np.rot90(Z_orig), extent=[-10, 10, -10, 10], cmap='inferno', vmin=0, vmax=init_rho)
im_split = ax_split.imshow(np.rot90(Z_combined), extent=[-10, 10, -10, 10], cmap='inferno', vmin=0, vmax=init_rho)

# Markers for centers
center_orig, = ax_orig.plot(0, 0, 'wx', markersize=8)
center_1, = ax_split.plot(mu_1[0], mu_1[1], 'cx', markersize=8)
center_2, = ax_split.plot(mu_2[0], mu_2[1], 'gx', markersize=8)

ax_orig.set_title('Original Gaussian')
ax_split.set_title(f'Split Gaussians\nRho: {rho_new:.3f}, Scale X: {sx_new:.3f}')

# ==========================================
# 4. SLIDERS
# ==========================================
ax_c = plt.axes((0.15, 0.20, 0.65, 0.03))
ax_sx = plt.axes((0.15, 0.15, 0.65, 0.03))
ax_sy = plt.axes((0.15, 0.10, 0.65, 0.03))
ax_theta = plt.axes((0.15, 0.05, 0.65, 0.03))

s_c = Slider(ax_c, 'Shift (c)', 0.01, 0.95, valinit=init_c)
s_sx = Slider(ax_sx, 'Scale X', 1.0, 8.0, valinit=init_sx)
s_sy = Slider(ax_sy, 'Scale Y', 0.5, 5.0, valinit=init_sy)
s_theta = Slider(ax_theta, 'Rotation', 0.0, 180.0, valinit=init_theta)

def update(val):
    c = s_c.val
    sx = s_sx.val
    sy = s_sy.val
    theta = s_theta.val
    rho = init_rho 
    
    # Catch the new values here too
    Z_orig, Z_combined, mu_1, mu_2, sx_new, rho_new = calculate_gaussians(sx, sy, theta, rho, c)
    
    im_orig.set_data(np.rot90(Z_orig))
    im_split.set_data(np.rot90(Z_combined))
    
    center_1.set_data([mu_1[0]], [mu_1[1]])
    center_2.set_data([mu_2[0]], [mu_2[1]])
    
    # Dynamically update the text on top of the split plot!
    ax_split.set_title(f'Split Gaussians\nRho: {rho_new:.3f}, Scale X: {sx_new:.3f}')
    
    fig.canvas.draw_idle()

s_c.on_changed(update)
s_sx.on_changed(update)
s_sy.on_changed(update)
s_theta.on_changed(update)

plt.show()