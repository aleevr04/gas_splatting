import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plot_utils import set_publication_style

def render_single_gaussian(grid_pos, mu, cov, concentration):
    """Renders a 2D Gaussian given its properties."""
    d = grid_pos - mu
    d = d.unsqueeze(-1)
    
    sig_inv = torch.linalg.inv(cov)
    dist = torch.matmul(d.transpose(-1, -2), torch.matmul(sig_inv, d)).squeeze()
    
    return concentration * torch.exp(-0.5 * dist)

def main():
    device = torch.device("cpu")
    map_size = 20.0
    res = 200  # High resolution for accurate RMSE
    
    # 1. Create the Grid
    x = torch.linspace(-map_size/2, map_size/2, res, device=device)
    y = torch.linspace(-map_size/2, map_size/2, res, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_pos = torch.stack([X, Y], dim=-1)

    # 2. Define the Original Gaussian (Stretched along the X-axis)
    mu_orig = torch.tensor([0.0, 0.0], device=device)
    sx_orig, sy_orig = 6.0, 1.0 
    cov_orig = torch.tensor([[sx_orig**2, 0.0], [0.0, sy_orig**2]], device=device)
    rho_orig = 1.0
    
    img_orig = render_single_gaussian(grid_pos, mu_orig, cov_orig, rho_orig)

    # 3. Prepare the sweep for factor 'c'
    c_values = np.linspace(0.0, 0.95, 50)
    errors = []
    
    saved_images = {}
    c_to_save = [0.4, 0.6, 0.9]

    # 4. Experimental loop applying the splitting formulas
    for c in c_values:
        # Scaling and concentration formulas (Law of Total Variance and Mass Conservation)
        sx_new = sx_orig * np.sqrt(1 - c**2)
        sy_new = sy_orig
        rho_new = rho_orig / (2.0 * np.sqrt(1 - c**2))
        
        cov_new = torch.tensor([[sx_new**2, 0.0], [0.0, sy_new**2]], device=device)
        
        # Displacement along the X-axis (u = [1, 0])
        mu_1 = torch.tensor([ c * sx_orig, 0.0], device=device)
        mu_2 = torch.tensor([-c * sx_orig, 0.0], device=device)
        
        # Render the sum of the two new Gaussians
        img_split_1 = render_single_gaussian(grid_pos, mu_1, cov_new, rho_new)
        img_split_2 = render_single_gaussian(grid_pos, mu_2, cov_new, rho_new)
        img_combined = img_split_1 + img_split_2
        
        # Calculate Error (RMSE)
        rmse = torch.sqrt(torch.mean((img_orig - img_combined)**2)).item()
        errors.append(rmse)
        
        # Save sample images for visualization
        if any(np.isclose(c, val, atol=0.01) for val in c_to_save) and len(saved_images) < 3:
            saved_images[f"c={c:.2f}"] = img_combined.numpy()

    # 5. Generate Plots (Publication Ready for LaTeX)
    print(f"\nSaving results for the Long-Axis Split experiment. Sampled 'c' values: {c_to_save}...")
    
    # Configure Matplotlib for LaTeX-style publication
    set_publication_style()

    # Slightly reduced figure size so relative font size appears larger when scaled in LaTeX
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2]) # Give slightly more vertical room to the line plot

    # Panel 1: Original
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_orig.numpy(), extent=(-10, 10, -10, 10), cmap='jet', origin='lower')
    ax_orig.set_title("Original Gaussian")
    ax_orig.axis('off')

    # Top Panels: Different 'c' splits
    for i, (title, img) in enumerate(saved_images.items()):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(img, extent=(-10, 10, -10, 10), cmap='jet', origin='lower', vmax=float(img_orig.max()))
        ax.set_title(f"Split ({title})")
        ax.axis('off')

    # Bottom Panel: Error Graph
    ax_err = fig.add_subplot(gs[1, :])
    # Increased line width and marker size for visibility
    ax_err.plot(c_values, errors, 'b-', linewidth=3.0, marker='o', markersize=6)
    ax_err.set_title("Reconstruction Error (RMSE) vs. Displacement Factor (c)", pad=15)
    ax_err.set_xlabel("Factor 'c' (Displacement proportion along the major axis)")
    ax_err.set_ylabel("RMSE")
    ax_err.grid(True, linestyle='--', alpha=0.7, linewidth=1.0)
    
    # Find and indicate the analytical minimum
    min_idx = np.argmin(errors)
    c_opt = c_values[min_idx]
    # Thicker vertical line
    ax_err.axvline(x=c_opt, color='r', linestyle='--', linewidth=2.5, label=f"Minimum error: c={c_opt:.2f}")
    ax_err.legend(loc='upper right')

    # Improve spacing between subplots
    plt.tight_layout(pad=2.0)
    
    # Save logic
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'long_axis_optimization.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Using a higher DPI ensures crispness when printing the LaTeX PDF
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[+] Plot saved successfully in: {save_path}")

if __name__ == "__main__":
    main()