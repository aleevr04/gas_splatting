import torch
import numpy as np

from utils.gaussian_utils import compute_predicted_projections

def test_multiple_gaussians_and_rays(num_gaussians, num_rays):
    """
    Tests compute_predicted_projections() with multiple gaussians and rays to check dimensionality problems.
    """

    means = torch.rand(num_gaussians, 2, requires_grad=True)

    L = torch.randn(num_gaussians, 2, 2)
    sigmas = torch.bmm(L, L.transpose(1, 2)) + torch.eye(2)*0.1
    inv_sigmas = torch.inverse(sigmas) # Precomputamos la inversa
    inv_sigmas.requires_grad = True

    p_rays = torch.randn(num_rays, 2)
    u_rays = torch.randn(num_rays, 2)

    # Forward pass
    proj_pred = compute_predicted_projections(means, inv_sigmas, torch.ones(num_gaussians), p_rays, u_rays)
    print(f"Projection's shape: {proj_pred.shape}")

    # Backward pass
    loss = torch.sum(proj_pred**2) # Loss dummy
    loss.backward()

    print(f"Mean gradient shape: {means.grad.shape if means.grad else ""}")
    print(f"Covariance inverse gradient shape: {inv_sigmas.grad.shape if inv_sigmas.grad else ""}")

def test_integral_value():
    """
    Tests compute_predicted_projections() to check that the Gaussian integral is computed properly.
    """
    
    # Standard Gaussian
    mean = torch.tensor([[0.0, 0.0]])
    sigma = torch.eye(2).unsqueeze(0)
    inv_sigma = torch.inverse(sigma)

    # Ray along X-axis
    p_ray = torch.tensor([[-5.0, 0.0]])
    u_ray = torch.tensor([[1.0, 0.0]])

    proj_pred = compute_predicted_projections(mean, inv_sigma, torch.tensor([1.0]), p_ray, u_ray)
    proj_real = np.sqrt(2*np.pi) # Expected value

    print(f"Computed integral: {proj_pred.item()}")
    print(f"Expected integral: {proj_real}")

def main():
    test_multiple_gaussians_and_rays(5, 10)
    test_integral_value()

if __name__ == "__main__":
    main()