import torch
import numpy as np
import math
import unittest

from utils.gaussian_utils import compute_integral, compute_definite_integral

class TestIntegral(unittest.TestCase):
    def setUp(self):
        self.num_gaussians = 3
        self.num_beams = 5

        # Gaussians parameters
        self.means = torch.rand(self.num_gaussians, 2)
        L = torch.randn(self.num_gaussians, 2, 2)
        sigmas = torch.bmm(L, L.transpose(1, 2)) + torch.eye(2) * 0.1
        self.inv_sigmas = torch.inverse(sigmas)
        self.concentrations = torch.ones(self.num_gaussians)

        # Beams
        self.beams = torch.randn(self.num_beams, 2, 2) * 5

    def test_predicted_measurements_shape(self):
        """
        Tests compute_integral() with multiple gaussians and beams to check dimensionality problems.
        """

        with torch.no_grad():
            y_pred = compute_integral(self.means, self.inv_sigmas, self.concentrations, self.beams)

        self.assertEqual(y_pred.shape, (self.num_beams,))

    def test_integral_value(self):
        """
        Tests compute_integral() to check that the integral value is computed properly.
        """
        
        # Standard Gaussian
        mean = torch.tensor([[0.0, 0.0]])
        sigma = torch.eye(2).unsqueeze(0)
        inv_sigma = torch.inverse(sigma)

        # Ray along X-axis
        beam = torch.tensor([
            [[-1.0, 0.0], [1.0, 0.0]],
        ])

        y_pred = compute_integral(mean, inv_sigma, torch.tensor([1.0]), beam)
        y_real = np.sqrt(2*np.pi) # Expected value

        self.assertAlmostEqual(y_pred, y_real)

class TestDefiniteIntegral(unittest.TestCase):
    def setUp(self):
        self.num_gaussians = 3
        self.num_beams = 5
        
        # Gaussians parameters
        self.means = torch.rand(self.num_gaussians, 2)
        L = torch.randn(self.num_gaussians, 2, 2)
        sigmas = torch.bmm(L, L.transpose(1, 2)) + torch.eye(2) * 0.1
        self.inv_sigmas = torch.inverse(sigmas)
        self.concentrations = torch.ones(self.num_gaussians)
            
        # Random beams
        self.beams = torch.randn(self.num_beams, 2, 2) * 5

    def test_integral_value(self):
        """
        Checks that compute_definite_integral() result is accurate, comparing it to Riemann Sum's result.
        """

        y_analytical = compute_definite_integral(self.means, self.inv_sigmas, self.concentrations, self.beams)
        
        # ---- Compute integral using Riemann Sum ----
        num_steps = 50000
        t = torch.linspace(0, 1, num_steps).view(-1, 1) # (steps, 1)
        dt = 1.0 / num_steps
        
        y_numerical = torch.zeros(self.num_beams)
        
        for n in range(self.num_beams):
            S = self.beams[n, 0]
            E = self.beams[n, 1]
            v = E - S
            ray_length = torch.norm(v)
            
            points = S + t * v # (steps, 2)
            
            integral_sum = 0
            for k in range(self.num_gaussians):
                # Evaluate Gaussian k at each point
                diff = points - self.means[k] # (steps, 2)
                
                dist_sq = torch.sum((diff @ self.inv_sigmas[k]) * diff, dim=1) 
                
                # g(p) = rho * exp(-0.5 * dist_sq)
                g_vals = self.concentrations[k] * torch.exp(-0.5 * dist_sq)
                
                # Riemann Sum: Sum(g(p) * dt * ||v||)
                integral_sum += torch.sum(g_vals) * dt * ray_length
                
            y_numerical[n] = integral_sum
            
        torch.testing.assert_close(y_analytical, y_numerical, atol=1e-3, rtol=1e-3)

    def test_beam_length_independence(self):
        """
        Checks that beam length doesn't affect compute_definite_integral() result (if beam goes through same amount of concentration)
        """

        # Standard Gaussian
        pos = torch.tensor([[0.0, 0.0]]) 
        cov_inv = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # Inversa de la matriz Identidad
        concentration = torch.tensor([1.0]) 

        # 3 Beams that end at (0, 0)
        beams = torch.tensor([
            [[-10.0, 0.0], [0.0, 0.0]],   # Beam 1: Starts at x=-10
            [[-20.0, 0.0], [0.0, 0.0]],   # Beam 2: Starts at x=-20
            [[0.0, -15.0], [0.0, 0.0]]    # Beam 3: Starts at y=-15
        ])

        y_pred = compute_definite_integral(pos, cov_inv, concentration, beams)

        # All of them must be equal
        self.assertAlmostEqual(y_pred[0].item(), y_pred[1].item())
        self.assertAlmostEqual(y_pred[1].item(), y_pred[2].item())

        # Expected integral value
        self.assertAlmostEqual(y_pred[0].item(), math.sqrt(math.pi/2))

if __name__ == "__main__":
    unittest.main()