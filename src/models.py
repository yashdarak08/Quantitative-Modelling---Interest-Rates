"""
models.py
---------
This module implements the Vasicek and CIR models for simulating interest rate dynamics.
"""

import numpy as np

class VasicekModel:
    def __init__(self, a, b, sigma, r0):
        """
        Vasicek model initialization.
        
        Parameters:
        - a: Speed of mean reversion.
        - b: Long-term mean level.
        - sigma: Volatility.
        - r0: Initial interest rate.
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def simulate(self, n_paths, n_steps, dt):
        """
        Simulate the Vasicek model.
        
        Returns:
        - rates: Array of simulated paths of shape (n_paths, n_steps+1).
        """
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0
        
        for i in range(1, n_steps + 1):
            dr = self.a * (self.b - rates[:, i-1]) * dt + self.sigma * np.sqrt(dt) * np.random.randn(n_paths)
            rates[:, i] = rates[:, i-1] + dr
        return rates

class CIRModel:
    def __init__(self, a, b, sigma, r0):
        """
        CIR model initialization.
        
        Parameters:
        - a: Speed of mean reversion.
        - b: Long-term mean level.
        - sigma: Volatility.
        - r0: Initial interest rate.
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def simulate(self, n_paths, n_steps, dt):
        """
        Simulate the CIR model ensuring non-negative rates.
        
        Returns:
        - rates: Array of simulated paths of shape (n_paths, n_steps+1).
        """
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0
        
        for i in range(1, n_steps + 1):
            sqrt_r = np.sqrt(np.maximum(rates[:, i-1], 0))
            dr = self.a * (self.b - rates[:, i-1]) * dt + self.sigma * sqrt_r * np.sqrt(dt) * np.random.randn(n_paths)
            rates[:, i] = np.maximum(rates[:, i-1] + dr, 0)  # Ensure non-negative
        return rates
