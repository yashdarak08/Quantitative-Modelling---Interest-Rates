"""
pde_solver.py
-------------
Finite Difference PDE solver to price a European swaption under a stochastic short rate model.
This example uses an explicit finite difference scheme.
"""

import numpy as np

def solve_european_swaption(S_max, r_steps, T, t_steps, r0, strike):
    """
    Solves a simple PDE for a European swaption pricing problem.
    For illustration, we assume the PDE:
        ∂V/∂t + 0.5 * sigma^2 * r * ∂^2V/∂r^2 + a*(b - r)*∂V/∂r - r*V = 0
    with terminal condition V(r, T) = max(P(r) - strike, 0),
    where P(r) is a placeholder payoff function.

    Parameters:
    - S_max: Maximum short rate value (upper boundary for r).
    - r_steps: Number of grid steps for the r dimension.
    - T: Time to maturity.
    - t_steps: Number of time steps.
    - r0: Initial short rate.
    - strike: Strike rate for the swaption.

    Returns:
    - Option price corresponding to initial rate r0.
    """

    # Parameters (for demonstration, fixed parameters)
    sigma = 0.01
    a = 0.1
    b = 0.05

    dr = S_max / r_steps
    dt = T / t_steps

    # Create grid for r and t
    r_grid = np.linspace(0, S_max, r_steps + 1)
    V = np.zeros((r_steps + 1, t_steps + 1))

    # Terminal condition: payoff at maturity
    # For a European call-like swaption, we use payoff: max(r - strike, 0)
    V[:, -1] = np.maximum(r_grid - strike, 0)

    # Finite difference method (explicit scheme) backward in time
    for j in range(t_steps - 1, -1, -1):
        for i in range(1, r_steps):
            # Second derivative in r (central difference)
            d2V_dr2 = (V[i+1, j+1] - 2*V[i, j+1] + V[i-1, j+1]) / (dr**2)
            # First derivative in r (central difference)
            dV_dr = (V[i+1, j+1] - V[i-1, j+1]) / (2*dr)
            
            # Explicit finite difference update
            V[i, j] = V[i, j+1] + dt * (
                0.5 * sigma**2 * r_grid[i] * d2V_dr2 +
                a * (b - r_grid[i]) * dV_dr -
                r_grid[i] * V[i, j+1]
            )
        
        # Boundary conditions
        V[0, j] = 0  # At r=0, option value is 0
        V[-1, j] = V[-2, j]  # Neumann boundary condition at r=S_max

    # Interpolate to find the value at initial rate r0
    price = np.interp(r0, r_grid, V[:, 0])
    return price
