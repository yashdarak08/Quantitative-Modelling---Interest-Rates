"""
pde_solver.py
-------------
Finite Difference PDE solver for interest rate derivative pricing.
Implements multiple finite difference schemes:
- Explicit scheme
- Implicit scheme
- Crank-Nicolson scheme

Includes pricing for:
- European swaptions
- Zero-coupon bonds
- Bond options
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solve_european_swaption(S_max, r_steps, T, t_steps, r0, strike, 
                           a=0.1, b=0.05, sigma=0.01, method='crank-nicolson',
                           swap_maturity=5.0, swap_frequency=0.5):
    """
    Solves PDE for a European swaption pricing under the Vasicek model.
    
    Parameters:
    - S_max: Maximum short rate value (upper boundary).
    - r_steps: Number of grid steps for the r dimension.
    - T: Time to option expiry.
    - t_steps: Number of time steps.
    - r0: Initial short rate.
    - strike: Strike rate for the swaption.
    - a: Mean reversion speed (Vasicek parameter).
    - b: Long-term mean level (Vasicek parameter).
    - sigma: Volatility (Vasicek parameter).
    - method: Finite difference method ('explicit', 'implicit', or 'crank-nicolson').
    - swap_maturity: Maturity of the underlying swap.
    - swap_frequency: Payment frequency of the swap.
    
    Returns:
    - Option price corresponding to initial rate r0.
    """
    
    # Grid setup
    dr = S_max / r_steps
    dt = T / t_steps
    
    # Create grid for r and t
    r_grid = np.linspace(0, S_max, r_steps + 1)
    t_grid = np.linspace(0, T, t_steps + 1)
    
    # Stability check for explicit method
    if method == 'explicit' and dt > dr**2 / (sigma**2 * S_max):
        print(f"Warning: Explicit method may be unstable. Consider reducing dt or increasing dr.")
        print(f"Current dt: {dt:.6f}, stability threshold: {dr**2 / (sigma**2 * S_max):.6f}")
    
    # Initialize solution grid
    V = np.zeros((r_steps + 1, t_steps + 1))
    
    # Terminal condition: payoff at maturity (underlying swap value)
    num_payments = int(swap_maturity / swap_frequency)
    for i, r in enumerate(r_grid):
        # Calculate the value of a fixed-for-floating swap at option expiry
        swap_value = 0
        
        # Value of receiving fixed and paying floating
        for j in range(1, num_payments + 1):
            payment_time = j * swap_frequency
            # Discount factor using Vasicek bond pricing formula
            B = (1 - np.exp(-a * payment_time)) / a
            A = np.exp((b - (sigma**2) / (2 * a**2)) * (B - payment_time) - 
                      (sigma**2) * (B**2) / (4 * a))
            discount = A * np.exp(-B * r)
            
            # Value of the payment
            swap_value += discount * swap_frequency * (strike - r)
        
        # Payoff of the swaption (call option on the swap value)
        V[i, -1] = max(swap_value, 0)
    
    # Select finite difference method
    if method == 'explicit':
        V = _explicit_fd_scheme(V, r_grid, t_grid, dr, dt, a, b, sigma)
    elif method == 'implicit':
        V = _implicit_fd_scheme(V, r_grid, t_grid, dr, dt, a, b, sigma)
    elif method == 'crank-nicolson':
        V = _crank_nicolson_fd_scheme(V, r_grid, t_grid, dr, dt, a, b, sigma)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'explicit', 'implicit', or 'crank-nicolson'.")
    
    # Interpolate to find the value at initial rate r0
    price = np.interp(r0, r_grid, V[:, 0])
    return price


def _explicit_fd_scheme(V, r_grid, t_grid, dr, dt, a, b, sigma):
    """
    Implement explicit finite difference scheme.
    """
    r_steps = len(r_grid) - 1
    t_steps = len(t_grid) - 1
    
    # Finite difference method (explicit scheme) backward in time
    for j in range(t_steps - 1, -1, -1):
        for i in range(1, r_steps):
            # Coefficients for the explicit scheme
            alpha = 0.5 * dt * (sigma**2 * r_grid[i] / dr**2)
            beta = 0.5 * dt * (a * (b - r_grid[i]) / dr)
            
            # Explicit update
            V[i, j] = V[i, j+1] * (1 - 2*alpha - r_grid[i]*dt) + \
                      V[i+1, j+1] * (alpha + beta) + \
                      V[i-1, j+1] * (alpha - beta)
        
        # Boundary conditions
        V[0, j] = V[1, j]  # Linear extrapolation at r=0
        V[-1, j] = 2*V[-2, j] - V[-3, j]  # Second-order extrapolation at r=S_max
    
    return V


def _implicit_fd_scheme(V, r_grid, t_grid, dr, dt, a, b, sigma):
    """
    Implement implicit finite difference scheme using sparse matrices.
    """
    r_steps = len(r_grid) - 1
    t_steps = len(t_grid) - 1
    
    # Initialize coefficient matrices
    alpha = np.zeros(r_steps + 1)
    beta = np.zeros(r_steps + 1)
    gamma = np.zeros(r_steps + 1)
    
    # Backward in time
    for j in range(t_steps - 1, -1, -1):
        # Set up the tridiagonal system
        for i in range(1, r_steps):
            alpha[i] = 0.5 * (sigma**2 * r_grid[i] / dr**2 - a * (b - r_grid[i]) / dr)
            beta[i] = -1/dt - sigma**2 * r_grid[i] / dr**2 - r_grid[i]
            gamma[i] = 0.5 * (sigma**2 * r_grid[i] / dr**2 + a * (b - r_grid[i]) / dr)
        
        # Boundary conditions
        alpha[0] = 0
        beta[0] = 1
        gamma[0] = -1  # V[0,j] = V[1,j]
        
        alpha[-1] = 2
        beta[-1] = -4
        gamma[-1] = 2  # V[r_steps,j] = 2*V[r_steps-1,j] - V[r_steps-2,j]
        
        # Create the tridiagonal matrix
        diagonals = [alpha[1:], beta, gamma[:-1]]
        positions = [-1, 0, 1]
        A = diags(diagonals, positions, shape=(r_steps+1, r_steps+1), format='csr')
        
        # Right-hand side
        rhs = -V[:, j+1] / dt
        
        # Apply boundary conditions to rhs
        rhs[0] = 0
        rhs[-1] = 0
        
        # Solve the system
        V[:, j] = spsolve(A, rhs)
    
    return V


def _crank_nicolson_fd_scheme(V, r_grid, t_grid, dr, dt, a, b, sigma):
    """
    Implement Crank-Nicolson finite difference scheme using sparse matrices.
    """
    r_steps = len(r_grid) - 1
    t_steps = len(t_grid) - 1
    
    # Initialize coefficient matrices for the implicit and explicit parts
    alpha_implicit = np.zeros(r_steps + 1)
    beta_implicit = np.zeros(r_steps + 1)
    gamma_implicit = np.zeros(r_steps + 1)
    
    alpha_explicit = np.zeros(r_steps + 1)
    beta_explicit = np.zeros(r_steps + 1)
    gamma_explicit = np.zeros(r_steps + 1)
    
    # Backward in time
    for j in range(t_steps - 1, -1, -1):
        # Set up coefficients
        for i in range(1, r_steps):
            # Diffusion term
            diff_coef = 0.25 * sigma**2 * r_grid[i] / dr**2
            # Drift term
            drift_coef = 0.25 * a * (b - r_grid[i]) / dr
            # Rate term
            rate_coef = 0.5 * r_grid[i]
            
            # Implicit part (left-hand side)
            alpha_implicit[i] = -diff_coef + drift_coef
            beta_implicit[i] = 1/dt + 2*diff_coef + rate_coef
            gamma_implicit[i] = -diff_coef - drift_coef
            
            # Explicit part (right-hand side)
            alpha_explicit[i] = diff_coef - drift_coef
            beta_explicit[i] = 1/dt - 2*diff_coef - rate_coef
            gamma_explicit[i] = diff_coef + drift_coef
        
        # Boundary conditions
        # At r=0
        alpha_implicit[0] = 0
        beta_implicit[0] = 1
        gamma_implicit[0] = -1
        
        alpha_explicit[0] = 0
        beta_explicit[0] = 0
        gamma_explicit[0] = 0
        
        # At r=S_max
        alpha_implicit[-1] = 1
        beta_implicit[-1] = -2
        gamma_implicit[-1] = 0
        
        alpha_explicit[-1] = 0
        beta_explicit[-1] = 0
        gamma_explicit[-1] = 0
        
        # Create the implicit matrix (LHS)
        diagonals_implicit = [alpha_implicit[1:], beta_implicit, gamma_implicit[:-1]]
        positions = [-1, 0, 1]
        A_implicit = diags(diagonals_implicit, positions, shape=(r_steps+1, r_steps+1), format='csr')
        
        # Create the explicit matrix (RHS)
        diagonals_explicit = [alpha_explicit[1:], beta_explicit, gamma_explicit[:-1]]
        A_explicit = diags(diagonals_explicit, positions, shape=(r_steps+1, r_steps+1), format='csr')
        
        # Right-hand side
        rhs = A_explicit @ V[:, j+1]
        
        # Apply boundary conditions
        rhs[0] = 0
        rhs[-1] = 0
        
        # Solve the system
        V[:, j] = spsolve(A_implicit, rhs)
    
    return V


def solve_bond_option(S_max, r_steps, T, t_steps, r0, strike, bond_maturity,
                     a=0.1, b=0.05, sigma=0.01, option_type='call', method='crank-nicolson'):
    """
    Solves PDE for a bond option (option on a zero-coupon bond) under the Vasicek model.
    
    Parameters:
    - S_max: Maximum short rate value (upper boundary).
    - r_steps: Number of grid steps for the r dimension.
    - T: Time to option expiry.
    - t_steps: Number of time steps.
    - r0: Initial short rate.
    - strike: Strike price for the bond option.
    - bond_maturity: Maturity of the underlying zero-coupon bond (from option expiry).
    - a, b, sigma: Vasicek model parameters.
    - option_type: 'call' or 'put' option.
    - method: Finite difference method.
    
    Returns:
    - Option price corresponding to initial rate r0.
    """
    # Grid setup
    dr = S_max / r_steps
    dt = T / t_steps
    
    # Create grid for r and t
    r_grid = np.linspace(0, S_max, r_steps + 1)
    
    # Initialize solution grid
    V = np.zeros((r_steps + 1, t_steps + 1))
    
    # Terminal condition: payoff at option expiry
    for i, r in enumerate(r_grid):
        # Bond price at option expiry
        B = (1 - np.exp(-a * bond_maturity)) / a
        A = np.exp((b - (sigma**2) / (2 * a**2)) * (B - bond_maturity) - 
                  (sigma**2) * (B**2) / (4 * a))
        bond_price = A * np.exp(-B * r)
        
        # Option payoff
        if option_type == 'call':
            V[i, -1] = max(bond_price - strike, 0)
        else:  # put
            V[i, -1] = max(strike - bond_price, 0)
    
    # Select finite difference method
    if method == 'explicit':
        V = _explicit_fd_scheme(V, r_grid, np.linspace(0, T, t_steps + 1), dr, dt, a, b, sigma)
    elif method == 'implicit':
        V = _implicit_fd_scheme(V, r_grid, np.linspace(0, T, t_steps + 1), dr, dt, a, b, sigma)
    elif method == 'crank-nicolson':
        V = _crank_nicolson_fd_scheme(V, r_grid, np.linspace(0, T, t_steps + 1), dr, dt, a, b, sigma)
    
    # Interpolate to find the value at initial rate r0
    price = np.interp(r0, r_grid, V[:, 0])
    return price


def plot_option_price_surface(solver_func, solver_args, r_range, title="Option Price Surface"):
    """
    Plot the option price as a function of interest rate and time to maturity.
    
    Parameters:
    - solver_func: Solver function to use.
    - solver_args: Dictionary of arguments for the solver.
    - r_range: Range of interest rates to plot.
    - title: Plot title.
    
    Returns:
    - Figure and axes objects.
    """
    # Extract time parameters
    T = solver_args['T']
    t_steps = solver_args['t_steps']
    dt = T / t_steps
    
    # Create grids
    r_values = np.linspace(r_range[0], r_range[1], 20)
    t_values = np.linspace(0, T, 10)
    R, T_mesh = np.meshgrid(r_values, t_values)
    
    # Initialize price surface
    Z = np.zeros_like(R)
    
    # Calculate option prices for each (r, t) pair
    for i, t in enumerate(t_values):
        for j, r in enumerate(r_values):
            # Update time to maturity
            updated_args = solver_args.copy()
            updated_args['T'] = T - t
            updated_args['t_steps'] = max(int((T - t) / dt), 1)
            updated_args['r0'] = r
            
            # Price the option
            if updated_args['T'] > 0:
                Z[i, j] = solver_func(**updated_args)
            else:
                # At maturity, use the payoff function
                if 'strike' in updated_args and 'bond_maturity' in updated_args:
                    # For bond options
                    a = updated_args.get('a', 0.1)
                    b = updated_args.get('b', 0.05)
                    sigma = updated_args.get('sigma', 0.01)
                    
                    B = (1 - np.exp(-a * updated_args['bond_maturity'])) / a
                    A = np.exp((b - (sigma**2) / (2 * a**2)) * (B - updated_args['bond_maturity']) - 
                              (sigma**2) * (B**2) / (4 * a))
                    bond_price = A * np.exp(-B * r)
                    
                    option_type = updated_args.get('option_type', 'call')
                    if option_type == 'call':
                        Z[i, j] = max(bond_price - updated_args['strike'], 0)
                    else:
                        Z[i, j] = max(updated_args['strike'] - bond_price, 0)
                else:
                    # For swaptions (simplified)
                    Z[i, j] = max(r - updated_args.get('strike', 0.05), 0)
    
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(R, T_mesh, Z, cmap='viridis', alpha=0.8)
    
    # Add labels and colorbar
    ax.set_xlabel('Interest Rate')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Price')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax


def compare_pricing_methods(option_type, r0, strike, maturities, methods=['explicit', 'implicit', 'crank-nicolson']):
    """
    Compare different finite difference methods for option pricing.
    
    Parameters:
    - option_type: 'swaption' or 'bond_option'.
    - r0: Initial interest rate.
    - strike: Strike price/rate.
    - maturities: List of option maturities to test.
    - methods: List of finite difference methods to compare.
    
    Returns:
    - Comparison DataFrame and plots.
    """
    import pandas as pd
    from time import time
    
    # Standard parameters
    S_max = 0.15
    r_steps = 100
    t_steps = 100
    a, b, sigma = 0.1, 0.05, 0.01
    
    # Initialize results
    results = []
    
    for T in maturities:
        for method in methods:
            start_time = time()
            
            if option_type == 'swaption':
                price = solve_european_swaption(
                    S_max=S_max, r_steps=r_steps, T=T, t_steps=t_steps,
                    r0=r0, strike=strike, a=a, b=b, sigma=sigma, method=method
                )
            elif option_type == 'bond_option':
                price = solve_bond_option(
                    S_max=S_max, r_steps=r_steps, T=T, t_steps=t_steps,
                    r0=r0, strike=strike, bond_maturity=5.0, a=a, b=b, sigma=sigma, method=method
                )
            else:
                raise ValueError(f"Unknown option type: {option_type}")
            
            elapsed = time() - start_time
            
            results.append({
                'Maturity': T,
                'Method': method,
                'Price': price,
                'Time (s)': elapsed
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot prices by maturity for each method
    for method in methods:
        method_df = df[df['Method'] == method]
        ax1.plot(method_df['Maturity'], method_df['Price'], marker='o', label=method)
    
    ax1.set_xlabel('Option Maturity (years)')
    ax1.set_ylabel('Option Price')
    ax1.set_title(f'{option_type.capitalize()} Prices by Method')
    ax1.grid(True)
    ax1.legend()
    
    # Plot computation time
    for method in methods:
        method_df = df[df['Method'] == method]
        ax2.plot(method_df['Maturity'], method_df['Time (s)'], marker='s', label=method)
    
    ax2.set_xlabel('Option Maturity (years)')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title('Performance Comparison')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    return df, fig, (ax1, ax2)