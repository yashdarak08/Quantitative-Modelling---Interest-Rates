"""
main.py
--------
Entry point for the Quantitative Modelling project.
This script runs simulations, prices a European swaption using a finite difference PDE solver,
performs Monte Carlo risk analysis, and runs statistical diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import modules from src
from src import models, pde_solver, monte_carlo, diagnostics, utils

def run_interest_rate_models():
    # Simulation parameters
    T = 1.0         # 1 year
    dt = 0.01
    n_steps = int(T/dt)
    n_paths = 1000  # For demonstration; increase for more precision
    
    # Vasicek model parameters
    vasicek_params = {'a': 0.1, 'b': 0.05, 'sigma': 0.01, 'r0': 0.03}
    vasicek_paths = models.VasicekModel(**vasicek_params).simulate(n_paths, n_steps, dt)
    
    # CIR model parameters
    cir_params = {'a': 0.15, 'b': 0.05, 'sigma': 0.02, 'r0': 0.03}
    cir_paths = models.CIRModel(**cir_params).simulate(n_paths, n_steps, dt)
    
    # Plot sample paths
    utils.plot_sample_paths(vasicek_paths, title="Vasicek Model Sample Paths")
    utils.plot_sample_paths(cir_paths, title="CIR Model Sample Paths")
    
    return vasicek_paths, cir_paths

def run_pde_solver():
    # Parameters for the PDE solver
    S_max = 0.15      # Maximum short rate
    r_steps = 100     # Grid steps in r dimension
    T = 1.0           # Maturity (in years)
    t_steps = 100     # Grid steps in time dimension
    
    # Option parameters for a European swaption
    strike = 0.04
    r0 = 0.03
    pde_price = pde_solver.solve_european_swaption(S_max, r_steps, T, t_steps, r0, strike)
    print(f"European Swaption Price from PDE Solver: {pde_price:.6f}")
    return pde_price

def run_monte_carlo(vasicek_paths):
    # Using the simulated paths from the Vasicek model for risk metrics
    # Compute portfolio returns or swaption price distribution from simulated rates
    # For illustration, we assume portfolio value depends linearly on short rates.
    portfolio_values = 100 * np.exp(-np.mean(vasicek_paths, axis=1))
    
    # Compute VaR and CVaR at 95% confidence level
    var, cvar = monte_carlo.compute_var_cvar(portfolio_values, confidence=0.95)
    print(f"VaR (95%): {var:.4f}")
    print(f"CVaR (95%): {cvar:.4f}")
    
    return portfolio_values

def run_diagnostics(portfolio_values):
    # Run statistical diagnostic tests on portfolio values
    jb_stat, jb_p = diagnostics.jarque_bera_test(portfolio_values)
    lb_stat, lb_p = diagnostics.ljung_box_test(portfolio_values)
    levene_stat, levene_p = diagnostics.levene_test(portfolio_values)
    
    print(f"Jarque-Bera test: statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
    print(f"Ljung-Box test: statistic = {lb_stat:.4f}, p-value = {lb_p:.4f}")
    print(f"Levene's test: statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")

if __name__ == "__main__":
    # Run interest rate model simulations
    vasicek_paths, cir_paths = run_interest_rate_models()
    
    # Solve PDE for European swaption pricing
    run_pde_solver()
    
    # Run Monte Carlo risk analysis using Vasicek paths
    portfolio_values = run_monte_carlo(vasicek_paths)
    
    # Run diagnostic tests on portfolio values
    run_diagnostics(portfolio_values)
    
    plt.show()