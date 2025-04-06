"""
monte_carlo.py
--------------
Monte Carlo simulation routines for risk metrics:
- Computes VaR and CVaR from a simulated portfolio distribution.
"""

import numpy as np

def compute_var_cvar(portfolio_values, confidence=0.95):
    """
    Compute Value at Risk (VaR) and Conditional VaR (CVaR) for the portfolio.
    
    Parameters:
    - portfolio_values: 1D numpy array of simulated portfolio values.
    - confidence: Confidence level (default 95%).
    
    Returns:
    - var: Value at Risk.
    - cvar: Conditional Value at Risk.
    """
    sorted_values = np.sort(portfolio_values)
    index = int((1 - confidence) * len(sorted_values))
    var = sorted_values[index]
    cvar = sorted_values[:index].mean() if index > 0 else var
    return var, cvar
