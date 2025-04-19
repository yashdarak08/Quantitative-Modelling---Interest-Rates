"""
monte_carlo.py
--------------
Monte Carlo simulation routines for interest rate derivatives pricing and risk analysis.
Includes:
- Portfolio valuation under multiple scenarios
- Risk metrics (VaR, CVaR, etc.)
- Stress testing framework
- Sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

def compute_var_cvar(portfolio_values, confidence=0.95):
    """
    Compute Value at Risk (VaR) and Conditional VaR (CVaR) for the portfolio.
    
    Parameters:
    - portfolio_values: 1D numpy array of simulated portfolio values.
    - confidence: Confidence level (default 95%).
    
    Returns:
    - var: Value at Risk (loss amount at the specified confidence level).
    - cvar: Conditional Value at Risk (expected loss beyond VaR).
    """
    # Compute profit/loss by comparing to initial value
    initial_value = np.mean(portfolio_values)
    pnl = portfolio_values - initial_value
    
    # Sort the P&L in ascending order
    sorted_pnl = np.sort(pnl)
    
    # Find the index at the confidence level
    index = int((1 - confidence) * len(sorted_pnl))
    
    # Calculate VaR (positive number indicates a loss)
    var = -sorted_pnl[index]
    
    # Calculate CVaR (expected loss beyond VaR)
    cvar = -sorted_pnl[:index].mean() if index > 0 else var
    
    return var, cvar


def compute_risk_metrics(portfolio_values, initial_value=None, confidence_levels=[0.95, 0.99]):
    """
    Compute comprehensive risk metrics for the portfolio.
    
    Parameters:
    - portfolio_values: 1D numpy array of simulated portfolio values.
    - initial_value: Initial portfolio value (if None, mean of portfolio_values is used).
    - confidence_levels: List of confidence levels for VaR/CVaR.
    
    Returns:
    - Dictionary of risk metrics.
    """
    if initial_value is None:
        initial_value = np.mean(portfolio_values)
    
    # Compute returns
    returns = portfolio_values / initial_value - 1
    
    # Initialize results dictionary
    metrics = {
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std_dev': np.std(returns),
        'skewness': pd.Series(returns).skew(),
        'kurtosis': pd.Series(returns).kurtosis(),
        'min': np.min(returns),
        'max': np.max(returns),
        'range': np.max(returns) - np.min(returns),
        'semi_variance': np.mean(np.maximum(0, -returns)**2),
        'downside_deviation': np.sqrt(np.mean(np.maximum(0, -returns)**2)),
    }
    
    # Compute VaR and CVaR at different confidence levels
    for cl in confidence_levels:
        var, cvar = compute_var_cvar(portfolio_values, confidence=cl)
        metrics[f'VaR_{int(cl*100)}'] = var / initial_value  # as a percentage
        metrics[f'CVaR_{int(cl*100)}'] = cvar / initial_value  # as a percentage
    
    # Compute additional risk metrics
    metrics['sharpe_ratio'] = metrics['mean'] / metrics['std_dev'] if metrics['std_dev'] > 0 else np.nan
    metrics['sortino_ratio'] = metrics['mean'] / metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else np.nan
    
    return metrics


def simulate_portfolio(rate_paths, bond_maturities, notionals, fixed_rates=None, is_receiver=None, valuation_func=None):
    """
    Simulate portfolio value across different interest rate scenarios.
    
    Parameters:
    - rate_paths: Array of simulated interest rate paths (n_paths x n_steps).
    - bond_maturities: List of bond maturities in the portfolio.
    - notionals: List of notional amounts for each bond.
    - fixed_rates: Fixed rates for swaps (if None, zero-coupon bonds are assumed).
    - is_receiver: Boolean array indicating if each swap is receiver (pay fixed) or payer (receive fixed).
    - valuation_func: Custom valuation function (if None, a simple duration-based approach is used).
    
    Returns:
    - portfolio_values: Array of portfolio values for each rate path.
    """
    n_paths = rate_paths.shape[0]
    n_instruments = len(bond_maturities)
    
    # Default parameters if not provided
    if fixed_rates is None:
        fixed_rates = [0.0] * n_instruments  # Zero-coupon bonds
        is_receiver = [True] * n_instruments  # Receiving at maturity
    elif is_receiver is None:
        is_receiver = [True] * n_instruments
    
    # Validate dimensions
    assert len(notionals) == n_instruments, "Mismatch between number of maturities and notionals"
    assert len(fixed_rates) == n_instruments, "Mismatch between number of maturities and fixed rates"
    assert len(is_receiver) == n_instruments, "Mismatch between number of instruments and is_receiver flags"
    
    # Initialize portfolio values
    portfolio_values = np.zeros(n_paths)
    
    # Terminal interest rates (use the last simulated rates)
    terminal_rates = rate_paths[:, -1]
    
    # Loop through each instrument in the portfolio
    for i in range(n_instruments):
        maturity = bond_maturities[i]
        notional = notionals[i]
        fixed_rate = fixed_rates[i]
        receiver = is_receiver[i]
        
        if valuation_func is None:
            # Simple valuation using duration approximation
            if fixed_rate == 0.0:  # Zero-coupon bond
                durations = np.array([maturity] * n_paths)
                discount_factors = np.exp(-terminal_rates * durations)
                instrument_values = notional * discount_factors
            else:  # Interest rate swap
                # Approximate swap valuation
                swap_rate = terminal_rates  # Simplified: use terminal rate as swap rate
                pv01 = maturity  # Simplified duration
                instrument_values = notional * pv01 * (fixed_rate - swap_rate)
                if not receiver:
                    instrument_values = -instrument_values
        else:
            # Use the provided custom valuation function
            instrument_values = valuation_func(terminal_rates, maturity, notional, fixed_rate, receiver)
        
        # Add to portfolio value
        portfolio_values += instrument_values
    
    return portfolio_values


def run_stress_test(model, scenarios, portfolio_config, n_paths=1000, n_steps=100, dt=0.01):
    """
    Run stress tests on a portfolio under different scenarios.
    
    Parameters:
    - model: Interest rate model class (e.g., VasicekModel).
    - scenarios: Dictionary of stress test scenarios with model parameters.
    - portfolio_config: Dictionary with portfolio configuration.
    - n_paths: Number of Monte Carlo paths.
    - n_steps: Number of time steps.
    - dt: Time step size.
    
    Returns:
    - results: DataFrame with stress test results.
    """
    # Extract portfolio configuration
    bond_maturities = portfolio_config.get('maturities', [1.0, 2.0, 5.0])
    notionals = portfolio_config.get('notionals', [1000, 1000, 1000])
    fixed_rates = portfolio_config.get('fixed_rates')
    is_receiver = portfolio_config.get('is_receiver')
    valuation_func = portfolio_config.get('valuation_func')
    
    # Initialize results
    results = []
    
    # Run simulations for each scenario
    for scenario_name, scenario_params in scenarios.items():
        # Create and configure the model
        model_instance = model(**scenario_params)
        
        # Simulate interest rate paths
        rate_paths = model_instance.simulate(n_paths, n_steps, dt)
        
        # Value the portfolio
        portfolio_values = simulate_portfolio(
            rate_paths, bond_maturities, notionals, fixed_rates, is_receiver, valuation_func
        )
        
        # Compute risk metrics
        risk_metrics = compute_risk_metrics(portfolio_values)
        
        # Add to results
        scenario_result = {'Scenario': scenario_name}
        scenario_result.update(risk_metrics)
        results.append(scenario_result)
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def sensitivity_analysis(model, base_params, param_ranges, portfolio_config, 
                       n_paths=1000, n_steps=100, dt=0.01):
    """
    Perform sensitivity analysis by varying model parameters.
    
    Parameters:
    - model: Interest rate model class.
    - base_params: Dictionary of base model parameters.
    - param_ranges: Dictionary mapping parameter names to lists of values to test.
    - portfolio_config: Dictionary with portfolio configuration.
    - n_paths, n_steps, dt: Simulation parameters.
    
    Returns:
    - results: DataFrame with sensitivity analysis results.
    """
    # Extract portfolio configuration
    bond_maturities = portfolio_config.get('maturities', [1.0, 2.0, 5.0])
    notionals = portfolio_config.get('notionals', [1000, 1000, 1000])
    fixed_rates = portfolio_config.get('fixed_rates')
    is_receiver = portfolio_config.get('is_receiver')
    valuation_func = portfolio_config.get('valuation_func')
    
    # Initialize results
    results = []
    
    # Loop through each parameter to analyze
    for param_name, param_values in param_ranges.items():
        for param_value in param_values:
            # Create parameters with the current value
            curr_params = base_params.copy()
            curr_params[param_name] = param_value
            
            # Create and configure the model
            model_instance = model(**curr_params)
            
            # Simulate interest rate paths
            rate_paths = model_instance.simulate(n_paths, n_steps, dt)
            
            # Value the portfolio
            portfolio_values = simulate_portfolio(
                rate_paths, bond_maturities, notionals, fixed_rates, is_receiver, valuation_func
            )
            
            # Compute key risk metrics
            var_95, cvar_95 = compute_var_cvar(portfolio_values, confidence=0.95)
            mean_value = np.mean(portfolio_values)
            std_dev = np.std(portfolio_values)
            
            # Add to results
            results.append({
                'Parameter': param_name,
                'Value': param_value,
                'Mean': mean_value,
                'StdDev': std_dev,
                'VaR_95': var_95,
                'CVaR_95': cvar_95
            })
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def plot_stress_test_results(stress_results, risk_metrics=None, figsize=(14, 10)):
    """
    Visualize the results of stress tests.
    
    Parameters:
    - stress_results: DataFrame with stress test results.
    - risk_metrics: List of risk metrics to plot (if None, defaults are used).
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    if risk_metrics is None:
        risk_metrics = ['mean', 'std_dev', 'VaR_95', 'CVaR_95']
    
    # Create the figure
    fig, axes = plt.subplots(len(risk_metrics), 1, figsize=figsize)
    
    # If only one metric, ensure axes is a list
    if len(risk_metrics) == 1:
        axes = [axes]
    
    # Plot each risk metric
    for i, metric in enumerate(risk_metrics):
        if metric in stress_results.columns:
            # Sort scenarios by the current metric
            sorted_results = stress_results.sort_values(by=metric)
            
            # Plot as a bar chart
            sorted_results.plot(
                x='Scenario', y=metric, kind='bar', ax=axes[i],
                color='skyblue', edgecolor='black', alpha=0.7
            )
            
            # Add labels and title
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Scenario')
            axes[i].set_ylabel(metric)
            
            # Rotate x-axis labels for better visibility
            for tick in axes[i].get_xticklabels():
                tick.set_rotation(45)
            
            # Add grid
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(sensitivity_results, figsize=(14, 10)):
    """
    Visualize the results of sensitivity analysis.
    
    Parameters:
    - sensitivity_results: DataFrame with sensitivity analysis results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Get unique parameters
    parameters = sensitivity_results['Parameter'].unique()
    
    # Create subplots for each parameter
    fig, axes = plt.subplots(len(parameters), 1, figsize=figsize)
    
    # If only one parameter, ensure axes is a list
    if len(parameters) == 1:
        axes = [axes]
    
    # Metrics to plot
    metrics = ['Mean', 'VaR_95', 'CVaR_95']
    colors = ['green', 'blue', 'red']
    
    # Plot each parameter
    for i, param in enumerate(parameters):
        param_data = sensitivity_results[sensitivity_results['Parameter'] == param]
        
        # Create a secondary axis for VaR and CVaR
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # Plot mean on primary axis
        param_data.plot(
            x='Value', y='Mean', kind='line', marker='o', color=colors[0],
            ax=ax1, label='Mean Value', legend=False
        )
        
        # Plot VaR and CVaR on secondary axis
        for j, metric in enumerate(metrics[1:], 1):
            param_data.plot(
                x='Value', y=metric, kind='line', marker='s', color=colors[j],
                ax=ax2, label=metric, legend=False
            )
        
        # Add labels and title
        ax1.set_title(f'Sensitivity to {param}')
        ax1.set_xlabel(param)
        ax1.set_ylabel('Mean Portfolio Value', color=colors[0])
        ax2.set_ylabel('Risk Metrics', color='black')
        
        # Adjust tick colors
        ax1.tick_params(axis='y', labelcolor=colors[0])
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    return fig


def monte_carlo_option_pricing(model, option_type, strike, maturity, r0, 
                             a=0.1, b=0.05, sigma=0.01, n_paths=10000, n_steps=100, dt=0.01):
    """
    Price options using Monte Carlo simulation.
    
    Parameters:
    - model: Interest rate model class.
    - option_type: 'call', 'put', 'swaption_receiver', or 'swaption_payer'.
    - strike: Strike price/rate.
    - maturity: Option maturity.
    - r0: Initial interest rate.
    - a, b, sigma: Model parameters.
    - n_paths, n_steps, dt: Simulation parameters.
    
    Returns:
    - price: Option price.
    - std_error: Standard error of the price.
    - confidence_interval: 95% confidence interval.
    """
    # Create and configure the model
    model_instance = model(a=a, b=b, sigma=sigma, r0=r0)
    
    # Simulate interest rate paths
    rate_paths = model_instance.simulate(n_paths, n_steps, dt)
    
    # Extract rates at maturity
    maturity_rates = rate_paths[:, -1]
    
    # Calculate payoffs based on option type
    if option_type == 'call':
        # Call option on zero-coupon bond
        payoffs = np.maximum(maturity_rates - strike, 0)
    elif option_type == 'put':
        # Put option on zero-coupon bond
        payoffs = np.maximum(strike - maturity_rates, 0)
    elif option_type == 'swaption_receiver':
        # Receiver swaption (right to receive fixed, pay floating)
        swap_rate = strike
        swap_maturity = 5.0  # Simplified: 5-year swap
        
        # Simplified valuation using PV01 approximation
        pv01 = swap_maturity
        payoffs = np.maximum(pv01 * (swap_rate - maturity_rates), 0)
    elif option_type == 'swaption_payer':
        # Payer swaption (right to pay fixed, receive floating)
        swap_rate = strike
        swap_maturity = 5.0  # Simplified: 5-year swap
        
        # Simplified valuation using PV01 approximation
        pv01 = swap_maturity
        payoffs = np.maximum(pv01 * (maturity_rates - swap_rate), 0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")
    
    # Discount payoffs to present value
    discount_factors = np.exp(-r0 * maturity)  # Simplified discounting
    present_values = payoffs * discount_factors
    
    # Calculate price and error metrics
    price = np.mean(present_values)
    std_dev = np.std(present_values)
    std_error = std_dev / np.sqrt(n_paths)
    
    # 95% confidence interval
    confidence_interval = (
        price - 1.96 * std_error,
        price + 1.96 * std_error
    )
    
    return price, std_error, confidence_interval