"""
stress_testing.py
----------------
Framework for stress testing interest rate models and portfolios.
Includes:
- Scenario generation
- Portfolio impact analysis
- Tail risk evaluation
- Sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def define_stress_scenarios():
    """
    Define standard stress scenarios for interest rate models.
    
    Returns:
    - scenarios: Dictionary of stress scenarios.
    """
    scenarios = {
        "Baseline": {},  # No adjustment to parameters
        
        "High_Volatility": {
            "sigma": lambda x: x * 2.0  # Double the volatility
        },
        
        "Low_Volatility": {
            "sigma": lambda x: x * 0.5  # Half the volatility
        },
        
        "Fast_Mean_Reversion": {
            "a": lambda x: x * 2.0  # Double the mean reversion speed
        },
        
        "Slow_Mean_Reversion": {
            "a": lambda x: x * 0.5  # Half the mean reversion speed
        },
        
        "High_Long_Term_Mean": {
            "b": lambda x: x * 1.5  # Increase long-term mean by 50%
        },
        
        "Low_Long_Term_Mean": {
            "b": lambda x: x * 0.75  # Decrease long-term mean by 25%
        },
        
        "Rate_Shock_Up": {
            "r0": lambda x: x + 0.02  # Add 200 bps to initial rate
        },
        
        "Rate_Shock_Down": {
            "r0": lambda x: max(x - 0.02, 0.0001)  # Subtract 200 bps, but keep positive
        },
        
        "Stagflation": {
            "b": lambda x: x * 1.5,  # High long-term mean
            "sigma": lambda x: x * 1.5  # High volatility
        },
        
        "Extreme_Volatility": {
            "sigma": lambda x: x * 3.0  # Triple the volatility
        }
    }
    
    return scenarios


def apply_scenario(model, scenario):
    """
    Apply a stress scenario to a model by adjusting its parameters.
    
    Parameters:
    - model: Interest rate model instance.
    - scenario: Dictionary with parameter adjustments.
    
    Returns:
    - adjusted_model: Model with adjusted parameters.
    """
    # Store original parameters
    original_params = {}
    for param_name in scenario.keys():
        if hasattr(model, param_name):
            original_params[param_name] = getattr(model, param_name)
    
    # Create a copy of the model (assuming the model has a copy method)
    if hasattr(model, 'copy'):
        adjusted_model = model.copy()
    else:
        # If no copy method, create a new instance with the same parameters
        # This is a simplified approach and may need to be adapted for specific models
        model_class = model.__class__
        adjusted_model = model_class(**{
            k: getattr(model, k) for k in ['a', 'b', 'sigma', 'r0'] 
            if hasattr(model, k)
        })
    
    # Apply parameter adjustments
    for param_name, adjustment_func in scenario.items():
        if hasattr(adjusted_model, param_name):
            current_value = getattr(adjusted_model, param_name)
            new_value = adjustment_func(current_value)
            setattr(adjusted_model, param_name, new_value)
    
    # Store the scenario information on the model
    adjusted_model.scenario_name = next(
        (name for name, s in define_stress_scenarios().items() if s == scenario),
        "Custom"
    )
    adjusted_model.original_params = original_params
    
    return adjusted_model


def run_scenario_simulation(model, scenario, n_paths=1000, n_steps=252, dt=1/252):
    """
    Run a simulation under a specific stress scenario.
    
    Parameters:
    - model: Interest rate model instance.
    - scenario: Dictionary with parameter adjustments.
    - n_paths: Number of paths to simulate.
    - n_steps: Number of time steps.
    - dt: Time step size.
    
    Returns:
    - result: Dictionary with scenario simulation results.
    """
    # Apply scenario adjustments to model
    adjusted_model = apply_scenario(model, scenario)
    
    # Run simulation
    paths = adjusted_model.simulate(n_paths, n_steps, dt)
    
    # Calculate summary statistics
    terminal_rates = paths[:, -1]
    summary_stats = {
        'mean': np.mean(terminal_rates),
        'median': np.median(terminal_rates),
        'std': np.std(terminal_rates),
        'min': np.min(terminal_rates),
        'max': np.max(terminal_rates),
        'q05': np.percentile(terminal_rates, 5),
        'q95': np.percentile(terminal_rates, 95)
    }
    
    # Compile results
    result = {
        'scenario_name': adjusted_model.scenario_name,
        'adjusted_params': {
            k: getattr(adjusted_model, k) for k in adjusted_model.original_params.keys()
        },
        'original_params': adjusted_model.original_params,
        'paths': paths,
        'terminal_rates': terminal_rates,
        'summary_stats': summary_stats
    }
    
    return result


def simulate_all_scenarios(model, custom_scenarios=None, n_paths=1000, n_steps=252, dt=1/252):
    """
    Run simulations for all standard stress scenarios and any custom scenarios.
    
    Parameters:
    - model: Interest rate model instance.
    - custom_scenarios: Dictionary of custom scenarios to include.
    - n_paths: Number of paths to simulate.
    - n_steps: Number of time steps.
    - dt: Time step size.
    
    Returns:
    - results: Dictionary of scenario simulation results.
    """
    # Get standard scenarios
    all_scenarios = define_stress_scenarios()
    
    # Add any custom scenarios
    if custom_scenarios:
        all_scenarios.update(custom_scenarios)
    
    # Run simulations for each scenario
    results = {}
    
    for scenario_name, scenario in all_scenarios.items():
        print(f"Simulating scenario: {scenario_name}")
        
        # Run simulation
        result = run_scenario_simulation(model, scenario, n_paths, n_steps, dt)
        
        # Store result
        results[scenario_name] = result
    
    return results


def calculate_portfolio_impact(scenario_results, portfolio_valuator):
    """
    Calculate the impact of stress scenarios on a portfolio.
    
    Parameters:
    - scenario_results: Dictionary of scenario simulation results.
    - portfolio_valuator: Function that takes rate paths and returns portfolio values.
    
    Returns:
    - portfolio_results: Dictionary of portfolio valuation results for each scenario.
    """
    # Initialize results
    portfolio_results = {}
    
    # Calculate portfolio values for each scenario
    for scenario_name, result in scenario_results.items():
        # Get rate paths
        paths = result['paths']
        
        # Value the portfolio
        portfolio_values = portfolio_valuator(paths)
        
        # Calculate summary statistics
        summary_stats = {
            'mean': np.mean(portfolio_values),
            'median': np.median(portfolio_values),
            'std': np.std(portfolio_values),
            'min': np.min(portfolio_values),
            'max': np.max(portfolio_values),
            'q05': np.percentile(portfolio_values, 5),
            'q95': np.percentile(portfolio_values, 95)
        }
        
        # Calculate risk metrics
        var_95 = np.mean(portfolio_values) - np.percentile(portfolio_values, 5)
        cvar_95 = np.mean(portfolio_values[portfolio_values <= np.percentile(portfolio_values, 5)])
        
        risk_metrics = {
            'VaR_95': var_95,
            'CVaR_95': cvar_95
        }
        
        # Store results
        portfolio_results[scenario_name] = {
            'values': portfolio_values,
            'summary_stats': summary_stats,
            'risk_metrics': risk_metrics
        }
    
    return portfolio_results


def plot_scenario_paths(scenario_results, num_paths=5, figsize=(15, 10)):
    """
    Plot sample paths for each scenario.
    
    Parameters:
    - scenario_results: Dictionary of scenario simulation results.
    - num_paths: Number of sample paths to plot per scenario.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Get scenario names
    scenario_names = list(scenario_results.keys())
    
    # Calculate grid dimensions
    n_scenarios = len(scenario_names)
    n_cols = 2
    n_rows = (n_scenarios + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each scenario
    for i, scenario_name in enumerate(scenario_names):
        ax = axes[i]
        result = scenario_results[scenario_name]
        paths = result['paths']
        
        # Get time axis
        n_steps = paths.shape[1]
        time_axis = np.linspace(0, n_steps-1, n_steps)
        
        # Plot sample paths
        for j in range(min(num_paths, paths.shape[0])):
            ax.plot(time_axis, paths[j], alpha=0.5, linewidth=1)
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        ax.plot(time_axis, mean_path, 'r-', linewidth=2, label='Mean')
        
        # Add confidence intervals
        std_path = np.std(paths, axis=0)
        ax.fill_between(
            time_axis,
            mean_path - 1.96 * std_path,
            mean_path + 1.96 * std_path,
            color='r', alpha=0.2, label='95% CI'
        )
        
        # Add scenario details
        adjusted_params = result['adjusted_params']
        param_text = ', '.join([f'{k}={v:.4f}' for k, v in adjusted_params.items()])
        
        ax.set_title(f'Scenario: {scenario_name}\n{param_text}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Interest Rate')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to first subplot to save space
        if i == 0:
            ax.legend()
    
    # Hide any unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_terminal_distributions(scenario_results, figsize=(12, 8)):
    """
    Plot the distribution of terminal interest rates for each scenario.
    
    Parameters:
    - scenario_results: Dictionary of scenario simulation results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot KDE for each scenario
    for scenario_name, result in scenario_results.items():
        terminal_rates = result['terminal_rates']
        
        # Plot KDE
        sns.kdeplot(terminal_rates, label=scenario_name, ax=ax)
    
    # Add labels and title
    ax.set_title('Terminal Rate Distributions by Scenario')
    ax.set_xlabel('Interest Rate')
    ax.set_ylabel('Density')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig


def plot_portfolio_impact(portfolio_results, figsize=(14, 10)):
    """
    Plot the impact of stress scenarios on portfolio values.
    
    Parameters:
    - portfolio_results: Dictionary of portfolio valuation results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get scenario names and statistics
    scenarios = list(portfolio_results.keys())
    means = [portfolio_results[s]['summary_stats']['mean'] for s in scenarios]
    stds = [portfolio_results[s]['summary_stats']['std'] for s in scenarios]
    var_95 = [portfolio_results[s]['risk_metrics']['VaR_95'] for s in scenarios]
    cvar_95 = [portfolio_results[s]['risk_metrics']['CVaR_95'] for s in scenarios]
    
    # Sort scenarios by mean portfolio value
    sorted_indices = np.argsort(means)
    sorted_scenarios = [scenarios[i] for i in sorted_indices]
    sorted_means = [means[i] for i in sorted_indices]
    sorted_stds = [stds[i] for i in sorted_indices]
    sorted_var = [var_95[i] for i in sorted_indices]
    sorted_cvar = [cvar_95[i] for i in sorted_indices]
    
    # 1. Plot mean portfolio values with error bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_scenarios)))
    
    y_pos = np.arange(len(sorted_scenarios))
    bars = ax1.barh(y_pos, sorted_means, color=colors, alpha=0.7)
    
    # Add error bars
    for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
        ax1.plot([mean - std, mean + std], [i, i], 'k-', alpha=0.7)
        ax1.plot([mean - std, mean - std], [i-0.2, i+0.2], 'k-', alpha=0.7)
        ax1.plot([mean + std, mean + std], [i-0.2, i+0.2], 'k-', alpha=0.7)
    
    # Add mean values on bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f'{sorted_means[i]:.2f}', va='center')
    
    ax1.set_title('Mean Portfolio Value by Scenario')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_scenarios)
    ax1.set_xlabel('Portfolio Value')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # 2. Plot risk metrics (VaR and CVaR)
    x_pos = np.arange(len(sorted_scenarios))
    width = 0.35
    
    ax2.barh(x_pos - width/2, sorted_var, width, label='VaR (95%)', color='skyblue', alpha=0.7)
    ax2.barh(x_pos + width/2, sorted_cvar, width, label='CVaR (95%)', color='tomato', alpha=0.7)
    
    # Add risk values
    for i, (var, cvar) in enumerate(zip(sorted_var, sorted_cvar)):
        ax2.text(var * 1.01, i - width/2, f'{var:.2f}', va='center')
        ax2.text(cvar * 1.01, i + width/2, f'{cvar:.2f}', va='center')
    
    ax2.set_title('Risk Metrics by Scenario')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(sorted_scenarios)
    ax2.set_xlabel('Value at Risk (95%)')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def tail_risk_analysis(portfolio_results, confidence_levels=[0.99, 0.995, 0.999]):
    """
    Perform tail risk analysis on portfolio values.
    
    Parameters:
    - portfolio_results: Dictionary of portfolio valuation results.
    - confidence_levels: List of confidence levels for VaR and CVaR.
    
    Returns:
    - tail_risk: DataFrame with tail risk metrics.
    """
    # Initialize results
    tail_risk_data = []
    
    # Calculate tail risk metrics for each scenario
    for scenario_name, result in portfolio_results.items():
        portfolio_values = result['values']
        
        # Calculate risk metrics at each confidence level
        for cl in confidence_levels:
            percentile = (1 - cl) * 100
            var = np.mean(portfolio_values) - np.percentile(portfolio_values, percentile)
            
            # Find values below the threshold for CVaR
            threshold = np.percentile(portfolio_values, percentile)
            tail_values = portfolio_values[portfolio_values <= threshold]
            
            # Calculate CVaR (Expected Shortfall)
            cvar = np.mean(tail_values) if len(tail_values) > 0 else var
            
            # Calculate expected loss as percentage of mean value
            expected_loss_pct = (np.mean(portfolio_values) - cvar) / np.mean(portfolio_values) * 100
            
            # Store results
            tail_risk_data.append({
                'Scenario': scenario_name,
                'Confidence Level': cl,
                'VaR': var,
                'CVaR': cvar,
                'Expected Loss (%)': expected_loss_pct,
                'Tail Size': len(tail_values) / len(portfolio_values) * 100  # Percentage of values in tail
            })
    
    # Convert to DataFrame
    tail_risk_df = pd.DataFrame(tail_risk_data)
    
    return tail_risk_df


def plot_tail_risk_comparison(tail_risk_df, figsize=(14, 10)):
    """
    Plot tail risk comparison across scenarios.
    
    Parameters:
    - tail_risk_df: DataFrame with tail risk metrics.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Get unique confidence levels and scenarios
    confidence_levels = sorted(tail_risk_df['Confidence Level'].unique())
    scenarios = sorted(tail_risk_df['Scenario'].unique())
    
    # Create a color map for confidence levels
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(confidence_levels)))
    
    # 1. Plot CVaR by scenario and confidence level
    ax = axes[0]
    
    # Group by scenario and confidence level
    grouped = tail_risk_df.groupby(['Scenario', 'Confidence Level'])['CVaR'].first().unstack()
    
    # Plot grouped bar chart
    grouped.plot(kind='bar', ax=ax, color=colors, width=0.8, alpha=0.7)
    
    ax.set_title('Conditional Value at Risk (CVaR) by Scenario')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('CVaR')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Plot expected loss percentage
    ax = axes[1]
    
    # Group by scenario and confidence level
    grouped = tail_risk_df.groupby(['Scenario', 'Confidence Level'])['Expected Loss (%)'].first().unstack()
    
    # Plot grouped bar chart
    grouped.plot(kind='bar', ax=ax, color=colors, width=0.8, alpha=0.7)
    
    ax.set_title('Expected Loss (% of Mean Portfolio Value) by Scenario')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Expected Loss (%)')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def create_stress_testing_report(model, scenario_results, portfolio_results, tail_risk_df):
    """
    Create a comprehensive stress testing report.
    
    Parameters:
    - model: Interest rate model used for simulations.
    - scenario_results: Dictionary of scenario simulation results.
    - portfolio_results: Dictionary of portfolio valuation results.
    - tail_risk_df: DataFrame with tail risk metrics.
    
    Returns:
    - report: Dictionary with report components.
    """
    # Create summary tables
    scenario_summary = []
    for scenario_name, result in scenario_results.items():
        # Extract key statistics
        stats = result['summary_stats']
        params = result['adjusted_params']
        
        # Create summary row
        row = {
            'Scenario': scenario_name,
            'Mean Rate': stats['mean'],
            'Std Dev': stats['std'],
            '5th Percentile': stats['q05'],
            '95th Percentile': stats['q95'],
        }
        
        # Add parameter values
        for param_name, value in params.items():
            row[f'Param_{param_name}'] = value
        
        scenario_summary.append(row)
    
    # Convert to DataFrame
    scenario_summary_df = pd.DataFrame(scenario_summary)
    
    # Portfolio summary
    portfolio_summary = []
    for scenario_name, result in portfolio_results.items():
        # Extract key statistics
        stats = result['summary_stats']
        risk = result['risk_metrics']
        
        # Create summary row
        portfolio_summary.append({
            'Scenario': scenario_name,
            'Mean Value': stats['mean'],
            'Std Dev': stats['std'],
            '5th Percentile': stats['q05'],
            '95th Percentile': stats['q95'],
            'VaR (95%)': risk['VaR_95'],
            'CVaR (95%)': risk['CVaR_95']
        })
    
    # Convert to DataFrame
    portfolio_summary_df = pd.DataFrame(portfolio_summary)
    
    # Create most severe scenarios ranking
    most_severe = portfolio_summary_df.sort_values('Mean Value').head(3)
    highest_risk = portfolio_summary_df.sort_values('CVaR (95%)', ascending=False).head(3)
    
    # Compile report
    report = {
        'model_info': {
            'model_type': model.__class__.__name__,
            'parameters': {
                k: getattr(model, k) for k in ['a', 'b', 'sigma', 'r0'] 
                if hasattr(model, k)
            }
        },
        'scenario_summary': scenario_summary_df,
        'portfolio_summary': portfolio_summary_df,
        'tail_risk': tail_risk_df,
        'most_severe_scenarios': most_severe,
        'highest_risk_scenarios': highest_risk
    }
    
    return report


def sensitivity_analysis(model, parameter_ranges, n_paths=1000, n_steps=252, dt=1/252):
    """
    Perform sensitivity analysis on model parameters.
    
    Parameters:
    - model: Interest rate model instance.
    - parameter_ranges: Dictionary mapping parameter names to lists of values.
    - n_paths: Number of paths to simulate.
    - n_steps: Number of time steps.
    - dt: Time step size.
    
    Returns:
    - sensitivity_results: Dictionary with sensitivity analysis results.
    """
    # Store original parameters
    original_params = {}
    for param_name in parameter_ranges.keys():
        if hasattr(model, param_name):
            original_params[param_name] = getattr(model, param_name)
    
    # Initialize results
    sensitivity_results = {
        'parameters': parameter_ranges,
        'original_params': original_params,
        'simulations': {}
    }
    
    # Run simulations for each parameter value
    for param_name, param_values in parameter_ranges.items():
        print(f"Analyzing sensitivity to parameter: {param_name}")
        
        # Initialize results for this parameter
        sensitivity_results['simulations'][param_name] = {}
        
        for param_value in param_values:
            # Set parameter value
            setattr(model, param_name, param_value)
            
            # Run simulation
            paths = model.simulate(n_paths, n_steps, dt)
            
            # Calculate statistics
            terminal_rates = paths[:, -1]
            summary_stats = {
                'mean': np.mean(terminal_rates),
                'median': np.median(terminal_rates),
                'std': np.std(terminal_rates),
                'min': np.min(terminal_rates),
                'max': np.max(terminal_rates),
                'q05': np.percentile(terminal_rates, 5),
                'q95': np.percentile(terminal_rates, 95)
            }
            
            # Store results
            sensitivity_results['simulations'][param_name][param_value] = {
                'paths': paths,
                'terminal_rates': terminal_rates,
                'summary_stats': summary_stats
            }
    
    # Restore original parameters
    for param_name, param_value in original_params.items():
        setattr(model, param_name, param_value)
    
    return sensitivity_results


def plot_sensitivity_results(sensitivity_results, figsize=(15, 10)):
    """
    Plot the results of sensitivity analysis.
    
    Parameters:
    - sensitivity_results: Dictionary with sensitivity analysis results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Get parameters analyzed
    parameters = list(sensitivity_results['simulations'].keys())
    
    # Calculate grid dimensions
    n_params = len(parameters)
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing (if needed)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot each parameter's sensitivity
    for i, param_name in enumerate(parameters):
        # Get appropriate axis based on grid dimensions
        if n_rows > 1 and n_cols > 1:
            ax = axes[i // n_cols, i % n_cols]
        else:
            ax = axes[i]
        
        # Get parameter values and results
        param_results = sensitivity_results['simulations'][param_name]
        param_values = sorted(param_results.keys())
        
        # Extract statistics for each parameter value
        means = [param_results[v]['summary_stats']['mean'] for v in param_values]
        q05 = [param_results[v]['summary_stats']['q05'] for v in param_values]
        q95 = [param_results[v]['summary_stats']['q95'] for v in param_values]
        
        # Plot mean values
        ax.plot(param_values, means, 'b-', marker='o', linewidth=2, label='Mean')
        
        # Plot confidence interval
        ax.fill_between(param_values, q05, q95, color='b', alpha=0.2, label='5%-95% Range')
        
        # Add labels and title
        ax.set_title(f'Sensitivity to {param_name}')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Terminal Interest Rate')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Hide any unused subplots
    for i in range(n_params, n_rows * n_cols):
        if n_rows > 1 and n_cols > 1:
            axes[i // n_cols, i % n_cols].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig