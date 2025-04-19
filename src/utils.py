"""
utils.py
--------
Utility functions for visualization, data processing, and general helper routines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import norm

def plot_sample_paths(paths, num_paths=10, title="Sample Paths", figsize=(10, 6)):
    """
    Plot a subset of simulated paths with enhanced visualization.
    
    Parameters:
    - paths: 2D numpy array (n_paths x n_steps+1).
    - num_paths: Number of paths to plot.
    - title: Plot title.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    n_paths, n_steps = paths.shape
    indices = np.linspace(0, n_paths - 1, num_paths, dtype=int)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap for the paths
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=num_paths-1)
    
    # Plot each path with a different color from the colormap
    time_axis = np.linspace(0, 1, n_steps)  # Assuming time from 0 to 1
    
    for i, idx in enumerate(indices):
        color = cmap(norm(i))
        ax.plot(time_axis, paths[idx, :], lw=1.5, alpha=0.8, color=color)
    
    # Add mean and confidence intervals
    mean_path = np.mean(paths, axis=0)
    std_path = np.std(paths, axis=0)
    
    ax.plot(time_axis, mean_path, 'r-', lw=2.5, label='Mean')
    ax.fill_between(time_axis, mean_path - 1.96 * std_path, mean_path + 1.96 * std_path, 
                    color='r', alpha=0.2, label='95% CI')
    
    # Add additional details
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Interest rate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add path distribution at the final time step
    divider = ax.inset_axes([1.05, 0, 0.25, 1], transform=ax.transAxes)
    final_rates = paths[:, -1]
    sns.kdeplot(y=final_rates, ax=divider, fill=True, color='steelblue', alpha=0.6)
    divider.set_ylabel('')
    divider.set_xlabel('Density')
    divider.set_title('Final Distribution')
    
    # Adjust final y-axis limits to match the main plot
    divider.set_ylim(ax.get_ylim())
    divider.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_yield_curve_3d(rate_paths, maturities, time_points=10, figsize=(14, 10)):
    """
    Create a 3D visualization of yield curve evolution over time.
    
    Parameters:
    - rate_paths: 2D numpy array of rate paths (n_paths x n_steps).
    - maturities: Array of bond maturities.
    - time_points: Number of time points to visualize.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    n_paths, n_steps = rate_paths.shape
    n_maturities = len(maturities)
    
    # Select time points to visualize
    time_indices = np.linspace(0, n_steps-1, time_points, dtype=int)
    times = np.linspace(0, 1, time_points)  # Assuming time from 0 to 1
    
    # Create a meshgrid for 3D plotting
    T, M = np.meshgrid(times, maturities)
    
    # Initialize yield surface
    Z = np.zeros((n_maturities, time_points))
    
    # Compute average rates at each time and convert to yields for each maturity
    for i, t_idx in enumerate(time_indices):
        r = np.mean(rate_paths[:, t_idx])
        
        # Simplified yield calculation (using constant short rate approximation)
        # In a real implementation, this would use a proper term structure model
        for j, mat in enumerate(maturities):
            # Simplified: y(t,T) ≈ r * (1 - exp(-a*mat))/(a*mat) + long_term_rate
            long_term_rate = 0.04  # Example long-term rate
            a = 0.1  # Example mean reversion speed
            
            adjustment = (1 - np.exp(-a * mat)) / (a * mat) if a * mat > 1e-6 else 1.0
            Z[j, i] = r * adjustment + long_term_rate * (1 - adjustment)
    
    # Create the 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the yield surface
    surf = ax.plot_surface(M, T, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Yield')
    
    # Add labels
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Time')
    ax.set_zlabel('Yield')
    ax.set_title('Yield Curve Evolution')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig


def plot_rate_distribution(paths, time_indices=None, figsize=(12, 8)):
    """
    Plot the distribution of interest rates at different time points.
    
    Parameters:
    - paths: 2D numpy array of rate paths (n_paths x n_steps).
    - time_indices: List of time indices to visualize (if None, evenly spaced points are used).
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    n_paths, n_steps = paths.shape
    
    # Select time points if not provided
    if time_indices is None:
        time_indices = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    
    # Create the figure
    fig, axes = plt.subplots(len(time_indices), 1, figsize=figsize)
    
    # If only one time index, ensure axes is a list
    if len(time_indices) == 1:
        axes = [axes]
    
    # Plot distribution at each time point
    for i, t_idx in enumerate(time_indices):
        # Extract rates at this time
        rates = paths[:, t_idx]
        
        # Normalize time to [0, 1]
        time = t_idx / (n_steps - 1)
        
        # Plot histogram and KDE
        sns.histplot(rates, kde=True, ax=axes[i], bins=30, alpha=0.6, color='steelblue')
        
        # Add normal distribution fit
        mu, std = norm.fit(rates)
        x = np.linspace(rates.min(), rates.max(), 100)
        p = norm.pdf(x, mu, std)
        axes[i].plot(x, p * n_paths * (rates.max() - rates.min()) / 30, 'r-', linewidth=2,
                     label=f'Normal: μ={mu:.4f}, σ={std:.4f}')
        
        # Add annotations
        axes[i].set_title(f'Rate Distribution at t={time:.2f}')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend()
        
        # Add statistics as text
        stats_text = (f'Mean: {np.mean(rates):.4f}\n'
                     f'Median: {np.median(rates):.4f}\n'
                     f'Std Dev: {np.std(rates):.4f}\n'
                     f'Min: {np.min(rates):.4f}\n'
                     f'Max: {np.max(rates):.4f}')
        
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_sensitivity_heatmap(results, x_param, y_param, metric, figsize=(12, 10)):
    """
    Create a heatmap visualizing the sensitivity of a metric to two parameters.
    
    Parameters:
    - results: DataFrame with sensitivity analysis results.
    - x_param: Parameter name for the x-axis.
    - y_param: Parameter name for the y-axis.
    - metric: Metric to visualize.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Filter the results for the two parameters
    filtered_results = results[
        (results['Parameter'] == x_param) | 
        (results['Parameter'] == y_param)
    ].copy()
    
    # Get unique values for each parameter
    x_values = sorted(filtered_results[filtered_results['Parameter'] == x_param]['Value'].unique())
    y_values = sorted(filtered_results[filtered_results['Parameter'] == y_param]['Value'].unique())
    
    # Create a grid for the heatmap
    heatmap_data = np.zeros((len(y_values), len(x_values)))
    
    # Fill the grid with metric values
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            # Create a temporary dataframe with results for each parameter combination
            temp_df = pd.DataFrame()
            
            # Get rows for x parameter
            x_rows = filtered_results[
                (filtered_results['Parameter'] == x_param) & 
                (filtered_results['Value'] == x_val)
            ]
            
            # Get rows for y parameter
            y_rows = filtered_results[
                (filtered_results['Parameter'] == y_param) & 
                (filtered_results['Value'] == y_val)
            ]
            
            # Combine and calculate the average metric value for this parameter combination
            combined_value = (x_rows[metric].values[0] + y_rows[metric].values[0]) / 2
            heatmap_data[i, j] = combined_value
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels([f'{x:.4f}' for x in x_values])
    ax.set_yticklabels([f'{y:.4f}' for y in y_values])
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add labels and title
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f'Sensitivity Heatmap: {metric}')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, len(x_values), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(y_values), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # Add values in each cell
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            text_color = 'white' if heatmap_data[i, j] > np.mean(heatmap_data) else 'black'
            ax.text(j, i, f'{heatmap_data[i, j]:.4f}', 
                    ha='center', va='center', color=text_color, fontsize=9)
    
    plt.tight_layout()
    return fig


def create_dashboard(model_results, portfolio_results=None, diagnostics_results=None, figsize=(16, 14)):
    """
    Create a comprehensive dashboard visualization of model and portfolio results.
    
    Parameters:
    - model_results: Dictionary with model simulation results.
    - portfolio_results: Dictionary with portfolio valuation results.
    - diagnostics_results: Dictionary with diagnostic test results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create a figure with a flexible grid layout
    fig = plt.figure(figsize=figsize)
    
    # Define grid specification based on available results
    if portfolio_results is not None and diagnostics_results is not None:
        # Full dashboard with all components
        gs = fig.add_gridspec(3, 2)
    elif portfolio_results is not None or diagnostics_results is not None:
        # Dashboard with model results and one other component
        gs = fig.add_gridspec(2, 2)
    else:
        # Only model results
        gs = fig.add_gridspec(2, 1)
    
    # 1. Plot model simulation results
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract model paths from results
    if 'paths' in model_results:
        paths = model_results['paths']
        n_paths, n_steps = paths.shape
        
        # Plot sample paths in the first subplot
        time_axis = np.linspace(0, 1, n_steps)  # Assuming time from 0 to 1
        for i in range(min(5, n_paths)):
            ax1.plot(time_axis, paths[i, :], lw=1, alpha=0.7)
        
        # Add mean and 95% CI
        mean_path = np.mean(paths, axis=0)
        std_path = np.std(paths, axis=0)
        ax1.plot(time_axis, mean_path, 'r-', lw=2, label='Mean')
        ax1.fill_between(time_axis, mean_path - 1.96 * std_path, mean_path + 1.96 * std_path, 
                         color='r', alpha=0.2, label='95% CI')
        
        ax1.set_title('Interest Rate Model Simulation')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Rate')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot rate distribution at final time in the second subplot
        final_rates = paths[:, -1]
        sns.histplot(final_rates, kde=True, ax=ax2, bins=30, alpha=0.7, color='steelblue')
        
        # Add normal fit
        mu, std = norm.fit(final_rates)
        x = np.linspace(final_rates.min(), final_rates.max(), 100)
        p = norm.pdf(x, mu, std)
        ax2.plot(x, p * n_paths * (final_rates.max() - final_rates.min()) / 30, 'r-', linewidth=2,
                label=f'Normal fit (μ={mu:.4f}, σ={std:.4f})')
        
        ax2.set_title('Terminal Rate Distribution')
        ax2.set_xlabel('Rate')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
    
    # 2. Plot portfolio results if available
    if portfolio_results is not None:
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'portfolio_values' in portfolio_results:
            portfolio_values = portfolio_results['portfolio_values']
            
            # Plot portfolio value distribution
            sns.histplot(portfolio_values, kde=True, ax=ax3, bins=30, alpha=0.7, color='lightgreen')
            
            # Add key statistics as vertical lines
            mean_value = np.mean(portfolio_values)
            median_value = np.median(portfolio_values)
            var_95 = np.percentile(portfolio_values, 5)  # 95% VaR
            
            ax3.axvline(mean_value, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_value:.2f}')
            ax3.axvline(median_value, color='black', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
            ax3.axvline(var_95, color='darkred', linestyle='-.', linewidth=2, label=f'95% VaR: {mean_value - var_95:.2f}')
            
            ax3.set_title('Portfolio Value Distribution')
            ax3.set_xlabel('Portfolio Value')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
            
            # Plot risk metrics in the fourth subplot (using a horizontal bar chart)
            if 'risk_metrics' in portfolio_results:
                risk_metrics = portfolio_results['risk_metrics']
                
                # Select key metrics to display
                key_metrics = ['VaR_95', 'CVaR_95', 'sharpe_ratio', 'sortino_ratio']
                metric_values = [risk_metrics.get(m, 0) for m in key_metrics]
                
                # Create horizontal bar chart
                bars = ax4.barh(key_metrics, metric_values, color='lightblue', alpha=0.7)
                
                # Add values on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax4.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                             f'{metric_values[i]:.4f}', va='center')
                
                ax4.set_title('Portfolio Risk Metrics')
                ax4.set_xlabel('Value')
                ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Plot diagnostic results if available
        if diagnostics_results is not None:
            if gs.get_geometry()[0] == 3:  # If we have 3 rows
                ax5 = fig.add_subplot(gs[2, 0])
                ax6 = fig.add_subplot(gs[2, 1])
            else:  # If we only have 2 rows but diagnostics are available
                # Create new axes for diagnostics
                ax5 = ax3
                ax6 = ax4
            
            if 'test_results' in diagnostics_results:
                test_results = diagnostics_results['test_results']
                
                # Display test results as a table in the fifth subplot
                test_names = list(test_results.keys())
                test_values = list(test_results.values())
                
                # Convert test values to strings with formatting
                test_values_str = []
                for val in test_values:
                    if isinstance(val, tuple):
                        # Format for test statistic and p-value
                        test_values_str.append(f'{val[0]:.4f} (p={val[1]:.4f})')
                    else:
                        test_values_str.append(str(val))
                
                # Create a colorful table based on test results
                cell_colors = []
                for val in test_values:
                    if isinstance(val, tuple) and len(val) > 1:
                        # Color based on p-value significance
                        if val[1] < 0.01:
                            cell_colors.append('lightcoral')  # Highly significant (rejected)
                        elif val[1] < 0.05:
                            cell_colors.append('lightsalmon')  # Significant (rejected)
                        elif val[1] < 0.1:
                            cell_colors.append('khaki')  # Marginally significant
                        else:
                            cell_colors.append('lightgreen')  # Not significant (not rejected)
                    else:
                        cell_colors.append('white')
                
                # Create table with colored cells
                if test_names and test_values_str:  # Only create the table if there's data
                    # Create table with colored cells
                    table = ax5.table(
                        cellText=[[name, val] for name, val in zip(test_names, test_values_str)],
                        colLabels=['Test', 'Result'],
                        loc='center',
                        cellColours=[[('lightgray' if i % 2 == 0 else 'white'), color] 
                                    for i, color in enumerate(cell_colors)]
                    )
                    
                    # Adjust table formatting
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                else:
                    # Handle the case where there are no test results
                    ax5.text(0.5, 0.5, 'No diagnostic test results available', 
                            horizontalalignment='center', verticalalignment='center')
                
                ax5.set_title('Diagnostic Test Results')
                ax5.axis('off')  # Hide axis for the table
                
            if 'residuals' in diagnostics_results:
                residuals = diagnostics_results['residuals']
                
                # Plot residuals over time
                time_points = np.arange(len(residuals))
                ax6.plot(time_points, residuals, 'o-', markersize=3, alpha=0.6)
                ax6.axhline(y=0, color='red', linestyle='-', alpha=0.7)
                
                # Add rolling mean
                window = max(20, len(residuals) // 10)
                rolling_mean = pd.Series(residuals).rolling(window=window).mean()
                ax6.plot(time_points, rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean ({window})')
                
                ax6.set_title('Model Residuals')
                ax6.set_xlabel('Time')
                ax6.set_ylabel('Residual')
                ax6.grid(True, linestyle='--', alpha=0.7)
                ax6.legend()
    
    plt.tight_layout()
    return fig


def plot_rate_vs_bond_price(model, maturities, rate_range, figsize=(12, 8)):
    """
    Create a visualization of the relationship between interest rates and bond prices.
    
    Parameters:
    - model: Interest rate model with analytical_zcb_price method.
    - maturities: List of bond maturities to visualize.
    - rate_range: Tuple of (min_rate, max_rate) to plot.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a range of interest rates
    rates = np.linspace(rate_range[0], rate_range[1], 100)
    
    # Calculate bond prices for each maturity and interest rate
    for maturity in maturities:
        prices = np.array([model.analytical_zcb_price(r, maturity) for r in rates])
        ax.plot(rates, prices, label=f'T = {maturity} years', linewidth=2)
    
    # Calculate duration and convexity for the longest maturity
    longest_maturity = max(maturities)
    r0 = np.mean(rates)  # Reference rate for calculations
    
    # Calculate price at reference rate
    p0 = model.analytical_zcb_price(r0, longest_maturity)
    
    # Calculate numerical duration
    dr = 0.0001  # Small rate change for numerical differentiation
    p_plus = model.analytical_zcb_price(r0 + dr, longest_maturity)
    p_minus = model.analytical_zcb_price(r0 - dr, longest_maturity)
    
    duration = -(p_plus - p_minus) / (2 * dr * p0)
    
    # Calculate numerical convexity
    convexity = (p_plus + p_minus - 2 * p0) / (dr**2 * p0)
    
    # Plot tangent line based on duration at r0
    tangent = p0 - duration * p0 * (rates - r0)
    ax.plot(rates, tangent, '--', color='black', alpha=0.6, 
            label=f'Duration approx. at r={r0:.2f}')
    
    # Plot quadratic approximation based on duration and convexity
    quadratic = p0 * (1 - duration * (rates - r0) + 0.5 * convexity * (rates - r0)**2)
    ax.plot(rates, quadratic, '-.', color='red', alpha=0.6, 
            label=f'Convexity approx. at r={r0:.2f}')
    
    # Add reference point
    ax.plot(r0, p0, 'o', color='black', markersize=6)
    
    # Add annotations
    ax.annotate(f'Duration = {duration:.2f}\nConvexity = {convexity:.2f}',
                xy=(r0, p0), xytext=(r0 + 0.01, p0 - 0.05),
                arrowprops=dict(arrowstyle='->'))
    
    # Add labels and title
    ax.set_xlabel('Interest Rate', fontsize=12)
    ax.set_ylabel('Bond Price', fontsize=12)
    ax.set_title('Relationship Between Interest Rates and Bond Prices', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig


def save_results_to_csv(results, filename):
    """
    Save simulation or analysis results to a CSV file.
    
    Parameters:
    - results: Dictionary or pandas DataFrame with results.
    - filename: Output filename.
    
    Returns:
    - success: Boolean indicating if the save was successful.
    """
    try:
        if isinstance(results, dict):
            # Convert dict to DataFrame if necessary
            df = pd.DataFrame()
            
            # Process each key in the dictionary
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    # For arrays, create columns with indexed names
                    if value.ndim == 1:
                        df[key] = value
                    else:
                        for i in range(value.shape[1]):
                            df[f'{key}_{i}'] = value[:, i]
                elif isinstance(value, (int, float, str, bool)):
                    # For scalar values, create a column with constant values
                    df[key] = [value] * (len(df) if len(df) > 0 else 1)
                elif isinstance(value, dict):
                    # For nested dictionaries, flatten them
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)):
                            df[f'{key}_{subkey}'] = [subvalue] * (len(df) if len(df) > 0 else 1)
            
            # Save DataFrame to CSV
            df.to_csv(filename, index=False)
        elif isinstance(results, pd.DataFrame):
            # If already a DataFrame, save directly
            results.to_csv(filename, index=False)
        else:
            print(f"Unsupported results type: {type(results)}")
            return False
        
        print(f"Results successfully saved to {filename}")
        return True
    
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False


def load_results_from_csv(filename):
    """
    Load simulation or analysis results from a CSV file.
    
    Parameters:
    - filename: Input CSV filename.
    
    Returns:
    - results: pandas DataFrame with loaded results.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filename)
        print(f"Results successfully loaded from {filename}")
        return df
    
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None


def compare_models(models_dict, simulation_params, plot_type='paths', figsize=(12, 8)):
    """
    Compare multiple interest rate models using the same simulation parameters.
    
    Parameters:
    - models_dict: Dictionary mapping model names to model instances.
    - simulation_params: Dictionary with n_paths, n_steps, and dt.
    - plot_type: Type of comparison plot ('paths', 'distribution', or 'stats').
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Extract simulation parameters
    n_paths = simulation_params.get('n_paths', 1000)
    n_steps = simulation_params.get('n_steps', 100)
    dt = simulation_params.get('dt', 0.01)
    
    # Simulate paths for each model
    simulated_paths = {}
    for model_name, model_instance in models_dict.items():
        simulated_paths[model_name] = model_instance.simulate(n_paths, n_steps, dt)
    
    # Create comparison plot based on plot type
    if plot_type == 'paths':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean paths for each model
        time_axis = np.linspace(0, n_steps * dt, n_steps + 1)
        
        for model_name, paths in simulated_paths.items():
            mean_path = np.mean(paths, axis=0)
            ax.plot(time_axis, mean_path, linewidth=2, label=f'{model_name} (Mean)')
            
            # Add confidence intervals
            std_path = np.std(paths, axis=0)
            ax.fill_between(time_axis, 
                           mean_path - 1.96 * std_path, 
                           mean_path + 1.96 * std_path, 
                           alpha=0.2)
        
        # Add labels and title
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Interest Rate', fontsize=12)
        ax.set_title('Comparison of Model Mean Paths', fontsize=14)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    elif plot_type == 'distribution':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot terminal rate distributions for each model
        for model_name, paths in simulated_paths.items():
            terminal_rates = paths[:, -1]
            
            # Plot KDE
            sns.kdeplot(terminal_rates, label=model_name, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Terminal Rate', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Comparison of Terminal Rate Distributions', fontsize=14)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    elif plot_type == 'stats':
        # Calculate statistics for each model
        stats = []
        
        for model_name, paths in simulated_paths.items():
            terminal_rates = paths[:, -1]
            
            stats.append({
                'Model': model_name,
                'Mean': np.mean(terminal_rates),
                'Median': np.median(terminal_rates),
                'Std Dev': np.std(terminal_rates),
                'Min': np.min(terminal_rates),
                'Max': np.max(terminal_rates),
                'Skewness': pd.Series(terminal_rates).skew(),
                'Kurtosis': pd.Series(terminal_rates).kurtosis()
            })
        
        # Convert to DataFrame for easier plotting
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.set_index('Model')
        
        # Create bar plots for the statistics
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each statistic
        metrics = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']
        for i, metric in enumerate(metrics):
            ax = axes[i]
            stats_df[metric].plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
            
            ax.set_title(metric)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for j, v in enumerate(stats_df[metric]):
                ax.text(j, v * 1.05, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    return fig