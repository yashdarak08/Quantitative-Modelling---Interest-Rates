"""
backtesting.py
--------------
Framework for backtesting interest rate models against historical data.
Includes:
- Parameter calibration
- Performance evaluation
- Model selection
- Statistical testing of model forecasts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_interest_rate_data(file_path=None, start_date=None, end_date=None):
    """
    Load historical interest rate data for backtesting.
    If file_path is None, synthetic data is generated.
    
    Parameters:
    - file_path: Path to CSV file with historical data.
    - start_date: Start date for filtering data.
    - end_date: End date for filtering data.
    
    Returns:
    - DataFrame with dates and interest rates.
    """
    if file_path is not None:
        # Load data from CSV file
        try:
            data = pd.read_csv(file_path, parse_dates=True)
            
            # Check if date column exists
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                data[date_col] = pd.to_datetime(data[date_col])
                data = data.set_index(date_col)
            
            # Filter by date range if provided
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            return data
        
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Generating synthetic data instead.")
    
    # Generate synthetic data if file not provided or loading failed
    print("Generating synthetic interest rate data for backtesting...")
    
    # Create date range
    if start_date is None:
        start_date = '2010-01-01'
    if end_date is None:
        end_date = '2023-12-31'
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Generate synthetic rates using a mean-reverting process
    n_days = len(dates)
    rates = np.zeros(n_days)
    
    # Initial rate
    rates[0] = 0.02
    
    # Model parameters
    a = 0.05  # Mean reversion speed
    b = 0.03  # Long-term mean
    sigma = 0.002  # Volatility
    
    # Simulate the process
    for i in range(1, n_days):
        dr = a * (b - rates[i-1]) + sigma * np.random.randn()
        rates[i] = rates[i-1] + dr
    
    # Ensure rates are positive
    rates = np.maximum(rates, 0.001)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'rate': rates
    })
    
    return df.set_index('date')


def split_data(data, train_ratio=0.7, validation_ratio=0.15):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    - data: DataFrame with interest rate data.
    - train_ratio: Proportion of data for training.
    - validation_ratio: Proportion of data for validation.
    
    Returns:
    - train_data, validation_data, test_data: Split DataFrames.
    """
    n = len(data)
    train_size = int(n * train_ratio)
    validation_size = int(n * validation_ratio)
    
    train_data = data.iloc[:train_size]
    validation_data = data.iloc[train_size:train_size + validation_size]
    test_data = data.iloc[train_size + validation_size:]
    
    return train_data, validation_data, test_data


def calibrate_model(model_class, rate_data, initial_params, bounds=None):
    """
    Calibrate model parameters to historical data.
    
    Parameters:
    - model_class: Model class to calibrate (e.g., VasicekModel).
    - rate_data: Series or array of historical rates.
    - initial_params: Dictionary of initial parameter values.
    - bounds: Dictionary of parameter bounds (min, max).
    
    Returns:
    - calibrated_model: Model instance with calibrated parameters.
    - optimization_result: Full optimization result.
    """
    # Extract rates
    if isinstance(rate_data, pd.DataFrame):
        rates = rate_data.iloc[:, 0].values
    else:
        rates = np.array(rate_data)
    
    # Initial parameter values and bounds
    param_names = list(initial_params.keys())
    initial_values = [initial_params[name] for name in param_names]
    
    if bounds is None:
        bounds = {}
        for name in param_names:
            if name == 'a':  # Mean reversion speed
                bounds[name] = (0.001, 1.0)
            elif name == 'b':  # Long-term mean
                bounds[name] = (0.001, 0.10)
            elif name == 'sigma':  # Volatility
                bounds[name] = (0.0001, 0.05)
            else:
                bounds[name] = (0.0001, 10.0)
    
    # Parameter bounds for optimization
    param_bounds = [bounds.get(name, (0.0001, 10.0)) for name in param_names]
    
    # Define the objective function to minimize
    def objective_function(params):
        # Create a dictionary of parameter values
        param_dict = {name: value for name, value in zip(param_names, params)}
        
        # Initial rate
        r0 = rates[0]
        param_dict['r0'] = r0
        
        # Create model instance
        model = model_class(**param_dict)
        
        # Calculate log-likelihood
        log_likelihood = 0
        
        for i in range(1, len(rates)):
            # Get conditional moments based on the model
            if hasattr(model, 'conditional_moments'):
                cond_mean, cond_var = model.conditional_moments(rates[i-1], 1/252)
            else:
                # Default for Vasicek model
                dt = 1/252  # Daily data (assuming 252 trading days per year)
                a, b, sigma = param_dict['a'], param_dict['b'], param_dict['sigma']
                
                # Conditional mean and variance
                cond_mean = rates[i-1] * np.exp(-a * dt) + b * (1 - np.exp(-a * dt))
                cond_var = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * dt))
            
            # Calculate log-likelihood contribution
            if cond_var > 0:
                log_likelihood -= 0.5 * np.log(2 * np.pi * cond_var)
                log_likelihood -= 0.5 * ((rates[i] - cond_mean) ** 2) / cond_var
            else:
                # If conditional variance is zero (numerical issues)
                log_likelihood -= 100  # Penalty
        
        # Return negative log-likelihood for minimization
        return -log_likelihood
    
    # Perform optimization
    result = minimize(
        objective_function,
        x0=initial_values,
        bounds=param_bounds,
        method='L-BFGS-B'
    )
    
    # Create calibrated model
    calibrated_params = {name: value for name, value in zip(param_names, result.x)}
    calibrated_params['r0'] = rates[-1]  # Set r0 to most recent rate for forecasting
    
    calibrated_model = model_class(**calibrated_params)
    
    # Add optimization result to model for reference
    calibrated_model.calibration_result = result
    calibrated_model.calibrated_params = calibrated_params
    
    return calibrated_model, result


def evaluate_calibration(calibrated_model, validation_data, horizon=1, n_simulations=1000):
    """
    Evaluate model calibration using validation data.
    
    Parameters:
    - calibrated_model: Calibrated model instance.
    - validation_data: Validation dataset.
    - horizon: Forecast horizon in days.
    - n_simulations: Number of Monte Carlo simulations for each forecast.
    
    Returns:
    - evaluation_metrics: Dictionary of evaluation metrics.
    """
    # Extract rates
    if isinstance(validation_data, pd.DataFrame):
        rates = validation_data.iloc[:, 0].values
    else:
        rates = np.array(validation_data)
    
    # Initialize arrays for forecasts and errors
    n_forecasts = len(rates) - horizon
    point_forecasts = np.zeros(n_forecasts)
    forecast_errors = np.zeros(n_forecasts)
    forecast_intervals = np.zeros((n_forecasts, 2))  # 95% prediction intervals
    coverage = 0  # Count how many actual values fall within the prediction interval
    
    # Perform rolling forecasts
    for i in range(n_forecasts):
        # Set current rate
        r0 = rates[i]
        calibrated_model.r0 = r0
        
        # Simulate future paths
        dt = 1/252  # Daily data
        paths = calibrated_model.simulate(n_simulations, horizon, dt)
        
        # Extract terminal rates (forecasts)
        forecasts = paths[:, -1]
        
        # Calculate point forecast (mean) and prediction interval
        point_forecast = np.mean(forecasts)
        lower_bound = np.percentile(forecasts, 2.5)
        upper_bound = np.percentile(forecasts, 97.5)
        
        # Store results
        point_forecasts[i] = point_forecast
        forecast_errors[i] = rates[i + horizon] - point_forecast
        forecast_intervals[i, :] = [lower_bound, upper_bound]
        
        # Check if actual value falls within prediction interval
        if lower_bound <= rates[i + horizon] <= upper_bound:
            coverage += 1
    
    # Calculate evaluation metrics
    mse = np.mean(forecast_errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(forecast_errors))
    coverage_ratio = coverage / n_forecasts
    
    # Calculate directional accuracy
    actual_directions = np.sign(rates[horizon:horizon+n_forecasts] - rates[:n_forecasts])
    predicted_directions = np.sign(point_forecasts - rates[:n_forecasts])
    directional_accuracy = np.mean(actual_directions == predicted_directions)
    
    # Calculate information ratio
    forecast_std = np.std(forecast_errors)
    information_ratio = rmse / forecast_std if forecast_std > 0 else np.inf
    
    # Check for bias
    bias = np.mean(forecast_errors)
    
    # Calculate autocorrelation of errors
    error_autocorr = acf(forecast_errors, nlags=5)[1:]  # Skip lag 0
    
    # Compile evaluation metrics
    evaluation_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Coverage': coverage_ratio,
        'Directional_Accuracy': directional_accuracy,
        'Bias': bias,
        'Information_Ratio': information_ratio,
        'Error_Autocorrelation': error_autocorr,
        'Forecasts': point_forecasts,
        'Errors': forecast_errors,
        'Prediction_Intervals': forecast_intervals
    }
    
    return evaluation_metrics


def compare_models(models, validation_data, horizon=1, n_simulations=1000):
    """
    Compare multiple calibrated models on validation data.
    
    Parameters:
    - models: Dictionary of calibrated model instances (name -> model).
    - validation_data: Validation dataset.
    - horizon: Forecast horizon in days.
    - n_simulations: Number of simulations for each forecast.
    
    Returns:
    - comparison_results: DataFrame with comparison metrics.
    """
    # Initialize dictionary to store evaluation metrics
    all_metrics = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        metrics = evaluate_calibration(model, validation_data, horizon, n_simulations)
        all_metrics[model_name] = metrics
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, metrics in all_metrics.items():
        row = {
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'Coverage': metrics['Coverage'],
            'Directional_Accuracy': metrics['Directional_Accuracy'],
            'Bias': metrics['Bias'],
            'Information_Ratio': metrics['Information_Ratio']
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add model rankings for each metric
    for metric in ['RMSE', 'MAE', 'Bias', 'Information_Ratio']:
        # Lower is better for these metrics
        comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank()
    
    for metric in ['Coverage', 'Directional_Accuracy']:
        # Higher is better for these metrics
        comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank(ascending=False)
    
    # Calculate overall rank
    rank_columns = [col for col in comparison_df.columns if col.endswith('_Rank')]
    comparison_df['Overall_Rank'] = comparison_df[rank_columns].mean(axis=1)
    
    # Store detailed metrics for reference
    comparison_df.all_metrics = all_metrics
    
    return comparison_df


def plot_forecast_evaluation(evaluation_metrics, actual_rates, figsize=(14, 10)):
    """
    Plot the results of forecast evaluation.
    
    Parameters:
    - evaluation_metrics: Dictionary of evaluation metrics.
    - actual_rates: Array of actual interest rates.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract relevant metrics
    forecasts = evaluation_metrics['Forecasts']
    errors = evaluation_metrics['Errors']
    prediction_intervals = evaluation_metrics['Prediction_Intervals']
    
    # Determine plot range
    n_forecasts = len(forecasts)
    actual_subset = actual_rates[:n_forecasts+1]
    
    # 1. Plot forecasts vs actual rates
    ax = axes[0, 0]
    
    # Plot actual rates
    ax.plot(actual_subset, 'b-', label='Actual Rates', linewidth=2)
    
    # Plot forecasts (shifted by 1 since they predict the next period)
    forecast_indices = np.arange(1, n_forecasts+1)
    ax.plot(forecast_indices, forecasts, 'r--', label='Forecasts', linewidth=2)
    
    # Plot prediction intervals
    ax.fill_between(
        forecast_indices,
        prediction_intervals[:, 0],
        prediction_intervals[:, 1],
        color='r', alpha=0.2, label='95% Prediction Interval'
    )
    
    ax.set_title('Forecasts vs Actual Rates')
    ax.set_xlabel('Time')
    ax.set_ylabel('Interest Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 2. Plot forecast errors over time
    ax = axes[0, 1]
    
    ax.plot(forecast_indices, errors, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Add error bands
    std_error = np.std(errors)
    ax.axhline(y=2*std_error, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=-2*std_error, color='gray', linestyle=':', alpha=0.7)
    ax.fill_between(forecast_indices, -2*std_error, 2*std_error, color='gray', alpha=0.1)
    
    ax.set_title('Forecast Errors')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Plot error histogram
    ax = axes[1, 0]
    
    ax.hist(errors, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add normal distribution fit
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    x = np.linspace(min(errors), max(errors), 100)
    p = norm.pdf(x, mu, std)
    
    # Scale the PDF to match histogram height
    hist_height = np.histogram(errors, bins=20)[0].max()
    p_scaled = p * hist_height / p.max()
    
    ax.plot(x, p_scaled, 'r-', linewidth=2, 
           label=f'Normal fit (μ={mu:.4f}, σ={std:.4f})')
    
    ax.set_title('Error Distribution')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 4. Plot error autocorrelation
    ax = axes[1, 1]
    
    error_autocorr = evaluation_metrics['Error_Autocorrelation']
    lags = np.arange(1, len(error_autocorr) + 1)
    
    ax.bar(lags, error_autocorr, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add significance bands
    n = len(errors)
    sig_level = 1.96 / np.sqrt(n)
    ax.axhline(y=sig_level, color='r', linestyle='--', alpha=0.7)
    ax.axhline(y=-sig_level, color='r', linestyle='--', alpha=0.7)
    
    ax.set_title('Error Autocorrelation')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df, figsize=(14, 10)):
    """
    Plot comparison of multiple models.
    
    Parameters:
    - comparison_df: DataFrame with model comparison results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. Plot RMSE and MAE
    ax = axes[0]
    
    models = comparison_df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['RMSE'], width, label='RMSE', color='skyblue', alpha=0.7)
    ax.bar(x + width/2, comparison_df['MAE'], width, label='MAE', color='lightgreen', alpha=0.7)
    
    ax.set_title('Error Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Error')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.legend()
    
    # 2. Plot Coverage and Directional Accuracy
    ax = axes[1]
    
    ax.bar(x - width/2, comparison_df['Coverage'], width, label='95% CI Coverage', color='coral', alpha=0.7)
    ax.bar(x + width/2, comparison_df['Directional_Accuracy'], width, label='Directional Accuracy', color='mediumpurple', alpha=0.7)
    
    # Add reference line for ideal 95% coverage
    ax.axhline(y=0.95, color='coral', linestyle='--', alpha=0.7)
    
    ax.set_title('Coverage and Directional Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.legend()
    
    # 3. Plot Bias
    ax = axes[2]
    
    ax.bar(x, comparison_df['Bias'], color='gold', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax.set_title('Forecast Bias')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Bias')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 4. Plot Overall Rank
    ax = axes[3]
    
    # Sort by rank
    sorted_df = comparison_df.sort_values('Overall_Rank')
    models_sorted = sorted_df['Model']
    ranks = sorted_df['Overall_Rank']
    
    bars = ax.barh(np.arange(len(models_sorted)), ranks, color='lightseagreen', alpha=0.7)
    
    # Add rank values
    for i, (bar, rank) in enumerate(zip(bars, ranks)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{rank:.2f}', va='center')
    
    ax.set_title('Overall Model Ranking')
    ax.set_yticks(np.arange(len(models_sorted)))
    ax.set_yticklabels(models_sorted)
    ax.set_xlabel('Rank (lower is better)')
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    return fig


def stress_test_model(calibrated_model, scenarios, n_simulations=1000, dt=1/252, horizon=252):
    """
    Perform stress testing on a calibrated model under different scenarios.
    
    Parameters:
    - calibrated_model: Calibrated model instance.
    - scenarios: Dictionary mapping scenario names to dictionaries of parameter adjustments.
    - n_simulations: Number of Monte Carlo simulations.
    - dt: Time step size.
    - horizon: Simulation horizon in steps.
    
    Returns:
    - stress_test_results: Dictionary of stress test results.
    """
    # Store original parameters
    original_params = calibrated_model.calibrated_params.copy()
    
    # Initialize results
    stress_test_results = {
        'parameters': {},
        'simulations': {},
        'statistics': {}
    }
    
    # Baseline scenario
    baseline_paths = calibrated_model.simulate(n_simulations, horizon, dt)
    
    # Calculate statistics for baseline
    baseline_final = baseline_paths[:, -1]
    baseline_stats = {
        'mean': np.mean(baseline_final),
        'median': np.median(baseline_final),
        'std': np.std(baseline_final),
        'min': np.min(baseline_final),
        'max': np.max(baseline_final),
        'q05': np.percentile(baseline_final, 5),
        'q95': np.percentile(baseline_final, 95)
    }
    
    # Store baseline results
    stress_test_results['parameters']['Baseline'] = original_params
    stress_test_results['simulations']['Baseline'] = baseline_paths
    stress_test_results['statistics']['Baseline'] = baseline_stats
    
    # Run stress scenarios
    for scenario_name, param_adjustments in scenarios.items():
        # Create a copy of the original parameters
        scenario_params = original_params.copy()
        
        # Apply parameter adjustments
        for param, adjustment in param_adjustments.items():
            if param in scenario_params:
                scenario_params[param] = adjustment
        
        # Update model with scenario parameters
        for param, value in scenario_params.items():
            setattr(calibrated_model, param, value)
        
        # Run simulation
        scenario_paths = calibrated_model.simulate(n_simulations, horizon, dt)
        
        # Calculate statistics
        scenario_final = scenario_paths[:, -1]
        scenario_stats = {
            'mean': np.mean(scenario_final),
            'median': np.median(scenario_final),
            'std': np.std(scenario_final),
            'min': np.min(scenario_final),
            'max': np.max(scenario_final),
            'q05': np.percentile(scenario_final, 5),
            'q95': np.percentile(scenario_final, 95)
        }
        
        # Store scenario results
        stress_test_results['parameters'][scenario_name] = scenario_params
        stress_test_results['simulations'][scenario_name] = scenario_paths
        stress_test_results['statistics'][scenario_name] = scenario_stats
    
    # Restore original parameters
    for param, value in original_params.items():
        setattr(calibrated_model, param, value)
    
    return stress_test_results


def plot_stress_test_results(stress_test_results, figsize=(16, 10)):
    """
    Plot the results of stress tests.
    
    Parameters:
    - stress_test_results: Dictionary of stress test results.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = fig.add_gridspec(2, 2)
    
    # 1. Plot terminal rate distributions for each scenario
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get scenario names and simulations
    scenarios = list(stress_test_results['simulations'].keys())
    
    # Plot distributions
    for scenario in scenarios:
        paths = stress_test_results['simulations'][scenario]
        final_rates = paths[:, -1]
        
        # Create KDE plot
        sns.kdeplot(final_rates, ax=ax1, label=scenario)
    
    ax1.set_title('Terminal Rate Distributions')
    ax1.set_xlabel('Interest Rate')
    ax1.set_ylabel('Density')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Plot statistics comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create a DataFrame from statistics
    stats_data = []
    
    for scenario, stats in stress_test_results['statistics'].items():
        row = {'Scenario': scenario}
        row.update(stats)
        stats_data.append(row)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Plot means with error bars
    scenarios = stats_df['Scenario']
    means = stats_df['mean']
    stds = stats_df['std']
    q05 = stats_df['q05']
    q95 = stats_df['q95']
    
    x = np.arange(len(scenarios))
    
    # Plot mean with standard deviation
    ax2.errorbar(x, means, yerr=stds, fmt='o', capsize=5, label='Mean ± Std Dev', color='blue')
    
    # Plot 5th-95th percentile range
    for i, (low, high) in enumerate(zip(q05, q95)):
        ax2.plot([i, i], [low, high], 'r-', alpha=0.7)
    
    # Add dots for min and max
    ax2.plot(x, stats_df['min'], 'v', color='gray', alpha=0.7, label='Min/Max')
    ax2.plot(x, stats_df['max'], '^', color='gray', alpha=0.7)
    
    ax2.set_title('Terminal Rate Statistics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.set_ylabel('Interest Rate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 3. Plot mean paths for each scenario
    ax3 = fig.add_subplot(gs[1, :])
    
    # Time axis
    n_steps = stress_test_results['simulations']['Baseline'].shape[1]
    time_axis = np.linspace(0, 1, n_steps)  # Normalized time
    
    # Plot mean path for each scenario
    for scenario in scenarios:
        paths = stress_test_results['simulations'][scenario]
        mean_path = np.mean(paths, axis=0)
        
        ax3.plot(time_axis, mean_path, label=scenario, linewidth=2)
    
    ax3.set_title('Mean Interest Rate Paths')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Interest Rate')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    return fig


def visualize_yields_and_forward_rates(model, maturities, r0, figsize=(14, 10)):
    """
    Visualize the implied yield curve and forward rates from a calibrated model.
    
    Parameters:
    - model: Calibrated interest rate model.
    - maturities: Array of maturities (in years).
    - r0: Current short rate.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Calculate zero-coupon bond prices and yields
    prices = np.zeros_like(maturities)
    yields = np.zeros_like(maturities)
    
    for i, T in enumerate(maturities):
        # Bond price using the model's analytical formula
        if hasattr(model, 'analytical_zcb_price'):
            prices[i] = model.analytical_zcb_price(r0, T)
            # Convert price to yield
            yields[i] = -np.log(prices[i]) / T
        else:
            # If no analytical formula, use short rate approximation
            yields[i] = r0  # Simplified
    
    # Calculate forward rates
    forward_rates = np.zeros_like(maturities)
    
    # First forward rate is just the spot rate
    forward_rates[0] = r0
    
    # Calculate forward rates from yields
    for i in range(1, len(maturities)):
        t1 = maturities[i-1]
        t2 = maturities[i]
        
        # Forward rate formula: f(t1,t2) = (Y(t2)*t2 - Y(t1)*t1) / (t2 - t1)
        forward_rates[i] = (yields[i] * t2 - yields[i-1] * t1) / (t2 - t1)
    
    # 1. Plot the yield curve
    ax1.plot(maturities, yields, 'b-', linewidth=2, marker='o')
    
    # Add reference line for current short rate
    ax1.axhline(y=r0, color='r', linestyle='--', alpha=0.7, label=f'Current Short Rate: {r0:.4f}')
    
    ax1.set_title('Model-Implied Yield Curve')
    ax1.set_xlabel('Maturity (years)')
    ax1.set_ylabel('Yield')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Plot forward rates
    ax2.plot(maturities, forward_rates, 'g-', linewidth=2, marker='s')
    
    # Add reference line for current short rate
    ax2.axhline(y=r0, color='r', linestyle='--', alpha=0.7, label=f'Current Short Rate: {r0:.4f}')
    
    # Add model parameters for reference
    params_text = ', '.join([f'{k}={v:.4f}' for k, v in model.calibrated_params.items() 
                           if k not in ['r0']])
    ax2.text(0.02, 0.02, f'Model Parameters: {params_text}', transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax2.set_title('Model-Implied Forward Rates')
    ax2.set_xlabel('Maturity (years)')
    ax2.set_ylabel('Forward Rate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    return fig