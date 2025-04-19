"""
diagnostics.py
--------------
Statistical diagnostic tests for model validation and backtesting.
Includes:
- Normality tests (Jarque-Bera, Shapiro-Wilk, D'Agostino)
- Autocorrelation tests (Ljung-Box, Durbin-Watson)
- Heteroskedasticity tests (Levene, Breusch-Pagan)
- Goodness-of-fit tests
- Backtesting framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    jarque_bera, shapiro, normaltest, 
    anderson, levene, kstest, chi2_contingency
)
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, het_breuschpagan, het_white
)
from statsmodels.tsa.stattools import adfuller   
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def jarque_bera_test(data):
    """
    Perform the Jarque-Bera test for normality.
    
    Parameters:
    - data: 1D numpy array of sample data.
    
    Returns:
    - jb_stat: Test statistic.
    - p_value: p-value of the test.
    - is_normal: Boolean indicating if the null hypothesis (normality) is not rejected at 5% level.
    """
    jb_stat, p_value = jarque_bera(data)
    is_normal = p_value > 0.05
    return jb_stat, p_value, is_normal


def ljung_box_test(data, lags=10):
    """
    Perform the Ljung-Box test for autocorrelation.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - lags: Number of lags to test (default 10).
    
    Returns:
    - lb_stat: Test statistic.
    - p_value: p-value of the test.
    - is_independent: Boolean indicating if the null hypothesis (no autocorrelation) is not rejected at 5% level.
    """
    # The function returns a tuple of arrays for statistics and p-values
    lb_test = acorr_ljungbox(data, lags=[lags], return_df=False)
    lb_stat = lb_test[0][0]
    p_value = lb_test[1][0]
    is_independent = p_value > 0.05
    return lb_stat, p_value, is_independent


def levene_test(data, n_groups=2):
    """
    Perform Levene's test for equal variances.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - n_groups: Number of groups to split the data into (default 2).
    
    Returns:
    - levene_stat: Test statistic.
    - p_value: p-value of the test.
    - is_homoskedastic: Boolean indicating if the null hypothesis (equal variances) is not rejected at 5% level.
    """
    # Split data into groups
    n = len(data)
    group_size = n // n_groups
    groups = [data[i*group_size:(i+1)*group_size] for i in range(n_groups-1)]
    groups.append(data[(n_groups-1)*group_size:])
    
    levene_stat, p_value = levene(*groups)
    is_homoskedastic = p_value > 0.05
    return levene_stat, p_value, is_homoskedastic


def shapiro_wilk_test(data):
    """
    Perform the Shapiro-Wilk test for normality.
    
    Parameters:
    - data: 1D numpy array of sample data.
    
    Returns:
    - sw_stat: Test statistic.
    - p_value: p-value of the test.
    - is_normal: Boolean indicating if the null hypothesis (normality) is not rejected at 5% level.
    """
    sw_stat, p_value = shapiro(data)
    is_normal = p_value > 0.05
    return sw_stat, p_value, is_normal


def dagostino_test(data):
    """
    Perform the D'Agostino's K^2 test for normality.
    
    Parameters:
    - data: 1D numpy array of sample data.
    
    Returns:
    - dagostino_stat: Test statistic.
    - p_value: p-value of the test.
    - is_normal: Boolean indicating if the null hypothesis (normality) is not rejected at 5% level.
    """
    dagostino_stat, p_value = normaltest(data)
    is_normal = p_value > 0.05
    return dagostino_stat, p_value, is_normal


# def durbin_watson_test(data):
#     """
#     Perform the Durbin-Watson test for autocorrelation.
    
#     Parameters:
#     - data: 1D numpy array of sample data.
    
#     Returns:
#     - dw_stat: Durbin-Watson statistic (between 0 and 4, with 2 indicating no autocorrelation).
#     - autocorrelation_type: 'positive', 'negative', or 'none' based on the statistic.
#     """
#     dw_stat = durbin_watson(data)
    
#     if dw_stat < 1.5:
#         autocorrelation_type = 'positive'
#     elif dw_stat > 2.5:
#         autocorrelation_type = 'negative'
#     else:
#         autocorrelation_type = 'none'
    
#     return dw_stat, autocorrelation_type


def breusch_pagan_test(data, exog_vars=None):
    """
    Perform the Breusch-Pagan test for heteroskedasticity.
    
    Parameters:
    - data: 1D numpy array of residuals or returns.
    - exog_vars: Exogenous variables (if None, a simple trend is used).
    
    Returns:
    - bp_stat: Test statistic.
    - p_value: p-value of the test.
    - is_homoskedastic: Boolean indicating if the null hypothesis (homoskedasticity) is not rejected at 5% level.
    """
    if exog_vars is None:
        exog_vars = np.column_stack((np.ones(len(data)), np.arange(len(data))))
    
    bp_test = het_breuschpagan(data, exog_vars)
    bp_stat = bp_test[0]
    p_value = bp_test[1]
    is_homoskedastic = p_value > 0.05
    
    return bp_stat, p_value, is_homoskedastic


def adf_test(data):
    """
    Perform the Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    - data: 1D numpy array of time series data.
    
    Returns:
    - adf_stat: Test statistic.
    - p_value: p-value of the test.
    - is_stationary: Boolean indicating if the null hypothesis (non-stationarity) is rejected at 5% level.
    """
    adf_result = adfuller(data)
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    is_stationary = p_value < 0.05  # Note: For ADF test, small p-value rejects the null of non-stationarity
    
    return adf_stat, p_value, is_stationary


def kolmogorov_smirnov_test(data, distribution='norm', dist_params=(0, 1)):
    """
    Perform the Kolmogorov-Smirnov test for goodness-of-fit to a theoretical distribution.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - distribution: String representing the distribution ('norm', 'uniform', etc.).
    - dist_params: Parameters for the distribution.
    
    Returns:
    - ks_stat: Test statistic.
    - p_value: p-value of the test.
    - is_good_fit: Boolean indicating if the null hypothesis (data follows the distribution) is not rejected at 5% level.
    """
    ks_stat, p_value = kstest(data, distribution, dist_params)
    is_good_fit = p_value > 0.05
    
    return ks_stat, p_value, is_good_fit


def qq_plot(data, distribution='norm', figsize=(10, 8)):
    """
    Create a Q-Q plot to assess if data follows a theoretical distribution.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - distribution: String representing the distribution ('norm', 'uniform', etc.).
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    from scipy.stats import probplot
    
    fig, ax = plt.subplots(figsize=figsize)
    probplot(data, dist=distribution, plot=ax)
    
    ax.set_title(f'Q-Q Plot ({distribution} distribution)')
    ax.grid(True)
    
    return fig


def run_diagnostic_tests(data, test_names=None):
    """
    Run a battery of diagnostic tests on the data.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - test_names: List of test names to run (if None, all tests are run).
    
    Returns:
    - results: DataFrame with test results.
    """
    # Define all available tests
    all_tests = {
        'jarque_bera': jarque_bera_test,
        'shapiro_wilk': shapiro_wilk_test,
        'dagostino': dagostino_test,
        'ljung_box': ljung_box_test,
        # 'durbin_watson': durbin_watson_test,
        'levene': levene_test,
        'breusch_pagan': breusch_pagan_test,
        'adf': adf_test,
        'ks_normal': lambda x: kolmogorov_smirnov_test(x, 'norm')
    }
    
    # Select tests to run
    if test_names is None:
        tests_to_run = all_tests
    else:
        tests_to_run = {name: all_tests[name] for name in test_names if name in all_tests}
    
    # Run tests and collect results
    results = []
    for test_name, test_func in tests_to_run.items():
        try:
            test_result = test_func(data)
            
            # Different tests return different formats, handle accordingly
            if test_name == 'durbin_watson':
                results.append({
                    'Test': test_name,
                    'Statistic': test_result[0],
                    'Result': test_result[1]
                })
            else:
                results.append({
                    'Test': test_name,
                    'Statistic': test_result[0],
                    'p-value': test_result[1],
                    'Result': 'Pass' if test_result[2] else 'Fail'
                })
        except Exception as e:
            results.append({
                'Test': test_name,
                'Error': str(e)
            })
    
    return pd.DataFrame(results)


def plot_diagnostic_charts(data, figsize=(16, 14)):
    """
    Create a set of diagnostic charts for the data.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - figsize: Figure size.
    
    Returns:
    - fig: Figure object.
    """
    # Create a grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. Histogram with normal fit
    from scipy.stats import norm
    ax = axes[0, 0]
    mu, std = norm.fit(data)
    x = np.linspace(min(data), max(data), 100)
    p = norm.pdf(x, mu, std)
    
    ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue')
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title('Histogram with Normal Fit')
    ax.grid(True)
    
    # 2. Q-Q plot
    from scipy.stats import probplot
    ax = axes[0, 1]
    probplot(data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    ax.grid(True)
    
    # 3. Time series plot
    ax = axes[1, 0]
    ax.plot(data, linewidth=1)
    ax.set_title('Time Series Plot')
    ax.grid(True)
    
    # 4. Autocorrelation plot
    ax = axes[1, 1]
    plot_acf(data, ax=ax, lags=min(30, len(data)//2))
    ax.set_title('Autocorrelation Function')
    
    # 5. Partial autocorrelation plot
    ax = axes[2, 0]
    plot_pacf(data, ax=ax, lags=min(30, len(data)//2))
    ax.set_title('Partial Autocorrelation Function')
    
    # 6. Rolling statistics
    ax = axes[2, 1]
    window = max(20, len(data) // 10)
    rolling_mean = pd.Series(data).rolling(window=window).mean()
    rolling_std = pd.Series(data).rolling(window=window).std()
    
    ax.plot(data, label='Original', alpha=0.4)
    ax.plot(rolling_mean, label=f'Rolling Mean ({window})', linewidth=2)
    ax.plot(rolling_std, label=f'Rolling Std ({window})', linewidth=2)
    ax.set_title('Rolling Statistics')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def fit_arima_model(data, order=(1, 0, 1)):
    """
    Fit an ARIMA model to the data.
    
    Parameters:
    - data: 1D numpy array of time series data.
    - order: ARIMA order (p, d, q).
    
    Returns:
    - model: Fitted ARIMA model.
    - residuals: Model residuals.
    - summary: Model summary.
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    
    return fitted_model, fitted_model.resid, fitted_model.summary()


def test_model_residuals(model_residuals):
    """
    Run diagnostic tests on model residuals.
    
    Parameters:
    - model_residuals: 1D numpy array of model residuals.
    
    Returns:
    - residual_diagnostics: DataFrame with diagnostic test results.
    - fig: Figure with diagnostic plots.
    """
    # Run diagnostics on residuals
    residual_diagnostics = run_diagnostic_tests(model_residuals)
    
    # Create diagnostic plots
    fig = plot_diagnostic_charts(model_residuals)
    
    return residual_diagnostics, fig


def backtest_model(model_class, data, train_ratio=0.7, horizon=1, params=None, refit_window=10):
    """
    Backtest a model using rolling window forecasts.
    
    Parameters:
    - model_class: Model class to use.
    - data: 1D numpy array of time series data.
    - train_ratio: Ratio of data to use for initial training.
    - horizon: Forecast horizon.
    - params: Model parameters (if None, default parameters are used).
    - refit_window: Number of steps after which to refit the model.
    
    Returns:
    - forecasts: Array of forecasted values.
    - actuals: Array of actual values.
    - errors: Array of forecast errors.
    - error_metrics: Dictionary of error metrics.
    """
    n = len(data)
    train_size = int(train_ratio * n)
    
    # Initialize arrays for storing results
    forecasts = np.zeros(n - train_size)
    actuals = data[train_size:]
    
    # Initial model fitting
    train_data = data[:train_size]
    
    # Initialize model
    if params is None:
        model = model_class()
    else:
        model = model_class(**params)
    
    # Perform rolling window forecasting
    for i in range(n - train_size):
        if i % refit_window == 0:
            # Refit the model
            if i == 0:
                model_data = train_data
            else:
                model_data = data[:train_size + i]
            
            # Fit model (assuming a fit method)
            model.fit(model_data)
        
        # Forecast next value (assuming a forecast method)
        forecast = model.forecast(horizon)[0]
        forecasts[i] = forecast
    
    # Calculate forecast errors
    errors = actuals - forecasts
    
    # Calculate error metrics
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actuals)) * 100
    
    error_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return forecasts, actuals, errors, error_metrics


def plot_backtest_results(forecasts, actuals, figsize=(12, 8)):
    """
    Plot the results of a backtest.
    
    Parameters:
    - forecasts: Array of forecasted values.
    - actuals: Array of actual values.
    
    Returns:
    - fig: Figure object.
    """
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot actuals vs forecasts
    ax1.plot(actuals, label='Actual', alpha=0.7)
    ax1.plot(forecasts, label='Forecast', linestyle='--', alpha=0.7)
    ax1.set_title('Backtest: Actual vs Forecast')
    ax1.grid(True)
    ax1.legend()
    
    # Plot forecast errors
    errors = actuals - forecasts
    ax2.plot(errors, color='red', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Forecast Errors')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig