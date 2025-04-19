"""
main.py
--------
Entry point for the Quantitative Modelling project.
This script runs simulations, prices European swaptions using finite difference PDE solvers,
performs Monte Carlo risk analysis, and runs statistical diagnostics.

The project demonstrates a comprehensive framework for interest rate modeling,
derivatives pricing, and risk management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import modules from src
from src import models, pde_solver, monte_carlo, diagnostics, utils, backtesting, stress_testing

def setup_output_directory():
    """Create output directory for saving results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_interest_rate_models(output_dir):
    """
    Run interest rate model simulations and visualize results.
    
    Parameters:
    - output_dir: Directory to save output files.
    
    Returns:
    - Dictionary with simulation results for different models.
    """
    print("Running interest rate model simulations...")
    
    # Simulation parameters
    T = 1.0         # 1 year
    dt = 0.01       # Time step
    n_steps = int(T/dt)
    n_paths = 10000  # Increased for more precision
    
    # Vasicek model parameters
    vasicek_params = {'a': 0.1, 'b': 0.05, 'sigma': 0.01, 'r0': 0.03}
    vasicek_model = models.VasicekModel(**vasicek_params)
    vasicek_paths = vasicek_model.simulate(n_paths, n_steps, dt)
    
    # CIR model parameters
    cir_params = {'a': 0.15, 'b': 0.05, 'sigma': 0.02, 'r0': 0.03}
    cir_model = models.CIRModel(**cir_params)
    cir_paths = cir_model.simulate(n_paths, n_steps, dt)
    
    # Hull-White model parameters
    hw_params = {'a': 0.1, 'sigma': 0.015, 'r0': 0.03}
    hw_model = models.HullWhiteModel(**hw_params)
    hw_paths = hw_model.simulate(n_paths, n_steps, dt)
    
    # Plot sample paths with enhanced visualization
    fig1 = utils.plot_sample_paths(vasicek_paths, title="Vasicek Model Sample Paths")
    fig1.savefig(f"{output_dir}/vasicek_paths.png", dpi=300)
    
    fig2 = utils.plot_sample_paths(cir_paths, title="CIR Model Sample Paths")
    fig2.savefig(f"{output_dir}/cir_paths.png", dpi=300)
    
    fig3 = utils.plot_sample_paths(hw_paths, title="Hull-White Model Sample Paths")
    fig3.savefig(f"{output_dir}/hw_paths.png", dpi=300)
    
    # Plot terminal rate distributions
    fig4 = utils.plot_rate_distribution(vasicek_paths)
    fig4.savefig(f"{output_dir}/vasicek_distributions.png", dpi=300)
    
    fig5 = utils.plot_rate_distribution(cir_paths)
    fig5.savefig(f"{output_dir}/cir_distributions.png", dpi=300)
    
    # Compare models
    models_dict = {
        'Vasicek': vasicek_model,
        'CIR': cir_model,
        'Hull-White': hw_model
    }
    
    sim_params = {'n_paths': n_paths, 'n_steps': n_steps, 'dt': dt}
    fig6 = utils.compare_models(models_dict, sim_params, plot_type='paths')
    fig6.savefig(f"{output_dir}/model_comparison_paths.png", dpi=300)
    
    fig7 = utils.compare_models(models_dict, sim_params, plot_type='distribution')
    fig7.savefig(f"{output_dir}/model_comparison_distributions.png", dpi=300)
    
    fig8 = utils.compare_models(models_dict, sim_params, plot_type='stats')
    fig8.savefig(f"{output_dir}/model_comparison_stats.png", dpi=300)
    
    # Save results
    results = {
        'vasicek': {'model': vasicek_model, 'paths': vasicek_paths},
        'cir': {'model': cir_model, 'paths': cir_paths},
        'hull_white': {'model': hw_model, 'paths': hw_paths}
    }
    
    # Save simulation data
    utils.save_results_to_csv({'terminal_rates_vasicek': vasicek_paths[:, -1]}, 
                             f"{output_dir}/vasicek_terminal_rates.csv")
    utils.save_results_to_csv({'terminal_rates_cir': cir_paths[:, -1]}, 
                             f"{output_dir}/cir_terminal_rates.csv")
    
    print("Interest rate model simulations completed successfully.")
    return results

def run_pde_solver(output_dir):
    """
    Solve PDE for option pricing and visualize results.
    
    Parameters:
    - output_dir: Directory to save output files.
    
    Returns:
    - Dictionary with PDE solver results.
    """
    print("Running PDE solver for option pricing...")
    
    # Parameters for the PDE solver
    S_max = 0.15      # Maximum short rate
    r_steps = 200     # Grid steps in r dimension (increased for better accuracy)
    T = 1.0           # Maturity (in years)
    t_steps = 200     # Grid steps in time dimension (increased for better accuracy)
    
    # Interest rate model parameters
    a = 0.1
    b = 0.05
    sigma = 0.01
    r0 = 0.03
    
    # Option parameters for a European swaption
    strike = 0.04
    swap_maturity = 5.0
    swap_frequency = 0.5
    
    # Solve using different methods
    methods = ['explicit', 'implicit', 'crank-nicolson']
    prices = {}
    
    for method in methods:
        prices[method] = pde_solver.solve_european_swaption(
            S_max, r_steps, T, t_steps, r0, strike, 
            a=a, b=b, sigma=sigma, method=method,
            swap_maturity=swap_maturity, swap_frequency=swap_frequency
        )
        print(f"European Swaption Price ({method} method): {prices[method]:.6f}")
    
    # Compare pricing methods
    maturities = [0.25, 0.5, 1.0, 2.0, 3.0]
    comparison_df, fig9, _ = pde_solver.compare_pricing_methods(
        'swaption', r0, strike, maturities, methods=methods
    )
    fig9.savefig(f"{output_dir}/pde_method_comparison.png", dpi=300)
    
    # Save comparison results
    utils.save_results_to_csv(comparison_df, f"{output_dir}/pde_method_comparison.csv")
    
    # Plot option price surface for the Crank-Nicolson method
    solver_args = {
        'S_max': S_max,
        'r_steps': r_steps,
        'T': T,
        't_steps': t_steps,
        'r0': r0,
        'strike': strike,
        'a': a,
        'b': b,
        'sigma': sigma,
        'method': 'crank-nicolson',
        'swap_maturity': swap_maturity,
        'swap_frequency': swap_frequency
    }
    
    r_range = (0.01, 0.10)
    fig10 = pde_solver.plot_option_price_surface(
        pde_solver.solve_european_swaption, solver_args, r_range, 
        title="European Swaption Price Surface"
    )
    fig10.savefig(f"{output_dir}/swaption_price_surface.png", dpi=300)
    
    # Price a bond option as well
    bond_maturity = 5.0
    bond_option_price = pde_solver.solve_bond_option(
        S_max, r_steps, T, t_steps, r0, strike, bond_maturity,
        a=a, b=b, sigma=sigma, option_type='call', method='crank-nicolson'
    )
    print(f"Bond Option Price (Crank-Nicolson): {bond_option_price:.6f}")
    
    results = {
        'swaption_prices': prices,
        'bond_option_price': bond_option_price,
        'comparison': comparison_df
    }
    
    print("PDE solver option pricing completed successfully.")
    return results

def run_monte_carlo(model_results, output_dir):
    """
    Run Monte Carlo simulations for portfolio valuation and risk analysis.
    
    Parameters:
    - model_results: Dictionary with model simulation results.
    - output_dir: Directory to save output files.
    
    Returns:
    - Dictionary with Monte Carlo analysis results.
    """
    print("Running Monte Carlo risk analysis...")
    
    # Extract Vasicek paths for portfolio valuation
    vasicek_paths = model_results['vasicek']['paths']
    
    # Define portfolio configuration
    portfolio_config = {
        'maturities': [1.0, 2.0, 5.0, 10.0],
        'notionals': [1000000, 2000000, 1500000, 1000000],
        'fixed_rates': [0.03, 0.035, 0.04, 0.045],
        'is_receiver': [True, False, True, False]
    }
    
    # Simulate portfolio values
    portfolio_values = monte_carlo.simulate_portfolio(
        vasicek_paths, 
        portfolio_config['maturities'],
        portfolio_config['notionals'],
        portfolio_config['fixed_rates'],
        portfolio_config['is_receiver']
    )
    
    # Compute comprehensive risk metrics
    risk_metrics = monte_carlo.compute_risk_metrics(portfolio_values)
    
    # Print key risk metrics
    print(f"Portfolio Mean: ${risk_metrics['mean']*5500000:.2f}")
    print(f"Portfolio Std Dev: ${risk_metrics['std_dev']*5500000:.2f}")
    print(f"VaR (95%): ${risk_metrics['VaR_95']*5500000:.2f}")
    print(f"CVaR (95%): ${risk_metrics['CVaR_95']*5500000:.2f}")
    print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.4f}")
    
    # Run stress tests on the portfolio
    base_params = model_results['vasicek']['model'].calibrated_params
    
    # Define stress test scenarios
    scenarios = {
        'Rate_Hike': {'r0': 0.05, 'b': 0.06},
        'Rate_Cut': {'r0': 0.01, 'b': 0.02},
        'High_Volatility': {'sigma': 0.03},
        'Stagflation': {'r0': 0.07, 'b': 0.08, 'sigma': 0.03}
    }
    
    # Run stress tests
    stress_results = monte_carlo.run_stress_test(
        models.VasicekModel,
        scenarios,
        portfolio_config
    )
    
    # Plot stress test results
    fig11 = monte_carlo.plot_stress_test_results(stress_results)
    fig11.savefig(f"{output_dir}/stress_test_results.png", dpi=300)
    
    # Save stress test results
    utils.save_results_to_csv(stress_results, f"{output_dir}/stress_test_results.csv")
    
    # Perform sensitivity analysis
    param_ranges = {
        'a': [0.05, 0.1, 0.15, 0.2, 0.3],
        'b': [0.02, 0.03, 0.04, 0.05, 0.06],
        'sigma': [0.005, 0.01, 0.015, 0.02, 0.03]
    }
    
    # Run sensitivity analysis
    sensitivity_results = monte_carlo.sensitivity_analysis(
        model_results['vasicek']['model'],
        param_ranges,
        portfolio_config
    )
    
    # Plot sensitivity analysis results
    fig12 = monte_carlo.plot_sensitivity_analysis(sensitivity_results)
    fig12.savefig(f"{output_dir}/sensitivity_analysis.png", dpi=300)
    
    # Save sensitivity analysis results
    utils.save_results_to_csv(sensitivity_results, f"{output_dir}/sensitivity_analysis.csv")
    
    # Monte Carlo option pricing
    option_types = ['call', 'put', 'swaption_receiver', 'swaption_payer']
    option_strikes = [0.03, 0.04, 0.05]
    option_maturities = [0.5, 1.0, 2.0]
    
    # Create a DataFrame to store option prices
    option_prices = []
    
    for option_type in option_types:
        for strike in option_strikes:
            for maturity in option_maturities:
                price, std_error, _ = monte_carlo.monte_carlo_option_pricing(
                    models.VasicekModel,
                    option_type,
                    strike,
                    maturity,
                    r0=0.03,
                    a=0.1,
                    b=0.05,
                    sigma=0.01,
                    n_paths=10000
                )
                
                option_prices.append({
                    'Option_Type': option_type,
                    'Strike': strike,
                    'Maturity': maturity,
                    'Price': price,
                    'Std_Error': std_error
                })
    
    # Convert to DataFrame
    option_prices_df = pd.DataFrame(option_prices)
    
    # Save option prices
    utils.save_results_to_csv(option_prices_df, f"{output_dir}/monte_carlo_option_prices.csv")
    
    # Compile results
    results = {
        'portfolio_values': portfolio_values,
        'risk_metrics': risk_metrics,
        'stress_results': stress_results,
        'sensitivity_results': sensitivity_results,
        'option_prices': option_prices_df
    }
    
    print("Monte Carlo risk analysis completed successfully.")
    return results

def run_diagnostics(monte_carlo_results, output_dir):
    """
    Run diagnostic tests on Monte Carlo results.
    
    Parameters:
    - monte_carlo_results: Dictionary with Monte Carlo analysis results.
    - output_dir: Directory to save output files.
    
    Returns:
    - Dictionary with diagnostic test results.
    """
    print("Running statistical diagnostics...")
    
    # Extract portfolio values
    portfolio_values = monte_carlo_results['portfolio_values']
    
    # Run comprehensive diagnostic tests
    diagnostic_results = diagnostics.run_diagnostic_tests(portfolio_values)
    
    # Print diagnostic test summary
    print("\nDiagnostic Test Summary:")
    print(diagnostic_results)
    
    # Save diagnostic results
    utils.save_results_to_csv(diagnostic_results, f"{output_dir}/diagnostic_results.csv")
    
    # Create diagnostic plots
    fig13 = diagnostics.plot_diagnostic_charts(portfolio_values)
    fig13.savefig(f"{output_dir}/diagnostic_charts.png", dpi=300)
    
    # Fit ARIMA model to portfolio values
    fitted_model, residuals, summary = diagnostics.fit_arima_model(portfolio_values, order=(1, 0, 1))
    
    # Save model summary as text
    with open(f"{output_dir}/arima_model_summary.txt", 'w') as f:
        f.write(str(summary))
    
    # Test residuals
    residual_diagnostics, fig14 = diagnostics.test_model_residuals(residuals)
    fig14.savefig(f"{output_dir}/residual_diagnostics.png", dpi=300)
    
    # Save residual diagnostics
    utils.save_results_to_csv(residual_diagnostics, f"{output_dir}/residual_diagnostics.csv")
    
    # Compile results
    results = {
        'diagnostic_tests': diagnostic_results,
        'arima_model': fitted_model,
        'residuals': residuals,
        'residual_diagnostics': residual_diagnostics
    }
    
    print("Statistical diagnostics completed successfully.")
    return results

def run_backtesting(output_dir):
    """
    Run backtesting of interest rate models.
    
    Parameters:
    - output_dir: Directory to save output files.
    
    Returns:
    - Dictionary with backtesting results.
    """
    print("Running backtesting framework...")
    
    # Load or generate historical interest rate data
    rate_data = backtesting.load_interest_rate_data()
    
    # Split into training, validation, and test sets
    train_data, validation_data, test_data = backtesting.split_data(rate_data)
    
    # Create models to test
    models_to_test = {
        'Vasicek': {
            'class': models.VasicekModel,
            'initial_params': {'a': 0.1, 'b': 0.04, 'sigma': 0.01, 'r0': train_data.iloc[-1, 0]}
        },
        'CIR': {
            'class': models.CIRModel,
            'initial_params': {'a': 0.15, 'b': 0.04, 'sigma': 0.02, 'r0': train_data.iloc[-1, 0]}
        }
    }
    
    # Calibrate models
    calibrated_models = {}
    for model_name, model_info in models_to_test.items():
        print(f"Calibrating {model_name} model...")
        calibrated_model, _ = backtesting.calibrate_model(
            model_info['class'],
            train_data,
            model_info['initial_params']
        )
        calibrated_models[model_name] = calibrated_model
    
    # Evaluate calibration on validation data
    evaluation_metrics = {}
    for model_name, model in calibrated_models.items():
        print(f"Evaluating {model_name} model...")
        metrics = backtesting.evaluate_calibration(model, validation_data)
        evaluation_metrics[model_name] = metrics
    
    # Compare models
    comparison_df = backtesting.compare_models(calibrated_models, validation_data)
    
    # Plot comparison results
    fig15 = backtesting.plot_model_comparison(comparison_df)
    fig15.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    
    # Plot forecast evaluation for the best model
    best_model_name = comparison_df.sort_values('Overall_Rank').iloc[0]['Model']
    best_model_metrics = evaluation_metrics[best_model_name]
    
    fig16 = backtesting.plot_forecast_evaluation(
        best_model_metrics,
        validation_data.iloc[:, 0].values
    )
    fig16.savefig(f"{output_dir}/forecast_evaluation.png", dpi=300)
    
    # Visualize yield curve from best model
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    fig17 = backtesting.visualize_yields_and_forward_rates(
        calibrated_models[best_model_name],
        maturities,
        validation_data.iloc[0, 0]
    )
    fig17.savefig(f"{output_dir}/yield_curve.png", dpi=300)
    
    # Save results
    utils.save_results_to_csv(comparison_df, f"{output_dir}/model_comparison.csv")
    
    # Compile results
    results = {
        'calibrated_models': calibrated_models,
        'evaluation_metrics': evaluation_metrics,
        'comparison': comparison_df,
        'best_model': best_model_name
    }
    
    print("Backtesting framework completed successfully.")
    return results

def run_stress_testing(backtesting_results, output_dir):
    """
    Run stress testing framework.
    
    Parameters:
    - backtesting_results: Dictionary with backtesting results.
    - output_dir: Directory to save output files.
    
    Returns:
    - Dictionary with stress testing results.
    """
    print("Running stress testing framework...")
    
    # Get the best model from backtesting
    best_model_name = backtesting_results['best_model']
    best_model = backtesting_results['calibrated_models'][best_model_name]
    
    # Define stress scenarios
    scenarios = stress_testing.define_stress_scenarios()
    
    # Run scenario simulations
    scenario_results = stress_testing.simulate_all_scenarios(best_model)
    
    # Define portfolio valuator function
    def portfolio_valuator(paths):
        # Simple portfolio of bonds with different maturities
        maturities = [1.0, 2.0, 5.0, 10.0]
        notionals = [1000000, 2000000, 1500000, 1000000]
        fixed_rates = [0.03, 0.035, 0.04, 0.045]
        is_receiver = [True, False, True, False]
        
        return monte_carlo.simulate_portfolio(
            paths, maturities, notionals, fixed_rates, is_receiver
        )
    
    # Calculate portfolio impact
    portfolio_results = stress_testing.calculate_portfolio_impact(
        scenario_results, portfolio_valuator
    )
    
    # Perform tail risk analysis
    tail_risk_df = stress_testing.tail_risk_analysis(portfolio_results)
    
    # Create visualizations
    fig18 = stress_testing.plot_scenario_paths(scenario_results)
    fig18.savefig(f"{output_dir}/stress_scenario_paths.png", dpi=300)
    
    fig19 = stress_testing.plot_terminal_distributions(scenario_results)
    fig19.savefig(f"{output_dir}/stress_terminal_distributions.png", dpi=300)
    
    fig20 = stress_testing.plot_portfolio_impact(portfolio_results)
    fig20.savefig(f"{output_dir}/stress_portfolio_impact.png", dpi=300)
    
    fig21 = stress_testing.plot_tail_risk_comparison(tail_risk_df)
    fig21.savefig(f"{output_dir}/stress_tail_risk.png", dpi=300)
    
    # Create stress testing report
    report = stress_testing.create_stress_testing_report(
        best_model, scenario_results, portfolio_results, tail_risk_df
    )
    
    # Save results
    utils.save_results_to_csv(report['scenario_summary'], f"{output_dir}/stress_scenario_summary.csv")
    utils.save_results_to_csv(report['portfolio_summary'], f"{output_dir}/stress_portfolio_summary.csv")
    utils.save_results_to_csv(tail_risk_df, f"{output_dir}/stress_tail_risk.csv")
    
    # Perform sensitivity analysis
    parameter_ranges = {
        'a': [0.05, 0.1, 0.15, 0.2, 0.3],
        'b': [0.02, 0.03, 0.04, 0.05, 0.06],
        'sigma': [0.005, 0.01, 0.015, 0.02, 0.03]
    }
    
    sensitivity_results = stress_testing.sensitivity_analysis(best_model, parameter_ranges)
    
    fig22 = stress_testing.plot_sensitivity_results(sensitivity_results)
    fig22.savefig(f"{output_dir}/stress_sensitivity.png", dpi=300)
    
    # Compile results
    results = {
        'model': best_model,
        'scenario_results': scenario_results,
        'portfolio_results': portfolio_results,
        'tail_risk': tail_risk_df,
        'report': report,
        'sensitivity_results': sensitivity_results
    }
    
    print("Stress testing framework completed successfully.")
    return results

def create_dashboard(all_results, output_dir):
    """
    Create a comprehensive dashboard of all results.
    
    Parameters:
    - all_results: Dictionary with all analysis results.
    - output_dir: Directory to save output files.
    
    Returns:
    - Dashboard figure.
    """
    print("Creating results dashboard...")
    
    # Extract key components for the dashboard
    model_results = {
        'paths': all_results['model_results']['vasicek']['paths']
    }
    
    portfolio_results = {
        'portfolio_values': all_results['monte_carlo_results']['portfolio_values'],
        'risk_metrics': all_results['monte_carlo_results']['risk_metrics']
    }
    
    diagnostics_results = {
        'test_results': {row['Test']: (row['Statistic'], row.get('p-value', 0)) 
                        for _, row in all_results['diagnostics_results']['diagnostic_tests'].iterrows()
                        if 'Error' not in row},
        'residuals': all_results['diagnostics_results']['residuals']
    }
    
    # Create dashboard
    dashboard = utils.create_dashboard(model_results, portfolio_results, diagnostics_results)
    
    # Save dashboard
    dashboard.savefig(f"{output_dir}/dashboard.png", dpi=300)
    
    print("Dashboard created successfully.")
    return dashboard

def main():
    """Main function to run the entire analysis pipeline."""
    print("Starting Quantitative Modelling project...")
    
    # Create output directory
    output_dir = setup_output_directory()
    print(f"Output will be saved to: {output_dir}")
    
    # Run components
    model_results = run_interest_rate_models(output_dir)
    pde_results = run_pde_solver(output_dir)
    monte_carlo_results = run_monte_carlo(model_results, output_dir)
    diagnostics_results = run_diagnostics(monte_carlo_results, output_dir)
    backtesting_results = run_backtesting(output_dir)
    stress_testing_results = run_stress_testing(backtesting_results, output_dir)
    
    # Compile all results
    all_results = {
        'model_results': model_results,
        'pde_results': pde_results,
        'monte_carlo_results': monte_carlo_results,
        'diagnostics_results': diagnostics_results,
        'backtesting_results': backtesting_results,
        'stress_testing_results': stress_testing_results
    }
    
    # Create dashboard
    dashboard = create_dashboard(all_results, output_dir)
    
    print("\nQuantitative Modelling project completed successfully.")
    print(f"All results saved to: {output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()
    plt.show()  # Show all figures at the end