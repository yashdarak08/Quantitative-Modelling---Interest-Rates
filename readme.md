# Quantitative Modelling Framework

A comprehensive framework for interest rate modeling, derivatives pricing, and risk management.

## Key Features

- **Interest Rate Models:**
  - **Vasicek Model**: Mean-reverting with constant volatility
  - **Cox-Ingersoll-Ross (CIR) Model**: Mean-reverting with rate-dependent volatility
  - **Hull-White Model**: Extended Vasicek with time-dependent parameters

- **Finite Difference PDE Solver:**
  - Multiple numerical schemes (Explicit, Implicit, Crank-Nicolson)
  - European swaption pricing
  - Bond option pricing
  - Price surface visualization

- **Monte Carlo Simulation:**
  - 10,000+ path simulations
  - Portfolio valuation under stochastic rates
  - VaR and CVaR computation
  - Advanced risk metrics (Sharpe ratio, Sortino ratio)

- **Statistical Diagnostics:**
  - Normality tests (Jarque-Bera, Shapiro-Wilk, D'Agostino)
  - Autocorrelation tests (Ljung-Box, Durbin-Watson)
  - Heteroskedasticity tests (Levene, Breusch-Pagan)
  - ARIMA model fitting and residual analysis

- **Backtesting Framework:**
  - Model calibration to historical data
  - Out-of-sample validation
  - Forecast evaluation
  - Model comparison

- **Stress Testing:**
  - Pre-defined stress scenarios (rate shocks, volatility changes)
  - Portfolio impact analysis
  - Tail risk evaluation
  - Sensitivity analysis

## Repository Structure

```
Quantitative-Modelling/
├── LICENSE
├── README.md
├── requirements.txt
├── main.py
└── src/
    ├── __init__.py
    ├── models.py
    ├── pde_solver.py
    ├── monte_carlo.py
    ├── diagnostics.py
    ├── utils.py
    ├── backtesting.py
    └── stress_testing.py
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Quantitative-Modelling.git
   cd Quantitative-Modelling
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to execute the full analysis pipeline:

```bash
python main.py
```

This will:
1. Run interest rate model simulations
2. Solve PDEs for option pricing
3. Perform Monte Carlo risk analysis
4. Run statistical diagnostics
5. Execute backtesting framework
6. Conduct stress testing
7. Generate a comprehensive dashboard with results

All outputs (figures, CSV files, reports) will be saved to a timestamped output directory.

## Module Details

### models.py

Implements stochastic interest rate models:
- Vasicek Model with mean reversion
- CIR Model with square-root diffusion
- Hull-White Model with time-dependent parameters
- Analytical zero-coupon bond pricing
- Model calibration methods

### pde_solver.py

Implements finite difference methods for option pricing:
- Explicit scheme
- Implicit scheme (using sparse matrices)
- Crank-Nicolson scheme
- European swaption pricing
- Bond option pricing
- Price surface visualization

### monte_carlo.py

Implements Monte Carlo simulation for risk analysis:
- Portfolio valuation under stochastic rates
- Comprehensive risk metrics computation
- Stress testing framework
- Sensitivity analysis
- Option pricing via simulation

### diagnostics.py

Implements statistical tests for model validation:
- Normality tests
- Autocorrelation tests
- Heteroskedasticity tests
- ARIMA model fitting
- Residual analysis
- Backtesting framework

### utils.py

Provides utility functions:
- Enhanced data visualization
- Yield curve and forward rate calculation
- Model comparison tools
- Dashboard creation
- Results export/import

### backtesting.py

Implements model validation against historical data:
- Parameter calibration
- Out-of-sample testing
- Forecast evaluation
- Model comparison

### stress_testing.py

Implements stress testing framework:
- Scenario generation
- Portfolio impact analysis
- Tail risk evaluation
- Sensitivity analysis

## Example Outputs

The framework generates numerous visualizations, including:

- Interest rate model simulations
- Option price surfaces
- Risk metric dashboards
- Diagnostic plots
- Backtesting results
- Stress test scenarios
- Sensitivity analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.