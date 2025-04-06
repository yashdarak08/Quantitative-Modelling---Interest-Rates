# Quantitative Modelling

- **Repository Structure:**
```bash
Quantitative-Modelling/
├── LICENSE
├── README.md
├── requirements.txt
├── main.py
└── src
    ├── __init__.py
    ├── diagnostics.py
    ├── monte_carlo.py
    ├── models.py
    ├── pde_solver.py
    └── utils.py
```


This repository contains a quantitative framework for modeling interest rate derivatives. It implements:

- **Interest Rate Models:**  
  - **Vasicek Model**
  - **CIR Model**

- **Finite Difference PDE Solver:**  
  A numerical solver (using finite difference methods) to price European swaptions under stochastic short rate models.

- **Monte Carlo Simulation:**  
  Runs 10,000+ simulations to compute Value at Risk (VaR) and Conditional VaR (CVaR), stress test exposures, and identify tail risks under interest rate shocks.

- **Statistical Diagnostics:**  
  Validates model assumptions using statistical tests:
  - Jarque-Bera
  - Ljung-Box
  - Levene’s test

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Quantitative-Modelling.git
   cd Quantitative-Modelling
    ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```