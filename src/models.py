"""
models.py
---------
This module implements stochastic interest rate models for simulating rate dynamics.
Includes:
- Vasicek Model: Mean-reverting with constant volatility
- Cox-Ingersoll-Ross (CIR) Model: Mean-reverting with rate-dependent volatility
- Hull-White Model: Extended Vasicek with time-dependent parameters
- Zero-Coupon Bond pricing under each model
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class VasicekModel:
    def __init__(self, a, b, sigma, r0):
        """
        Vasicek model initialization.
        
        Parameters:
        - a: Speed of mean reversion.
        - b: Long-term mean level.
        - sigma: Volatility.
        - r0: Initial interest rate.
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0
        
        # Validate Feller condition (not applicable to Vasicek, but keeping interface consistent)
        self.is_feller_satisfied = True

    def simulate(self, n_paths, n_steps, dt):
        """
        Simulate short rate paths using the Vasicek model.
        
        Parameters:
        - n_paths: Number of paths to simulate.
        - n_steps: Number of time steps.
        - dt: Time step size.
        
        Returns:
        - rates: Array of simulated paths of shape (n_paths, n_steps+1).
        """
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0
        
        for i in range(1, n_steps + 1):
            # Analytical discretization for more accuracy
            exp_factor = np.exp(-self.a * dt)
            mean = self.b + (rates[:, i-1] - self.b) * exp_factor
            var = (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * dt))
            rates[:, i] = mean + np.sqrt(var) * np.random.randn(n_paths)
        
        return rates
    
    def analytical_zcb_price(self, r, T):
        """
        Calculate the zero-coupon bond price analytically under the Vasicek model.
        
        Parameters:
        - r: Current short rate.
        - T: Time to maturity of the bond.
        
        Returns:
        - price: Zero-coupon bond price.
        """
        B = (1 - np.exp(-self.a * T)) / self.a
        A = np.exp((self.b - (self.sigma**2) / (2 * self.a**2)) * (B - T) - 
                  (self.sigma**2) * (B**2) / (4 * self.a))
        
        return A * np.exp(-B * r)
    
    def calibrate(self, maturities, market_prices, initial_params=None):
        """
        Calibrate model parameters to match market zero-coupon bond prices.
        
        Parameters:
        - maturities: Array of bond maturities.
        - market_prices: Array of market zero-coupon bond prices.
        - initial_params: Initial guess for parameters [a, b, sigma].
        
        Returns:
        - Calibrated parameters [a, b, sigma].
        """
        r0 = self.r0
        
        if initial_params is None:
            initial_params = [self.a, self.b, self.sigma]
        
        def objective_function(params):
            a, b, sigma = params
            temp_model = VasicekModel(a, b, sigma, r0)
            
            model_prices = np.array([temp_model.analytical_zcb_price(r0, t) for t in maturities])
            return np.sum((model_prices - market_prices)**2)
        
        bounds = [(0.001, 1.0), (0.001, 0.20), (0.001, 0.20)]
        result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Update model parameters with calibrated values
        self.a, self.b, self.sigma = result.x
        return result.x


class CIRModel:
    def __init__(self, a, b, sigma, r0):
        """
        Cox-Ingersoll-Ross (CIR) model initialization.
        
        Parameters:
        - a: Speed of mean reversion.
        - b: Long-term mean level.
        - sigma: Volatility.
        - r0: Initial interest rate.
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0
        
        # Check Feller condition (2*a*b > sigma^2) for strictly positive rates
        self.is_feller_satisfied = (2 * self.a * self.b > self.sigma**2)
        if not self.is_feller_satisfied:
            print("Warning: Feller condition not satisfied. Rates may hit zero.")

    def simulate(self, n_paths, n_steps, dt):
        """
        Simulate the CIR model with square-root diffusion.
        
        Parameters:
        - n_paths: Number of paths to simulate.
        - n_steps: Number of time steps.
        - dt: Time step size.
        
        Returns:
        - rates: Array of simulated paths of shape (n_paths, n_steps+1).
        """
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0
        
        # Check if we need a non-central chi-square sampling approach
        use_central_approx = self.is_feller_satisfied and self.a * self.b > 5 * self.sigma**2
        
        for i in range(1, n_steps + 1):
            if use_central_approx:
                # Approximate with normal distribution when far from zero boundary
                sqrt_r = np.sqrt(np.maximum(rates[:, i-1], 0))
                dr = self.a * (self.b - rates[:, i-1]) * dt + self.sigma * sqrt_r * np.sqrt(dt) * np.random.randn(n_paths)
                rates[:, i] = np.maximum(rates[:, i-1] + dr, 0)
            else:
                # More accurate simulation using non-central chi-square distribution approximation
                df = 4 * self.a * self.b / self.sigma**2
                ncp = 4 * self.a * np.exp(-self.a * dt) / (self.sigma**2 * (1 - np.exp(-self.a * dt))) * rates[:, i-1]
                rates[:, i] = self.sigma**2 * (1 - np.exp(-self.a * dt)) / (4 * self.a) * np.random.noncentral_chisquare(df, ncp, n_paths)
        
        return rates
    
    def analytical_zcb_price(self, r, T):
        """
        Calculate the zero-coupon bond price analytically under the CIR model.
        
        Parameters:
        - r: Current short rate.
        - T: Time to maturity of the bond.
        
        Returns:
        - price: Zero-coupon bond price.
        """
        gamma = np.sqrt(self.a**2 + 2 * self.sigma**2)
        denominator = 2 * gamma + (self.a + gamma) * (np.exp(gamma * T) - 1)
        B = 2 * (np.exp(gamma * T) - 1) / denominator
        A = (2 * gamma * np.exp((self.a + gamma) * T / 2) / denominator) ** (2 * self.a * self.b / self.sigma**2)
        
        return A * np.exp(-B * r)
    
    def calibrate(self, maturities, market_prices, initial_params=None):
        """
        Calibrate model parameters to match market zero-coupon bond prices.
        
        Parameters:
        - maturities: Array of bond maturities.
        - market_prices: Array of market zero-coupon bond prices.
        - initial_params: Initial guess for parameters [a, b, sigma].
        
        Returns:
        - Calibrated parameters [a, b, sigma].
        """
        r0 = self.r0
        
        if initial_params is None:
            initial_params = [self.a, self.b, self.sigma]
        
        def objective_function(params):
            a, b, sigma = params
            temp_model = CIRModel(a, b, sigma, r0)
            
            model_prices = np.array([temp_model.analytical_zcb_price(r0, t) for t in maturities])
            return np.sum((model_prices - market_prices)**2)
        
        # Ensure Feller condition as a constraint
        def feller_constraint(params):
            a, b, sigma = params
            return 2 * a * b - sigma**2
        
        bounds = [(0.001, 1.0), (0.001, 0.20), (0.001, 0.20)]
        constraints = {'type': 'ineq', 'fun': feller_constraint}
        result = minimize(objective_function, initial_params, bounds=bounds, constraints=constraints, method='SLSQP')
        
        # Update model parameters with calibrated values
        self.a, self.b, self.sigma = result.x
        # Update Feller condition status
        self.is_feller_satisfied = (2 * self.a * self.b > self.sigma**2)
        
        return result.x


class HullWhiteModel:
    def __init__(self, a, sigma, r0, theta_t=None):
        """
        Hull-White (extended Vasicek) model initialization.
        
        Parameters:
        - a: Speed of mean reversion.
        - sigma: Volatility.
        - r0: Initial interest rate.
        - theta_t: Function for time-dependent drift adjustment (default: constant zero).
        """
        self.a = a
        self.sigma = sigma
        self.r0 = r0
        self.theta_t = theta_t if theta_t is not None else lambda t: 0.0

    def simulate(self, n_paths, n_steps, dt):
        """
        Simulate the Hull-White model with time-dependent drift.
        
        Parameters:
        - n_paths: Number of paths to simulate.
        - n_steps: Number of time steps.
        - dt: Time step size.
        
        Returns:
        - rates: Array of simulated paths of shape (n_paths, n_steps+1).
        """
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0
        
        for i in range(1, n_steps + 1):
            t = i * dt
            theta = self.theta_t(t)
            
            # Exact simulation scheme
            exp_factor = np.exp(-self.a * dt)
            mean = rates[:, i-1] * exp_factor + theta / self.a * (1 - exp_factor)
            var = (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * dt))
            
            rates[:, i] = mean + np.sqrt(var) * np.random.randn(n_paths)
        
        return rates
    
    def calibrate_to_yield_curve(self, maturities, zero_rates):
        """
        Calibrate the Hull-White model to a given yield curve.
        
        Parameters:
        - maturities: Array of maturities.
        - zero_rates: Array of zero rates corresponding to the maturities.
        
        Returns:
        - theta_t: Calibrated time-dependent drift function.
        """
        # Implement a simple piecewise linear interpolation for theta
        times = np.linspace(0, max(maturities), 100)
        thetas = np.zeros_like(times)
        
        # Numerical differentiation of the yield curve
        for i, t in enumerate(times):
            if i == 0:
                continue
                
            # Find closest zero rate
            idx = np.abs(maturities - t).argmin()
            r_t = zero_rates[idx]
            
            # Forward rate approximation (f = -d/dt ln P(t))
            if idx > 0:
                dt_forward = maturities[idx] - maturities[idx-1]
                forward_rate = (zero_rates[idx] * maturities[idx] - zero_rates[idx-1] * maturities[idx-1]) / dt_forward
            else:
                forward_rate = r_t
            
            # Set theta based on the formula: theta(t) = d/dt f(0,t) + a*f(0,t) + sigma^2/(2*a)*(1-exp(-2*a*t))
            if i > 1:
                dfdt = (forward_rate - prev_forward) / (t - times[i-1])
            else:
                dfdt = 0
                
            thetas[i] = dfdt + self.a * forward_rate + \
                       (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * t))
            
            prev_forward = forward_rate
        
        # Create interpolation function for theta
        from scipy.interpolate import interp1d
        theta_func = interp1d(times, thetas, bounds_error=False, fill_value="extrapolate")
        self.theta_t = theta_func
        
        return theta_func