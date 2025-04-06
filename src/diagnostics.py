"""
diagnostics.py
--------------
Statistical diagnostic tests for model validation.
Includes:
- Jarque-Bera test
- Ljung-Box test
- Levene's test
"""

import numpy as np
from scipy.stats import jarque_bera, levene
from statsmodels.stats.diagnostic import acorr_ljungbox

def jarque_bera_test(data):
    """
    Perform the Jarque-Bera test for normality.
    
    Parameters:
    - data: 1D numpy array of sample data.
    
    Returns:
    - jb_stat: Test statistic.
    - p_value: p-value of the test.
    """
    jb_stat, p_value = jarque_bera(data)
    return jb_stat, p_value

def ljung_box_test(data, lags=10):
    """
    Perform the Ljung-Box test for autocorrelation.
    
    Parameters:
    - data: 1D numpy array of sample data.
    - lags: Number of lags to test (default 10).
    
    Returns:
    - lb_stat: Test statistic.
    - p_value: p-value of the test.
    """
    # The function returns a tuple of arrays for statistics and p-values.
    lb_test = acorr_ljungbox(data, lags=[lags], return_df=False)
    lb_stat = lb_test[0][0]
    p_value = lb_test[1][0]
    return lb_stat, p_value

def levene_test(data):
    """
    Perform Levene's test for equal variances.
    
    Parameters:
    - data: 1D numpy array of sample data.
    
    Returns:
    - levene_stat: Test statistic.
    - p_value: p-value of the test.
    """
    # For demonstration, we compare two halves of the data.
    n = len(data)
    group1 = data[:n//2]
    group2 = data[n//2:]
    levene_stat, p_value = levene(group1, group2)
    return levene_stat, p_value
