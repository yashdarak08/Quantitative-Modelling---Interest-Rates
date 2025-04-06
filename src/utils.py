"""
utils.py
--------
Utility functions for plotting and general helper routines.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_sample_paths(paths, num_paths=10, title="Sample Paths"):
    """
    Plot a subset of simulated paths.
    
    Parameters:
    - paths: 2D numpy array (n_paths x n_steps+1).
    - num_paths: Number of paths to plot.
    - title: Plot title.
    """
    n_paths = paths.shape[0]
    indices = np.linspace(0, n_paths - 1, num_paths, dtype=int)
    for idx in indices:
        plt.plot(paths[idx, :], lw=0.8, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel("Interest rate")
    plt.grid(True)
    plt.show()
