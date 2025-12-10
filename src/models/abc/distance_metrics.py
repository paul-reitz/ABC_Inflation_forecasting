"""
Distance metrics and summary statistics for ABC.
"""

import numpy as np
from typing import Callable, List, Dict


def summary_statistics(data: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics for time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
        
    Returns:
    --------
    stats : np.ndarray
        Vector of summary statistics
    """
    stats = []
    
    # Mean
    stats.append(np.mean(data))
    
    # Variance
    stats.append(np.var(data))
    
    # Autocorrelations at lags 1, 2, 3, 6, 12
    for lag in [1, 2, 3, 6, 12]:
        if len(data) > lag:
            autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            stats.append(autocorr)
        else:
            stats.append(0.0)
    
    # Skewness
    from scipy.stats import skew
    stats.append(skew(data))
    
    # Kurtosis
    from scipy.stats import kurtosis
    stats.append(kurtosis(data))
    
    # Quantiles
    for q in [0.25, 0.5, 0.75]:
        stats.append(np.quantile(data, q))
    
    return np.array(stats)


def compute_distance(
    summary_obs: np.ndarray,
    summary_sim: np.ndarray,
    metric: str = 'euclidean',
    weights: np.ndarray = None
) -> float:
    """
    Compute distance between observed and simulated summary statistics.
    
    Parameters:
    -----------
    summary_obs : np.ndarray
        Summary statistics from observed data
    summary_sim : np.ndarray
        Summary statistics from simulated data
    metric : str
        Distance metric ('euclidean', 'manhattan', 'weighted_euclidean')
    weights : np.ndarray, optional
        Weights for weighted distance metrics
        
    Returns:
    --------
    distance : float
        Distance between summaries
    """
    if metric == 'euclidean':
        return np.linalg.norm(summary_obs - summary_sim)
    
    elif metric == 'manhattan':
        return np.sum(np.abs(summary_obs - summary_sim))
    
    elif metric == 'weighted_euclidean':
        if weights is None:
            weights = np.ones_like(summary_obs)
        return np.sqrt(np.sum(weights * (summary_obs - summary_sim)**2))
    
    elif metric == 'mahalanobis':
        # Requires covariance matrix estimation
        diff = summary_obs - summary_sim
        # Simplified version - would need proper covariance estimation
        return np.sqrt(np.sum(diff**2))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def autocovariance_based_distance(
    data_obs: np.ndarray,
    data_sim: np.ndarray,
    max_lag: int = 12
) -> float:
    """
    Distance based on autocovariance function.
    
    Useful for time series comparison.
    
    Parameters:
    -----------
    data_obs : np.ndarray
        Observed data
    data_sim : np.ndarray
        Simulated data
    max_lag : int
        Maximum lag for autocovariance
        
    Returns:
    --------
    distance : float
    """
    from statsmodels.tsa.stattools import acovf
    
    acov_obs = acovf(data_obs, nlag=max_lag, fft=False)
    acov_sim = acovf(data_sim, nlag=max_lag, fft=False)
    
    return np.linalg.norm(acov_obs - acov_sim)


def spectral_distance(
    data_obs: np.ndarray,
    data_sim: np.ndarray
) -> float:
    """
    Distance based on spectral density.
    
    Parameters:
    -----------
    data_obs : np.ndarray
        Observed data
    data_sim : np.ndarray
        Simulated data
        
    Returns:
    --------
    distance : float
    """
    from scipy.signal import periodogram
    
    # Compute periodograms
    _, psd_obs = periodogram(data_obs)
    _, psd_sim = periodogram(data_sim)
    
    # Ensure same length
    min_len = min(len(psd_obs), len(psd_sim))
    
    return np.linalg.norm(np.log(psd_obs[:min_len] + 1e-10) - 
                         np.log(psd_sim[:min_len] + 1e-10))
