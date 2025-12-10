"""
Forecast evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_forecast_errors(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Compute various forecast error metrics.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
        
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary of error metrics
    """
    errors = actual - predicted
    
    metrics = {
        'ME': np.mean(errors),  # Mean Error
        'MAE': np.mean(np.abs(errors)),  # Mean Absolute Error
        'MSE': np.mean(errors**2),  # Mean Squared Error
        'RMSE': np.sqrt(np.mean(errors**2)),  # Root Mean Squared Error
        'MAPE': np.mean(np.abs(errors / actual)) * 100,  # Mean Absolute Percentage Error
        'MPE': np.mean(errors / actual) * 100,  # Mean Percentage Error
    }
    
    return metrics


def evaluate_forecasts(
    actual: np.ndarray,
    predicted: np.ndarray,
    prediction_intervals: Tuple[np.ndarray, np.ndarray] = None,
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive forecast evaluation.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Point forecasts
    prediction_intervals : Tuple[np.ndarray, np.ndarray], optional
        (lower, upper) bounds of prediction intervals
    alpha : float
        Significance level for intervals
        
    Returns:
    --------
    evaluation : Dict
        Dictionary with evaluation metrics
    """
    evaluation = {}
    
    # Point forecast metrics
    evaluation['point_metrics'] = compute_forecast_errors(actual, predicted)
    
    # Interval forecast metrics
    if prediction_intervals is not None:
        lower, upper = prediction_intervals
        
        # Coverage
        in_interval = (actual >= lower) & (actual <= upper)
        coverage = np.mean(in_interval)
        expected_coverage = 1 - alpha
        
        # Interval width
        mean_width = np.mean(upper - lower)
        
        # Interval score (Gneiting & Raftery, 2007)
        interval_scores = []
        for i in range(len(actual)):
            score = (upper[i] - lower[i])
            if actual[i] < lower[i]:
                score += 2 * (lower[i] - actual[i]) / alpha
            elif actual[i] > upper[i]:
                score += 2 * (actual[i] - upper[i]) / alpha
            interval_scores.append(score)
        
        evaluation['interval_metrics'] = {
            'coverage': coverage,
            'expected_coverage': expected_coverage,
            'mean_width': mean_width,
            'mean_interval_score': np.mean(interval_scores)
        }
    
    return evaluation


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    horizon: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for forecast comparison.
    
    Tests if two forecasts have equal accuracy.
    
    Parameters:
    -----------
    errors1 : np.ndarray
        Forecast errors from model 1
    errors2 : np.ndarray
        Forecast errors from model 2
    horizon : int
        Forecast horizon (for HAC standard errors)
        
    Returns:
    --------
    statistic : float
        DM test statistic
    p_value : float
        Two-sided p-value
    """
    from scipy import stats
    
    # Loss differential
    d = errors1**2 - errors2**2
    
    # Mean of loss differential
    d_mean = np.mean(d)
    
    # Variance (with HAC correction for multi-step forecasts)
    d_var = np.var(d, ddof=1) / len(d)
    
    # DM statistic
    dm_stat = d_mean / np.sqrt(d_var)
    
    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


def continuous_ranked_probability_score(
    actual: np.ndarray,
    forecast_samples: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values (n,)
    forecast_samples : np.ndarray
        Forecast samples (n x n_samples)
        
    Returns:
    --------
    crps : float
        Mean CRPS
    """
    n, n_samples = forecast_samples.shape
    crps_values = []
    
    for i in range(n):
        # Sort forecast samples
        samples = np.sort(forecast_samples[i])
        
        # Empirical CDF
        p_levels = np.arange(1, n_samples + 1) / n_samples
        
        # CRPS computation
        crps_i = 0.0
        for j in range(n_samples):
            if actual[i] <= samples[j]:
                crps_i += (p_levels[j] - 1)**2 * (samples[j] - (samples[j-1] if j > 0 else samples[j]))
            else:
                crps_i += p_levels[j]**2 * (samples[j] - (samples[j-1] if j > 0 else samples[j]))
        
        crps_values.append(crps_i)
    
    return np.mean(crps_values)


def log_score(
    actual: np.ndarray,
    forecast_samples: np.ndarray
) -> float:
    """
    Logarithmic score for density forecasts.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast_samples : np.ndarray
        Forecast samples for density estimation
        
    Returns:
    --------
    log_score : float
        Mean logarithmic score (higher is better)
    """
    from scipy.stats import gaussian_kde
    
    scores = []
    
    for i in range(len(actual)):
        # Estimate density
        kde = gaussian_kde(forecast_samples[i])
        
        # Evaluate at actual value
        density = kde(actual[i])[0]
        
        # Log score
        scores.append(np.log(density))
    
    return np.mean(scores)
