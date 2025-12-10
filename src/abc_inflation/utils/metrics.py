"""
Forecast Evaluation Metrics

This module provides metrics for evaluating and comparing inflation forecasts.
"""

import numpy as np
from typing import Dict


def forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast evaluation metrics.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted values
    model_name : str, optional
        Name of the model for reporting
        
    Returns
    -------
    dict
        Dictionary containing various metrics
    """
    # Ensure arrays are the same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-10)) * 100
    
    # Mean Error (Bias)
    me = np.mean(actual - predicted)
    
    # Theil's U Statistic
    if len(actual) > 1:
        naive_forecast = np.roll(actual, 1)[1:]
        actual_subset = actual[1:]
        predicted_subset = predicted[1:]
        
        numerator = np.sqrt(np.mean((actual_subset - predicted_subset) ** 2))
        denominator = np.sqrt(np.mean((actual_subset - naive_forecast) ** 2))
        theil_u = numerator / (denominator + 1e-10)
    else:
        theil_u = np.nan
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-10))
    
    return {
        'model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'ME': me,
        'Theil_U': theil_u,
        'R2': r_squared,
    }


def compare_forecasts(
    actual: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple forecasts against actual values.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    forecasts_dict : dict
        Dictionary mapping model names to their forecasts
        
    Returns
    -------
    dict
        Dictionary of metrics for each model
    """
    results = {}
    
    for model_name, forecast in forecasts_dict.items():
        results[model_name] = forecast_metrics(actual, forecast, model_name)
    
    return results


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1
) -> Dict[str, float]:
    """
    Diebold-Mariano test for forecast comparison.
    
    Tests whether two forecasts have equal predictive accuracy.
    
    Parameters
    ----------
    errors1 : np.ndarray
        Forecast errors from model 1
    errors2 : np.ndarray
        Forecast errors from model 2
    h : int, optional
        Forecast horizon
        
    Returns
    -------
    dict
        Test statistic and p-value
    """
    # Ensure same length
    min_len = min(len(errors1), len(errors2))
    errors1 = errors1[:min_len]
    errors2 = errors2[:min_len]
    
    # Loss differential
    d = errors1**2 - errors2**2
    
    # Mean of loss differential
    d_bar = np.mean(d)
    
    # Variance of loss differential with HAC correction
    gamma0 = np.var(d, ddof=1)
    
    # Long-run variance (simple version without autocorrelation correction)
    variance = gamma0 / len(d)
    
    # Test statistic
    dm_stat = d_bar / np.sqrt(variance)
    
    # p-value (two-tailed)
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return {
        'DM_statistic': dm_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def coverage_probability(
    actual: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    nominal_coverage: float = 0.95
) -> float:
    """
    Calculate coverage probability of prediction intervals.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    lower_bound : np.ndarray
        Lower bounds of prediction intervals
    upper_bound : np.ndarray
        Upper bounds of prediction intervals
    nominal_coverage : float, optional
        Nominal coverage level (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    float
        Empirical coverage probability
    """
    min_len = min(len(actual), len(lower_bound), len(upper_bound))
    actual = actual[:min_len]
    lower_bound = lower_bound[:min_len]
    upper_bound = upper_bound[:min_len]
    
    covered = np.sum((actual >= lower_bound) & (actual <= upper_bound))
    coverage = covered / len(actual)
    
    return coverage


def interval_score(
    actual: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Calculate interval score (proper scoring rule for prediction intervals).
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    lower_bound : np.ndarray
        Lower bounds of prediction intervals
    upper_bound : np.ndarray
        Upper bounds of prediction intervals
    alpha : float, optional
        Significance level (e.g., 0.05 for 95% CI)
        
    Returns
    -------
    float
        Mean interval score (lower is better)
    """
    min_len = min(len(actual), len(lower_bound), len(upper_bound))
    actual = actual[:min_len]
    lower_bound = lower_bound[:min_len]
    upper_bound = upper_bound[:min_len]
    
    interval_width = upper_bound - lower_bound
    
    # Penalty for being outside the interval
    lower_penalty = (2 / alpha) * (lower_bound - actual) * (actual < lower_bound)
    upper_penalty = (2 / alpha) * (actual - upper_bound) * (actual > upper_bound)
    
    scores = interval_width + lower_penalty + upper_penalty
    
    return np.mean(scores)
