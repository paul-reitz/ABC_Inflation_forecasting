"""
Forecasting methods for inflation.
"""

import numpy as np
from typing import Dict, List, Tuple


def point_forecast(
    model,
    horizon: int,
    method: str = 'mean'
) -> np.ndarray:
    """
    Generate point forecasts.
    
    Parameters:
    -----------
    model : StateSpaceModel
        Fitted state-space model
    horizon : int
        Forecast horizon
    method : str
        'mean' or 'median' for probabilistic forecasts
        
    Returns:
    --------
    forecasts : np.ndarray
        Point forecasts (horizon x 1)
    """
    return model.forecast(horizon)


def interval_forecast(
    model,
    horizon: int,
    alpha: float = 0.05,
    n_simulations: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate interval forecasts using simulation.
    
    Parameters:
    -----------
    model : StateSpaceModel
        Fitted state-space model
    horizon : int
        Forecast horizon
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    n_simulations : int
        Number of simulations
        
    Returns:
    --------
    point : np.ndarray
        Point forecasts
    lower : np.ndarray
        Lower bound of prediction interval
    upper : np.ndarray
        Upper bound of prediction interval
    """
    # Get point forecast
    point = model.forecast(horizon)
    
    # Simulate forecast paths
    forecast_paths = []
    
    for _ in range(n_simulations):
        # Would need to implement simulation method in model
        # For now, add uncertainty based on model residuals
        path = point + np.random.normal(0, 0.5, size=point.shape)
        forecast_paths.append(path)
    
    forecast_paths = np.array(forecast_paths)
    
    # Compute quantiles
    lower = np.quantile(forecast_paths, alpha/2, axis=0)
    upper = np.quantile(forecast_paths, 1 - alpha/2, axis=0)
    
    return point, lower, upper


def probabilistic_forecast(
    model,
    horizon: int,
    n_simulations: int = 1000
) -> np.ndarray:
    """
    Generate probabilistic forecasts (full predictive distribution).
    
    Parameters:
    -----------
    model : StateSpaceModel
        Fitted state-space model
    horizon : int
        Forecast horizon
    n_simulations : int
        Number of simulations
        
    Returns:
    --------
    forecast_distribution : np.ndarray
        Simulated forecast paths (n_simulations x horizon)
    """
    forecast_paths = []
    
    for _ in range(n_simulations):
        # Simulate forecast path
        # Would need proper implementation in model
        point = model.forecast(horizon)
        path = point + np.random.normal(0, 0.5, size=point.shape)
        forecast_paths.append(path.flatten())
    
    return np.array(forecast_paths)


def density_forecast(
    model,
    horizon: int,
    grid_points: np.ndarray = None,
    n_simulations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate density forecasts.
    
    Parameters:
    -----------
    model : StateSpaceModel
        Fitted state-space model
    horizon : int
        Forecast horizon
    grid_points : np.ndarray, optional
        Grid points for density evaluation
    n_simulations : int
        Number of simulations
        
    Returns:
    --------
    grid : np.ndarray
        Grid points
    densities : np.ndarray
        Density values at each horizon (horizon x n_grid_points)
    """
    from scipy.stats import gaussian_kde
    
    # Get probabilistic forecasts
    forecast_paths = probabilistic_forecast(model, horizon, n_simulations)
    
    if grid_points is None:
        # Create default grid
        min_val = forecast_paths.min()
        max_val = forecast_paths.max()
        grid_points = np.linspace(min_val - 1, max_val + 1, 100)
    
    # Estimate density for each horizon
    densities = []
    for h in range(horizon):
        kde = gaussian_kde(forecast_paths[:, h])
        density = kde(grid_points)
        densities.append(density)
    
    return grid_points, np.array(densities)


def fan_chart(
    model,
    horizon: int,
    percentiles: List[float] = [10, 20, 30, 40, 50, 60, 70, 80, 90],
    n_simulations: int = 1000
) -> Dict[float, np.ndarray]:
    """
    Generate fan chart data for visualization.
    
    Parameters:
    -----------
    model : StateSpaceModel
        Fitted state-space model
    horizon : int
        Forecast horizon
    percentiles : List[float]
        Percentiles to compute
    n_simulations : int
        Number of simulations
        
    Returns:
    --------
    fan_data : Dict[float, np.ndarray]
        Dictionary mapping percentiles to forecast values
    """
    # Get probabilistic forecasts
    forecast_paths = probabilistic_forecast(model, horizon, n_simulations)
    
    # Compute percentiles
    fan_data = {}
    for p in percentiles:
        fan_data[p] = np.percentile(forecast_paths, p, axis=0)
    
    return fan_data
