"""
Forecasting module for inflation models.
"""

from .forecast_methods import point_forecast, interval_forecast, probabilistic_forecast
from .evaluation import evaluate_forecasts, compute_forecast_errors

__all__ = [
    'point_forecast',
    'interval_forecast', 
    'probabilistic_forecast',
    'evaluate_forecasts',
    'compute_forecast_errors'
]
