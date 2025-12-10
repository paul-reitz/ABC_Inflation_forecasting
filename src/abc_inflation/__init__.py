"""
ABC Inflation Forecasting Package

This package implements Approximate Bayesian Computation (ABC) methods
for inflation forecasting, with a focus on comparing performance against
South African Reserve Bank (SARB) forecasts.
"""

__version__ = "0.1.0"

from .models.abc_model import ABCInflationModel
from .models.baseline_models import ARIMAModel, SARIMAModel
from .utils.metrics import forecast_metrics

__all__ = [
    "ABCInflationModel",
    "ARIMAModel",
    "SARIMAModel",
    "forecast_metrics",
]
