"""
Data processing module for inflation forecasting.

This module handles data loading, cleaning, preprocessing, and feature engineering.
"""

from .data_loader import load_inflation_data, load_features
from .preprocessing import clean_data, handle_missing_values
from .feature_engineering import create_lags, compute_growth_rates

__all__ = [
    'load_inflation_data',
    'load_features',
    'clean_data',
    'handle_missing_values',
    'create_lags',
    'compute_growth_rates'
]
