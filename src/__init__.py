"""
ABC Inflation Forecasting Package

This package provides tools for inflation forecasting using Approximate Bayesian
Computation (ABC) and structural Bayesian state-space models with multivariate features.
"""

__version__ = "0.1.0"
__author__ = "Paul Reitz"

# Import main components for easy access
from src.models.state_space import *
from src.models.abc import *
from src.models.forecasting import *
