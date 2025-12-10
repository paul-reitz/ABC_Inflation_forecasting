"""
State-space models for inflation forecasting.

This module implements structural Bayesian state-space models including:
- Local level models
- Local linear trend models
- Unobserved components models with multivariate features
"""

from .base_model import StateSpaceModel
from .local_level import LocalLevelModel
from .local_linear_trend import LocalLinearTrendModel
from .unobserved_components import UnobservedComponentsModel

__all__ = [
    'StateSpaceModel',
    'LocalLevelModel',
    'LocalLinearTrendModel',
    'UnobservedComponentsModel'
]
