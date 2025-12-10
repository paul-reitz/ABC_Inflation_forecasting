"""
Approximate Bayesian Computation (ABC) methods for parameter estimation.

This module implements ABC algorithms for estimating parameters of
state-space models when the likelihood is intractable or expensive to compute.
"""

from .abc_rejection import ABCRejection
from .abc_smc import ABCSMC
from .distance_metrics import compute_distance, summary_statistics

__all__ = [
    'ABCRejection',
    'ABCSMC',
    'compute_distance',
    'summary_statistics'
]
