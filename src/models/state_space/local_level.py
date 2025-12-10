"""
Local level model (random walk plus noise).
"""

import numpy as np
from typing import Dict
from .base_model import StateSpaceModel


class LocalLevelModel(StateSpaceModel):
    """
    Local level model for inflation forecasting.
    
    Observation equation: y_t = alpha_t + eps_t,  eps_t ~ N(0, sigma_eps^2)
    State equation: alpha_t = alpha_{t-1} + eta_t,  eta_t ~ N(0, sigma_eta^2)
    
    This is the simplest state-space model, equivalent to exponential smoothing.
    """
    
    def __init__(self):
        """Initialize local level model."""
        super().__init__(n_state=1, n_obs=1)
    
    def build_matrices(self, params: Dict) -> None:
        """
        Build state-space matrices.
        
        Parameters:
        -----------
        params : Dict
            Must contain 'sigma_eps' and 'sigma_eta'
        """
        sigma_eps = params.get('sigma_eps', 1.0)
        sigma_eta = params.get('sigma_eta', 1.0)
        
        # Observation matrix
        self.Z = np.array([[1.0]])
        
        # Transition matrix
        self.T = np.array([[1.0]])
        
        # Selection matrix
        self.R = np.array([[1.0]])
        
        # Observation covariance
        self.H = np.array([[sigma_eps**2]])
        
        # State covariance
        self.Q = np.array([[sigma_eta**2]])
        
        # Initial state (diffuse initialization)
        self.alpha_0 = np.array([0.0])
        self.P_0 = np.array([[1e6]])  # Large variance for diffuse initialization
