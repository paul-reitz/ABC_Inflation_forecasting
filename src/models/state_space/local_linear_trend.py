"""
Local linear trend model.
"""

import numpy as np
from typing import Dict
from .base_model import StateSpaceModel


class LocalLinearTrendModel(StateSpaceModel):
    """
    Local linear trend model for inflation forecasting.
    
    Observation equation: y_t = mu_t + eps_t,  eps_t ~ N(0, sigma_eps^2)
    State equations:
        mu_t = mu_{t-1} + nu_{t-1} + eta1_t,  eta1_t ~ N(0, sigma_mu^2)
        nu_t = nu_{t-1} + eta2_t,  eta2_t ~ N(0, sigma_nu^2)
    
    where mu_t is the level and nu_t is the slope (trend).
    """
    
    def __init__(self):
        """Initialize local linear trend model."""
        super().__init__(n_state=2, n_obs=1)
    
    def build_matrices(self, params: Dict) -> None:
        """
        Build state-space matrices.
        
        Parameters:
        -----------
        params : Dict
            Must contain 'sigma_eps', 'sigma_mu', 'sigma_nu'
        """
        sigma_eps = params.get('sigma_eps', 1.0)
        sigma_mu = params.get('sigma_mu', 1.0)
        sigma_nu = params.get('sigma_nu', 0.1)
        
        # Observation matrix [1, 0]
        self.Z = np.array([[1.0, 0.0]])
        
        # Transition matrix
        self.T = np.array([
            [1.0, 1.0],  # mu_t = mu_{t-1} + nu_{t-1} + eta1_t
            [0.0, 1.0]   # nu_t = nu_{t-1} + eta2_t
        ])
        
        # Selection matrix
        self.R = np.eye(2)
        
        # Observation covariance
        self.H = np.array([[sigma_eps**2]])
        
        # State covariance
        self.Q = np.array([
            [sigma_mu**2, 0.0],
            [0.0, sigma_nu**2]
        ])
        
        # Initial state (diffuse initialization)
        self.alpha_0 = np.array([0.0, 0.0])
        self.P_0 = np.eye(2) * 1e6
