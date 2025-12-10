"""
Unobserved components model with multivariate features.
"""

import numpy as np
from typing import Dict, Optional
from .base_model import StateSpaceModel


class UnobservedComponentsModel(StateSpaceModel):
    """
    Unobserved components model with trend, cycle, and seasonal components.
    Includes multivariate features in the observation equation.
    
    Observation equation: 
        y_t = mu_t + gamma_t + beta' X_t + eps_t
    
    State equations:
        Trend: mu_t = mu_{t-1} + nu_{t-1} + eta_mu_t
               nu_t = nu_{t-1} + eta_nu_t
        Cycle: gamma_t = rho * gamma_{t-1} + eta_gamma_t
        (Seasonal component can be added)
    
    where X_t are exogenous features with coefficients beta.
    """
    
    def __init__(
        self, 
        n_features: int = 0,
        include_cycle: bool = True,
        seasonal_period: Optional[int] = None
    ):
        """
        Initialize unobserved components model.
        
        Parameters:
        -----------
        n_features : int
            Number of exogenous features
        include_cycle : bool
            Whether to include a cyclical component
        seasonal_period : int, optional
            Seasonal period (e.g., 12 for monthly data)
        """
        # State dimension: trend (2) + cycle (1 if included) + seasonal (period-1 if included)
        n_state = 2  # Trend: level + slope
        
        if include_cycle:
            n_state += 1
        
        if seasonal_period:
            n_state += seasonal_period - 1
        
        super().__init__(n_state=n_state, n_obs=1)
        
        self.n_features = n_features
        self.include_cycle = include_cycle
        self.seasonal_period = seasonal_period
        
        # Feature coefficients
        self.beta = None
    
    def build_matrices(self, params: Dict, X: Optional[np.ndarray] = None) -> None:
        """
        Build state-space matrices.
        
        Parameters:
        -----------
        params : Dict
            Model parameters including:
            - sigma_eps: observation noise std
            - sigma_mu: level noise std
            - sigma_nu: slope noise std
            - sigma_gamma: cycle noise std (if include_cycle)
            - rho: cycle AR(1) coefficient (if include_cycle)
            - beta: feature coefficients (if n_features > 0)
        X : np.ndarray, optional
            Exogenous features (T x n_features)
        """
        sigma_eps = params.get('sigma_eps', 1.0)
        sigma_mu = params.get('sigma_mu', 1.0)
        sigma_nu = params.get('sigma_nu', 0.1)
        
        # Build observation matrix
        Z = [1.0, 0.0]  # Level component
        
        idx = 2  # Current state index
        
        # Cycle component
        if self.include_cycle:
            Z.append(1.0)
            idx += 1
        
        # Seasonal component
        if self.seasonal_period:
            Z.extend([1.0] + [0.0] * (self.seasonal_period - 2))
            idx += self.seasonal_period - 1
        
        self.Z = np.array([Z])
        
        # Build transition matrix
        T = np.zeros((self.n_state, self.n_state))
        
        # Trend component
        T[0, 0] = 1.0  # Level
        T[0, 1] = 1.0  # Slope
        T[1, 1] = 1.0  # Slope persistence
        
        idx = 2
        
        # Cycle component
        if self.include_cycle:
            rho = params.get('rho', 0.9)
            T[idx, idx] = rho
            idx += 1
        
        # Seasonal component
        if self.seasonal_period:
            s = self.seasonal_period
            T[idx:idx+s-1, idx:idx+s-1] = np.eye(s-1)
            T[idx, idx:idx+s-1] = -np.ones(s-1)
            idx += s - 1
        
        self.T = T
        
        # Selection matrix
        self.R = np.eye(self.n_state)
        
        # Observation covariance
        self.H = np.array([[sigma_eps**2]])
        
        # State covariance
        Q = np.zeros((self.n_state, self.n_state))
        Q[0, 0] = sigma_mu**2
        Q[1, 1] = sigma_nu**2
        
        idx = 2
        
        if self.include_cycle:
            sigma_gamma = params.get('sigma_gamma', 1.0)
            Q[idx, idx] = sigma_gamma**2
            idx += 1
        
        if self.seasonal_period:
            sigma_seasonal = params.get('sigma_seasonal', 0.5)
            s = self.seasonal_period
            Q[idx, idx] = sigma_seasonal**2
        
        self.Q = Q
        
        # Feature coefficients
        if self.n_features > 0:
            self.beta = params.get('beta', np.zeros(self.n_features))
        
        # Initial state
        self.alpha_0 = np.zeros(self.n_state)
        self.P_0 = np.eye(self.n_state) * 1e6
    
    def kalman_filter_with_features(
        self, 
        y: np.ndarray,
        X: np.ndarray
    ) -> tuple:
        """
        Kalman filter with exogenous features.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed data (T x 1)
        X : np.ndarray
            Exogenous features (T x n_features)
            
        Returns:
        --------
        Same as parent kalman_filter method
        """
        # Adjust observations by feature contribution
        if self.beta is not None and X is not None:
            y_adjusted = y - X @ self.beta
        else:
            y_adjusted = y
        
        return self.kalman_filter(y_adjusted)
