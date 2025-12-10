"""
Base class for state-space models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class StateSpaceModel(ABC):
    """
    Base class for state-space models.
    
    State-space representation:
    Observation equation: y_t = Z_t @ alpha_t + eps_t
    State equation: alpha_t = T_t @ alpha_{t-1} + R_t @ eta_t
    
    where:
    - y_t: observed data (n_obs x 1)
    - alpha_t: state vector (n_state x 1)
    - Z_t: observation matrix
    - T_t: transition matrix
    - R_t: selection matrix
    - eps_t ~ N(0, H_t): observation noise
    - eta_t ~ N(0, Q_t): state noise
    """
    
    def __init__(self, n_state: int, n_obs: int = 1):
        """
        Initialize state-space model.
        
        Parameters:
        -----------
        n_state : int
            Dimension of state vector
        n_obs : int
            Dimension of observation vector
        """
        self.n_state = n_state
        self.n_obs = n_obs
        
        # Model matrices (to be defined by subclasses)
        self.Z = None  # Observation matrix
        self.T = None  # Transition matrix
        self.R = None  # Selection matrix
        self.H = None  # Observation covariance
        self.Q = None  # State covariance
        
        # Initial state
        self.alpha_0 = None  # Initial state mean
        self.P_0 = None      # Initial state covariance
        
        # Fitted values
        self.filtered_states = None
        self.smoothed_states = None
        self.log_likelihood = None
    
    @abstractmethod
    def build_matrices(self, params: Dict) -> None:
        """
        Build state-space matrices from parameters.
        
        Parameters:
        -----------
        params : Dict
            Model parameters
        """
        pass
    
    def kalman_filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply Kalman filter to obtain filtered states.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed data (T x n_obs)
            
        Returns:
        --------
        alpha_filtered : np.ndarray
            Filtered state estimates (T x n_state)
        P_filtered : np.ndarray
            Filtered state covariances (T x n_state x n_state)
        log_likelihood : float
            Log-likelihood of the data
        """
        T = len(y)
        
        # Initialize
        alpha_filtered = np.zeros((T, self.n_state))
        P_filtered = np.zeros((T, self.n_state, self.n_state))
        
        alpha_pred = self.alpha_0.copy()
        P_pred = self.P_0.copy()
        
        log_likelihood = 0.0
        
        for t in range(T):
            # Prediction error
            v_t = y[t] - self.Z @ alpha_pred
            
            # Prediction error variance
            F_t = self.Z @ P_pred @ self.Z.T + self.H
            
            # Kalman gain
            K_t = P_pred @ self.Z.T @ np.linalg.inv(F_t)
            
            # Filtered state
            alpha_filtered[t] = alpha_pred + K_t @ v_t
            P_filtered[t] = P_pred - K_t @ self.Z @ P_pred
            
            # Log-likelihood contribution
            log_likelihood += -0.5 * (
                np.log(2 * np.pi) + 
                np.log(np.linalg.det(F_t)) + 
                v_t.T @ np.linalg.inv(F_t) @ v_t
            )
            
            # Prediction for next period
            if t < T - 1:
                alpha_pred = self.T @ alpha_filtered[t]
                P_pred = self.T @ P_filtered[t] @ self.T.T + self.R @ self.Q @ self.R.T
        
        self.filtered_states = alpha_filtered
        self.log_likelihood = log_likelihood
        
        return alpha_filtered, P_filtered, log_likelihood
    
    def kalman_smoother(
        self, 
        y: np.ndarray,
        alpha_filtered: np.ndarray,
        P_filtered: np.ndarray
    ) -> np.ndarray:
        """
        Apply Kalman smoother to obtain smoothed states.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed data
        alpha_filtered : np.ndarray
            Filtered states from Kalman filter
        P_filtered : np.ndarray
            Filtered covariances from Kalman filter
            
        Returns:
        --------
        alpha_smoothed : np.ndarray
            Smoothed state estimates
        """
        T = len(y)
        alpha_smoothed = np.zeros_like(alpha_filtered)
        
        # Initialize with last filtered state
        alpha_smoothed[-1] = alpha_filtered[-1]
        
        # Backward recursion
        for t in range(T - 2, -1, -1):
            P_pred = self.T @ P_filtered[t] @ self.T.T + self.R @ self.Q @ self.R.T
            J_t = P_filtered[t] @ self.T.T @ np.linalg.inv(P_pred)
            alpha_smoothed[t] = alpha_filtered[t] + J_t @ (alpha_smoothed[t + 1] - self.T @ alpha_filtered[t])
        
        self.smoothed_states = alpha_smoothed
        
        return alpha_smoothed
    
    def fit(self, y: np.ndarray, params: Dict) -> None:
        """
        Fit the state-space model.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed data
        params : Dict
            Model parameters
        """
        # Build matrices
        self.build_matrices(params)
        
        # Run Kalman filter
        alpha_filtered, P_filtered, log_likelihood = self.kalman_filter(y)
        
        # Run Kalman smoother
        self.kalman_smoother(y, alpha_filtered, P_filtered)
    
    def forecast(self, h: int) -> np.ndarray:
        """
        Generate forecasts h steps ahead.
        
        Parameters:
        -----------
        h : int
            Forecast horizon
            
        Returns:
        --------
        forecasts : np.ndarray
            Point forecasts (h x n_obs)
        """
        if self.filtered_states is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = np.zeros((h, self.n_obs))
        alpha_t = self.filtered_states[-1]
        
        for i in range(h):
            # Forecast state
            alpha_t = self.T @ alpha_t
            
            # Forecast observation
            forecasts[i] = self.Z @ alpha_t
        
        return forecasts
