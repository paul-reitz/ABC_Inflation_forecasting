"""
Baseline Models for Inflation Forecasting

This module implements traditional time series models (ARIMA, SARIMA) to serve
as baselines for comparison with the ABC approach.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Optional
import warnings


class ARIMAModel:
    """
    ARIMA model for inflation forecasting.
    
    Parameters
    ----------
    order : tuple
        (p, d, q) order of the ARIMA model
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: np.ndarray, **kwargs):
        """
        Fit the ARIMA model to data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        **kwargs
            Additional arguments for ARIMA.fit()
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit(**kwargs)
    
    def forecast(
        self,
        steps: int,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        alpha : float, optional
            Significance level for confidence intervals
            
        Returns
        -------
        tuple
            (forecast, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
        
        # Handle different return types from statsmodels
        if isinstance(forecast_result, pd.DataFrame):
            forecast = forecast_result['mean'].values
            lower = forecast_result.iloc[:, 1].values
            upper = forecast_result.iloc[:, 2].values
        else:
            forecast = forecast_result
            # Get prediction intervals separately
            forecast_obj = self.fitted_model.get_forecast(steps=steps)
            pred_int = forecast_obj.conf_int(alpha=alpha)
            lower = pred_int.iloc[:, 0].values
            upper = pred_int.iloc[:, 1].values
        
        return forecast, lower, upper
    
    def get_params(self) -> dict:
        """
        Get fitted model parameters.
        
        Returns
        -------
        dict
            Dictionary of model parameters
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return {
            'order': self.order,
            'params': self.fitted_model.params.to_dict(),
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
        }
    
    def summary(self):
        """Print model summary."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.summary()


class SARIMAModel:
    """
    SARIMA model for inflation forecasting with seasonal components.
    
    Parameters
    ----------
    order : tuple
        (p, d, q) order of the ARIMA model
    seasonal_order : tuple
        (P, D, Q, s) seasonal order
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: np.ndarray, **kwargs):
        """
        Fit the SARIMA model to data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        **kwargs
            Additional arguments for SARIMAX.fit()
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False, **kwargs)
    
    def forecast(
        self,
        steps: int,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        alpha : float, optional
            Significance level for confidence intervals
            
        Returns
        -------
        tuple
            (forecast, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_obj = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean.values
        pred_int = forecast_obj.conf_int(alpha=alpha)
        lower = pred_int.iloc[:, 0].values
        upper = pred_int.iloc[:, 1].values
        
        return forecast, lower, upper
    
    def get_params(self) -> dict:
        """
        Get fitted model parameters.
        
        Returns
        -------
        dict
            Dictionary of model parameters
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'params': self.fitted_model.params.to_dict(),
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
        }
    
    def summary(self):
        """Print model summary."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.summary()
