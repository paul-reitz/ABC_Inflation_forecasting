"""
Tests for Baseline Models
"""

import numpy as np
import pytest
from abc_inflation.models import ARIMAModel, SARIMAModel


def test_arima_initialization():
    """Test ARIMA model initialization."""
    model = ARIMAModel(order=(1, 1, 1))
    assert model.order == (1, 1, 1)
    assert model.model is None
    assert model.fitted_model is None


def test_arima_fit_forecast():
    """Test ARIMA fit and forecast."""
    np.random.seed(42)
    
    # Generate test data
    data = np.cumsum(np.random.normal(0, 1, 100)) + 50
    
    model = ARIMAModel(order=(1, 1, 1))
    model.fit(data)
    
    assert model.fitted_model is not None
    
    # Forecast
    forecast, lower, upper = model.forecast(steps=10)
    
    assert len(forecast) == 10
    assert len(lower) == 10
    assert len(upper) == 10
    assert np.all(lower <= forecast)
    assert np.all(forecast <= upper)


def test_sarima_initialization():
    """Test SARIMA model initialization."""
    model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
    assert model.order == (1, 1, 1)
    assert model.seasonal_order == (1, 0, 1, 12)


def test_sarima_fit_forecast():
    """Test SARIMA fit and forecast."""
    np.random.seed(42)
    
    # Generate test data with seasonality
    t = np.arange(100)
    data = 50 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 100)
    
    model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
    model.fit(data)
    
    assert model.fitted_model is not None
    
    # Forecast
    forecast, lower, upper = model.forecast(steps=12)
    
    assert len(forecast) == 12
    assert len(lower) == 12
    assert len(upper) == 12


def test_get_params_before_fit():
    """Test that getting params before fit raises error."""
    model = ARIMAModel(order=(1, 1, 1))
    
    with pytest.raises(ValueError, match="Model must be fitted first"):
        model.get_params()


if __name__ == "__main__":
    pytest.main([__file__])
