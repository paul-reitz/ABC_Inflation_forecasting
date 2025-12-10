"""
Tests for forecasting methods.
"""

import pytest
import numpy as np
from src.models.forecasting import (
    compute_forecast_errors,
    evaluate_forecasts
)


class TestForecastEvaluation:
    """Tests for forecast evaluation metrics."""
    
    def test_compute_forecast_errors(self):
        """Test forecast error computation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = compute_forecast_errors(actual, predicted)
        
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        assert metrics['MAE'] > 0
        assert metrics['RMSE'] > 0
    
    def test_evaluate_forecasts_point(self):
        """Test point forecast evaluation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        evaluation = evaluate_forecasts(actual, predicted)
        
        assert 'point_metrics' in evaluation
        assert 'MAE' in evaluation['point_metrics']
    
    def test_evaluate_forecasts_intervals(self):
        """Test interval forecast evaluation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        lower = predicted - 0.5
        upper = predicted + 0.5
        
        evaluation = evaluate_forecasts(
            actual, 
            predicted,
            prediction_intervals=(lower, upper),
            alpha=0.05
        )
        
        assert 'interval_metrics' in evaluation
        assert 'coverage' in evaluation['interval_metrics']


if __name__ == "__main__":
    pytest.main([__file__])
