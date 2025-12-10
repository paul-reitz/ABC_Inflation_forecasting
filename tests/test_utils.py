"""
Tests for Utility Functions
"""

import numpy as np
import pytest
from abc_inflation.utils import forecast_metrics, generate_summary_statistics


def test_forecast_metrics():
    """Test forecast metrics calculation."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    metrics = forecast_metrics(actual, predicted, model_name="Test")
    
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'MAPE' in metrics
    assert 'R2' in metrics
    
    assert metrics['model'] == "Test"
    assert metrics['MAE'] > 0
    assert metrics['RMSE'] > 0


def test_generate_summary_statistics():
    """Test summary statistics generation."""
    data = np.random.normal(5.0, 1.0, 100)
    
    stats = generate_summary_statistics(data)
    
    assert len(stats) > 0
    assert not np.any(np.isnan(stats))
    
    # Should include mean, std, etc.
    assert np.isclose(stats[0], np.mean(data), rtol=1e-5)
    assert np.isclose(stats[1], np.std(data), rtol=1e-5)


def test_summary_statistics_shape():
    """Test that summary statistics have consistent shape."""
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(5, 2, 100)
    
    stats1 = generate_summary_statistics(data1)
    stats2 = generate_summary_statistics(data2)
    
    assert stats1.shape == stats2.shape


if __name__ == "__main__":
    pytest.main([__file__])
