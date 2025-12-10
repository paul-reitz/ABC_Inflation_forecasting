"""
Tests for ABC Inflation Model
"""

import numpy as np
import pytest
from scipy import stats
from abc_inflation.models import ABCInflationModel
from abc_inflation.utils import generate_summary_statistics


def test_abc_model_initialization():
    """Test ABC model initialization."""
    prior_distributions = {
        'phi': stats.uniform(0.3, 0.6),
        'sigma': stats.uniform(0.1, 1.0),
        'mu': stats.norm(4.5, 1.0),
    }
    
    model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics,
        n_particles=100
    )
    
    assert model.n_particles == 100
    assert model.distance_metric == "euclidean"
    assert len(model.epsilon_schedule) == 4


def test_sample_prior():
    """Test prior sampling."""
    prior_distributions = {
        'phi': stats.uniform(0.3, 0.6),
        'sigma': stats.uniform(0.1, 1.0),
    }
    
    model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics
    )
    
    sample = model.sample_prior()
    
    assert 'phi' in sample
    assert 'sigma' in sample
    assert 0.3 <= sample['phi'] <= 0.9
    assert 0.1 <= sample['sigma'] <= 1.1


def test_compute_distance():
    """Test distance computation."""
    prior_distributions = {
        'phi': stats.uniform(0.3, 0.6),
    }
    
    model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics,
        distance_metric='euclidean'
    )
    
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.1, 2.1, 3.1])
    
    distance = model.compute_distance(obs, sim)
    
    expected = np.sqrt(3 * (0.1 ** 2))
    assert np.isclose(distance, expected)


def test_simulate_model():
    """Test model simulation."""
    prior_distributions = {
        'phi': stats.uniform(0.3, 0.6),
        'sigma': stats.uniform(0.1, 1.0),
        'mu': stats.norm(4.5, 1.0),
    }
    
    model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics
    )
    
    params = {'phi': 0.8, 'sigma': 0.5, 'mu': 2.0}
    n_steps = 50
    
    simulated = model.simulate_model(params, n_steps)
    
    assert len(simulated) == n_steps
    assert not np.any(np.isnan(simulated))


def test_forecast_without_fit():
    """Test that forecasting without fitting raises error."""
    prior_distributions = {
        'phi': stats.uniform(0.3, 0.6),
    }
    
    model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics
    )
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.forecast(n_steps=10)


def test_abc_rejection_simple():
    """Test ABC rejection sampling with simple data."""
    np.random.seed(42)
    
    # Generate test data
    data = np.random.normal(5.0, 1.0, 50)
    
    prior_distributions = {
        'phi': stats.uniform(0.4, 0.5),  # uniform[0.4, 0.9]
        'sigma': stats.uniform(0.1, 1.0),  # uniform[0.1, 1.1]
        'mu': stats.norm(4.5, 2.0),
    }
    
    model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics,
        n_particles=10
    )
    
    # Run rejection with lenient tolerance
    accepted = model.abc_rejection(data, epsilon=10.0, max_iterations=1000)
    
    assert len(accepted) > 0
    assert len(accepted) <= 10


if __name__ == "__main__":
    pytest.main([__file__])
