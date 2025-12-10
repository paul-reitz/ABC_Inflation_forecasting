"""
Tests for state-space models.
"""

import pytest
import numpy as np
from src.models.state_space import (
    LocalLevelModel,
    LocalLinearTrendModel,
    UnobservedComponentsModel
)


class TestLocalLevelModel:
    """Tests for local level model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = LocalLevelModel()
        
        assert model.n_state == 1
        assert model.n_obs == 1
    
    def test_build_matrices(self):
        """Test matrix construction."""
        model = LocalLevelModel()
        params = {'sigma_eps': 1.0, 'sigma_eta': 0.5}
        
        model.build_matrices(params)
        
        assert model.Z is not None
        assert model.T is not None
        assert model.H[0, 0] == 1.0
        assert model.Q[0, 0] == 0.25
    
    def test_kalman_filter(self):
        """Test Kalman filter."""
        model = LocalLevelModel()
        params = {'sigma_eps': 1.0, 'sigma_eta': 0.5}
        model.build_matrices(params)
        
        # Generate some data
        np.random.seed(42)
        y = np.random.randn(50, 1)
        
        alpha_filtered, P_filtered, log_lik = model.kalman_filter(y)
        
        assert alpha_filtered.shape == (50, 1)
        assert P_filtered.shape == (50, 1, 1)
        assert isinstance(log_lik, (int, float))


class TestLocalLinearTrendModel:
    """Tests for local linear trend model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = LocalLinearTrendModel()
        
        assert model.n_state == 2
        assert model.n_obs == 1
    
    def test_forecast(self):
        """Test forecasting."""
        model = LocalLinearTrendModel()
        params = {'sigma_eps': 1.0, 'sigma_mu': 0.5, 'sigma_nu': 0.1}
        
        # Fit model
        np.random.seed(42)
        y = np.random.randn(50, 1)
        model.fit(y, params)
        
        # Generate forecasts
        forecasts = model.forecast(h=10)
        
        assert forecasts.shape == (10, 1)


class TestUnobservedComponentsModel:
    """Tests for unobserved components model."""
    
    def test_model_with_features(self):
        """Test model with features."""
        model = UnobservedComponentsModel(
            n_features=3,
            include_cycle=True
        )
        
        assert model.n_features == 3
        assert model.include_cycle is True


if __name__ == "__main__":
    pytest.main([__file__])
