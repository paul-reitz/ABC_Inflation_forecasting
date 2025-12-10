"""
Tests for data processing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_processing import (
    load_inflation_data,
    load_features,
    clean_data,
    handle_missing_values,
    create_lags,
    compute_growth_rates
)


class TestDataLoader:
    """Tests for data loading functions."""
    
    def test_create_lags(self):
        """Test lag creation."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5]
        })
        
        result = create_lags(df, ['x'], [1, 2])
        
        assert 'x_lag1' in result.columns
        assert 'x_lag2' in result.columns
        assert result['x_lag1'].iloc[1] == 1
        assert result['x_lag2'].iloc[2] == 1
    
    def test_compute_growth_rates(self):
        """Test growth rate computation."""
        df = pd.DataFrame({
            'x': [100, 110, 120, 130]
        })
        
        result = compute_growth_rates(df, ['x'], [1])
        
        assert 'x_growth1' in result.columns
        assert np.isclose(result['x_growth1'].iloc[1], 10.0)


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_handle_missing_values_interpolate(self):
        """Test missing value interpolation."""
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0, 4.0]
        })
        
        result = handle_missing_values(df, method='interpolate')
        
        assert not result['x'].isna().any()
        assert result['x'].iloc[1] == 2.0
    
    def test_clean_data_outliers(self):
        """Test outlier removal."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 100, 5]
        })
        
        result = clean_data(df, remove_outliers=True)
        
        # Outlier should be clipped
        assert result['x'].max() < 100


if __name__ == "__main__":
    pytest.main([__file__])
