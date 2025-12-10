"""
Data Loading and Preprocessing Utilities

This module provides functions for loading and preprocessing inflation data,
including South African inflation data and SARB forecasts.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings


def load_inflation_data(
    filepath: str,
    date_column: str = 'date',
    value_column: str = 'inflation',
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Load inflation time series data.
    
    Parameters
    ----------
    filepath : str
        Path to the data file (CSV, Excel, etc.)
    date_column : str, optional
        Name of the date column
    value_column : str, optional
        Name of the inflation value column
    freq : str, optional
        Frequency of the time series ('M' for monthly, 'Q' for quarterly)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and inflation values
    """
    # Determine file type and load
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df = df.set_index(date_column)
    
    # Sort by date
    df = df.sort_index()
    
    # Ensure regular frequency
    df = df.asfreq(freq)
    
    return df


def preprocess_data(
    data: pd.DataFrame,
    value_column: str = 'inflation',
    handle_missing: str = 'interpolate',
    remove_outliers: bool = False,
    outlier_std: float = 3.0
) -> pd.Series:
    """
    Preprocess inflation data.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with inflation data
    value_column : str, optional
        Name of the value column
    handle_missing : str, optional
        How to handle missing values ('interpolate', 'forward_fill', 'drop')
    remove_outliers : bool, optional
        Whether to remove outliers
    outlier_std : float, optional
        Number of standard deviations for outlier detection
        
    Returns
    -------
    pd.Series
        Preprocessed inflation series
    """
    series = data[value_column].copy()
    
    # Handle missing values
    if handle_missing == 'interpolate':
        series = series.interpolate(method='linear')
    elif handle_missing == 'forward_fill':
        series = series.fillna(method='ffill')
    elif handle_missing == 'drop':
        series = series.dropna()
    else:
        raise ValueError(f"Unknown missing value handling method: {handle_missing}")
    
    # Remove outliers if requested
    if remove_outliers:
        mean = series.mean()
        std = series.std()
        lower_bound = mean - outlier_std * std
        upper_bound = mean + outlier_std * std
        
        # Replace outliers with boundary values
        series = series.clip(lower=lower_bound, upper=upper_bound)
    
    return series


def create_train_test_split(
    data: np.ndarray,
    test_size: int = 12,
    validation_size: int = 0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Create train-test split for time series data.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    test_size : int, optional
        Number of observations to use for testing
    validation_size : int, optional
        Number of observations to use for validation
        
    Returns
    -------
    tuple
        (train, test, validation) arrays. Validation is None if validation_size=0
    """
    if validation_size > 0:
        train = data[:-test_size-validation_size]
        validation = data[-test_size-validation_size:-test_size]
        test = data[-test_size:]
        return train, test, validation
    else:
        train = data[:-test_size]
        test = data[-test_size:]
        return train, test, None


def generate_summary_statistics(data: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics for ABC.
    
    This function computes a set of summary statistics that capture
    key features of the inflation time series for use in ABC.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
        
    Returns
    -------
    np.ndarray
        Array of summary statistics
    """
    stats = []
    
    # Basic moments
    stats.append(np.mean(data))
    stats.append(np.std(data))
    stats.append(np.median(data))
    
    # Skewness and kurtosis
    from scipy import stats as sp_stats
    stats.append(sp_stats.skew(data))
    stats.append(sp_stats.kurtosis(data))
    
    # Autocorrelation at different lags
    for lag in [1, 2, 3, 6, 12]:
        if len(data) > lag:
            acf = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            stats.append(acf)
        else:
            stats.append(0.0)
    
    # Quantiles
    stats.append(np.percentile(data, 25))
    stats.append(np.percentile(data, 75))
    
    # Volatility (rolling std)
    if len(data) > 12:
        rolling_std = pd.Series(data).rolling(window=12).std().mean()
        stats.append(rolling_std)
    else:
        stats.append(np.std(data))
    
    return np.array(stats)


def load_sarb_forecasts(filepath: str) -> pd.DataFrame:
    """
    Load SARB inflation forecasts.
    
    Parameters
    ----------
    filepath : str
        Path to SARB forecast file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with SARB forecasts
    """
    # This is a placeholder - actual implementation depends on SARB data format
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Assume the file has columns: date, forecast, lower_bound, upper_bound
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    return df


def create_rolling_forecasts(
    data: np.ndarray,
    model_class,
    window_size: int,
    forecast_horizon: int = 1,
    **model_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rolling window forecasts.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    model_class : class
        Model class to use (must have fit and forecast methods)
    window_size : int
        Size of the rolling window
    forecast_horizon : int, optional
        Number of steps ahead to forecast
    **model_kwargs
        Additional arguments for model initialization
        
    Returns
    -------
    tuple
        (forecasts, actuals) arrays
    """
    n = len(data)
    forecasts = []
    actuals = []
    
    for i in range(window_size, n - forecast_horizon + 1):
        # Training window
        train = data[i-window_size:i]
        
        # Actual value
        actual = data[i:i+forecast_horizon]
        
        # Fit model and forecast
        model = model_class(**model_kwargs)
        try:
            model.fit(train)
            forecast, _, _ = model.forecast(forecast_horizon)
            forecasts.append(forecast)
            actuals.append(actual)
        except Exception as e:
            warnings.warn(f"Failed to fit model at index {i}: {e}")
            continue
    
    return np.array(forecasts), np.array(actuals)
