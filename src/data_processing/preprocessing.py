"""
Data preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def clean_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by handling outliers and data quality issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    remove_outliers : bool
        Whether to remove outliers using IQR method
        
    Returns:
    --------
    pd.DataFrame
        Cleaned data
    """
    df_clean = df.copy()
    
    if remove_outliers:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'interpolate',
    **kwargs
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with missing values
    method : str
        Method to handle missing values:
        - 'interpolate': Linear interpolation
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        - 'mean': Fill with column mean
        
    Returns:
    --------
    pd.DataFrame
        Data with missing values handled
    """
    df_filled = df.copy()
    
    if method == 'interpolate':
        df_filled = df_filled.interpolate(method='linear', **kwargs)
    elif method == 'forward_fill':
        df_filled = df_filled.fillna(method='ffill')
    elif method == 'backward_fill':
        df_filled = df_filled.fillna(method='bfill')
    elif method == 'mean':
        df_filled = df_filled.fillna(df_filled.mean())
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_filled


def seasonal_adjustment(
    series: pd.Series,
    period: int = 12,
    method: str = 'multiplicative'
) -> pd.Series:
    """
    Apply seasonal adjustment to time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to adjust
    period : int
        Seasonal period (12 for monthly data)
    method : str
        'additive' or 'multiplicative'
        
    Returns:
    --------
    pd.Series
        Seasonally adjusted series
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(
        series.dropna(),
        model=method,
        period=period,
        extrapolate_trend='freq'
    )
    
    if method == 'additive':
        adjusted = series - decomposition.seasonal
    else:
        adjusted = series / decomposition.seasonal
    
    return adjusted
