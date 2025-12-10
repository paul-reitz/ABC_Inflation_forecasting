"""
Feature engineering for inflation forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_lags(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        List of lag periods
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lagged features
    """
    df_lags = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lags[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df_lags


def compute_growth_rates(
    df: pd.DataFrame,
    columns: List[str],
    periods: List[int] = [1, 3, 12]
) -> pd.DataFrame:
    """
    Compute growth rates for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : List[str]
        Columns to compute growth rates for
    periods : List[int]
        Periods for growth rate calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with growth rate features
    """
    df_growth = df.copy()
    
    for col in columns:
        for period in periods:
            # Percentage change
            df_growth[f'{col}_growth{period}'] = df[col].pct_change(periods=period) * 100
            
            # Log difference
            df_growth[f'{col}_log_diff{period}'] = np.log(df[col]).diff(periods=period)
    
    return df_growth


def compute_rolling_statistics(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [3, 6, 12],
    stats: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    Compute rolling window statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : List[str]
        Columns to compute statistics for
    windows : List[int]
        Rolling window sizes
    stats : List[str]
        Statistics to compute ('mean', 'std', 'min', 'max')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling statistics
    """
    df_rolling = df.copy()
    
    for col in columns:
        for window in windows:
            for stat in stats:
                if stat == 'mean':
                    df_rolling[f'{col}_roll{window}_mean'] = df[col].rolling(window).mean()
                elif stat == 'std':
                    df_rolling[f'{col}_roll{window}_std'] = df[col].rolling(window).std()
                elif stat == 'min':
                    df_rolling[f'{col}_roll{window}_min'] = df[col].rolling(window).min()
                elif stat == 'max':
                    df_rolling[f'{col}_roll{window}_max'] = df[col].rolling(window).max()
    
    return df_rolling


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[tuple]
) -> pd.DataFrame:
    """
    Create interaction features between pairs of variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    feature_pairs : List[tuple]
        Pairs of features to interact
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with interaction features
    """
    df_interact = df.copy()
    
    for feat1, feat2 in feature_pairs:
        df_interact[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    
    return df_interact
