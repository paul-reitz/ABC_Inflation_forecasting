"""
Data loader for inflation and economic indicators.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def load_inflation_data(filepath: str, date_col: str = 'date') -> pd.DataFrame:
    """
    Load inflation data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    date_col : str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with inflation data indexed by date
    """
    df = pd.read_csv(filepath, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    return df


def load_features(
    filepath: str, 
    feature_cols: Optional[List[str]] = None,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Load multivariate features for inflation forecasting.
    
    Parameters:
    -----------
    filepath : str
        Path to the features file
    feature_cols : List[str], optional
        Specific columns to load
    date_col : str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features indexed by date
    """
    df = pd.read_csv(filepath, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)
    
    if feature_cols:
        df = df[feature_cols]
        
    return df


def merge_datasets(
    inflation_data: pd.DataFrame,
    features_data: pd.DataFrame,
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merge inflation data with feature data.
    
    Parameters:
    -----------
    inflation_data : pd.DataFrame
        Inflation time series
    features_data : pd.DataFrame
        Feature time series
    how : str
        Type of merge ('inner', 'outer', 'left', 'right')
        
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    return pd.merge(
        inflation_data, 
        features_data,
        left_index=True,
        right_index=True,
        how=how
    )
