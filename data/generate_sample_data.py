"""
Script to generate sample inflation data.

This script creates synthetic South African inflation data for testing
and demonstration purposes.
"""

import numpy as np
import pandas as pd
import os


def generate_sa_inflation_data(
    start_date='2010-01-01',
    end_date='2023-12-31',
    freq='M',
    seed=42
):
    """
    Generate synthetic South African inflation data.
    
    Parameters
    ----------
    start_date : str
        Start date for the time series
    end_date : str
        End date for the time series
    freq : str
        Frequency ('M' for monthly, 'Q' for quarterly)
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        DataFrame with inflation data
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    
    # Time index
    time = np.arange(n)
    
    # Long-term trend (targeting 4.5% - midpoint of SARB target range)
    trend = 4.5 + 0.3 * np.sin(time / 30)
    
    # Seasonal component (stronger for monthly data)
    if freq == 'M':
        seasonal = 0.5 * np.sin(2 * np.pi * time / 12) + 0.3 * np.cos(4 * np.pi * time / 12)
    else:
        seasonal = 0.3 * np.sin(2 * np.pi * time / 4)
    
    # Cyclical component (business cycle effects)
    cyclical = 0.8 * np.sin(time / 48)
    
    # AR(2) component for persistence
    ar_component = np.zeros(n)
    ar_component[0] = np.random.normal(0, 0.3)
    ar_component[1] = 0.6 * ar_component[0] + np.random.normal(0, 0.3)
    for t in range(2, n):
        ar_component[t] = (0.6 * ar_component[t-1] + 
                          0.25 * ar_component[t-2] + 
                          np.random.normal(0, 0.3))
    
    # Combine all components
    inflation = trend + seasonal + cyclical + ar_component
    
    # Add occasional shocks (e.g., commodity price shocks)
    shock_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    shocks = np.zeros(n)
    shocks[shock_indices] = np.random.normal(0, 1.5, len(shock_indices))
    inflation += shocks
    
    # Ensure inflation stays realistic (between 2% and 8%)
    inflation = np.clip(inflation, 2.0, 8.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'inflation': inflation,
        'trend': trend,
        'seasonal': seasonal,
        'cyclical': cyclical
    })
    
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def generate_sarb_forecasts(
    actual_data,
    forecast_horizon=12,
    noise_level=0.3,
    seed=42
):
    """
    Generate synthetic SARB forecasts based on actual data.
    
    Parameters
    ----------
    actual_data : pd.DataFrame
        DataFrame with actual inflation data
    forecast_horizon : int
        Number of periods to forecast
    noise_level : float
        Amount of noise to add to forecasts
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        DataFrame with SARB forecasts
    """
    np.random.seed(seed)
    
    # Use last known value as anchor
    last_inflation = actual_data['inflation'].iloc[-1]
    
    # Assume SARB forecasts reversion to target midpoint (4.5%)
    target = 4.5
    
    # Generate forecast path
    forecasts = []
    dates = []
    
    last_date = actual_data['date'].iloc[-1]
    freq = pd.infer_freq(actual_data['date'])
    
    for h in range(1, forecast_horizon + 1):
        # Exponential reversion to target
        weight = np.exp(-h / 6)
        forecast = weight * last_inflation + (1 - weight) * target
        
        # Add noise
        forecast += np.random.normal(0, noise_level)
        
        forecasts.append(forecast)
        
        # Calculate date
        if freq == 'M':
            forecast_date = last_date + pd.DateOffset(months=h)
        else:
            forecast_date = last_date + pd.DateOffset(months=3*h)
        
        dates.append(forecast_date)
    
    # Create DataFrame
    df_forecast = pd.DataFrame({
        'date': dates,
        'sarb_forecast': forecasts
    })
    
    # Add confidence intervals
    df_forecast['sarb_lower'] = df_forecast['sarb_forecast'] - 1.0
    df_forecast['sarb_upper'] = df_forecast['sarb_forecast'] + 1.0
    
    return df_forecast


def main():
    """Generate and save sample data."""
    print("Generating sample South African inflation data...")
    
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate monthly data
    print("\n1. Generating monthly inflation data (2010-2023)...")
    df_monthly = generate_sa_inflation_data(
        start_date='2010-01-01',
        end_date='2023-12-31',
        freq='M'
    )
    
    # Save
    output_file = 'data/raw/sa_inflation_monthly.csv'
    df_monthly.to_csv(output_file, index=False)
    print(f"   Saved: {output_file}")
    print(f"   Shape: {df_monthly.shape}")
    print(f"   Date range: {df_monthly['date'].min()} to {df_monthly['date'].max()}")
    
    # Generate SARB forecasts
    print("\n2. Generating synthetic SARB forecasts...")
    df_sarb = generate_sarb_forecasts(df_monthly, forecast_horizon=12)
    
    # Save
    output_file_sarb = 'data/raw/sarb_forecasts.csv'
    df_sarb.to_csv(output_file_sarb, index=False)
    print(f"   Saved: {output_file_sarb}")
    print(f"   Shape: {df_sarb.shape}")
    
    # Display summary statistics
    print("\n3. Summary statistics:")
    print(df_monthly['inflation'].describe())
    
    print("\n" + "=" * 70)
    print("Sample data generation complete!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  - {output_file}")
    print(f"  - {output_file_sarb}")
    print("\nYou can now use these files for testing and demonstration.")


if __name__ == "__main__":
    main()
