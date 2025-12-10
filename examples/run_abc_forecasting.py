"""
Example script demonstrating ABC inflation forecasting.

This script shows how to:
1. Load/generate inflation data
2. Fit ABC and baseline models
3. Generate forecasts
4. Compare performance
"""

import numpy as np
import pandas as pd
from scipy import stats

from abc_inflation.models import ABCInflationModel, ARIMAModel, SARIMAModel
from abc_inflation.utils import (
    generate_summary_statistics,
    create_train_test_split,
    forecast_metrics,
    compare_forecasts,
    plot_forecasts,
    plot_comparison,
    create_metrics_table
)


def simulate_inflation_data(n_periods=120, seed=42):
    """
    Simulate realistic inflation data.
    
    Parameters
    ----------
    n_periods : int
        Number of periods to simulate
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with inflation data
    """
    np.random.seed(seed)
    
    # Time index
    time = np.arange(n_periods)
    
    # Components
    trend = 4.5 + 0.5 * np.sin(time / 20)
    seasonal = 0.8 * np.sin(2 * np.pi * time / 12)
    
    # AR(1) component
    ar_component = np.zeros(n_periods)
    ar_component[0] = np.random.normal(0, 0.5)
    for t in range(1, n_periods):
        ar_component[t] = 0.7 * ar_component[t-1] + np.random.normal(0, 0.5)
    
    # Combine
    inflation = trend + seasonal + ar_component
    
    # Create DataFrame
    dates = pd.date_range(start='2014-01-01', periods=n_periods, freq='M')
    df = pd.DataFrame({'inflation': inflation}, index=dates)
    
    return df


def main():
    """Run the example."""
    print("=" * 70)
    print("ABC Inflation Forecasting Example")
    print("=" * 70)
    
    # 1. Generate data
    print("\n1. Generating data...")
    df = simulate_inflation_data(n_periods=120)
    print(f"   Generated {len(df)} observations")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Split data
    print("\n2. Splitting data...")
    test_size = 12
    train_data = df['inflation'].values[:-test_size]
    test_data = df['inflation'].values[-test_size:]
    print(f"   Training: {len(train_data)} observations")
    print(f"   Testing: {len(test_data)} observations")
    
    # 3. Fit ARIMA model
    print("\n3. Fitting ARIMA model...")
    arima_model = ARIMAModel(order=(2, 1, 2))
    arima_model.fit(train_data)
    arima_forecast, arima_lower, arima_upper = arima_model.forecast(test_size)
    print("   ARIMA model fitted successfully")
    
    # 4. Fit SARIMA model
    print("\n4. Fitting SARIMA model...")
    sarima_model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
    sarima_model.fit(train_data)
    sarima_forecast, sarima_lower, sarima_upper = sarima_model.forecast(test_size)
    print("   SARIMA model fitted successfully")
    
    # 5. Fit ABC model
    print("\n5. Fitting ABC model...")
    prior_distributions = {
        'phi': stats.uniform(0.3, 0.6),
        'sigma': stats.uniform(0.1, 1.0),
        'mu': stats.norm(4.5, 1.0),
    }
    
    abc_model = ABCInflationModel(
        prior_distributions=prior_distributions,
        summary_stats_fn=generate_summary_statistics,
        distance_metric='euclidean',
        n_particles=300,
        epsilon_schedule=[2.0, 1.0, 0.5, 0.25]
    )
    
    abc_model.fit(train_data, method='smc', verbose=True)
    abc_forecast, abc_lower, abc_upper = abc_model.forecast(
        n_steps=test_size,
        n_simulations=100
    )
    print("   ABC model fitted successfully")
    
    # 6. Display posterior summary
    print("\n6. ABC Posterior Summary:")
    posterior_summary = abc_model.get_posterior_summary()
    for param, stats_dict in posterior_summary.items():
        print(f"   {param}:")
        print(f"     Mean: {stats_dict['mean']:.4f}")
        print(f"     Std: {stats_dict['std']:.4f}")
        print(f"     95% CI: [{stats_dict['q025']:.4f}, {stats_dict['q975']:.4f}]")
    
    # 7. Compare forecasts
    print("\n7. Comparing forecasts...")
    metrics = compare_forecasts(
        actual=test_data,
        forecasts_dict={
            'ARIMA': arima_forecast,
            'SARIMA': sarima_forecast,
            'ABC': abc_forecast,
        }
    )
    
    # 8. Display results
    print("\n8. Forecast Performance Metrics:")
    metrics_df = create_metrics_table(metrics)
    print(metrics_df[['MAE', 'RMSE', 'MAPE', 'R2']])
    
    # 9. Identify best model
    print("\n9. Model Ranking (by RMSE):")
    ranking = metrics_df.sort_values('RMSE')[['RMSE', 'MAE', 'MAPE']]
    print(ranking)
    
    best_model = ranking.index[0]
    print(f"\n   Best performing model: {best_model}")
    
    # 10. Generate visualizations
    print("\n10. Generating visualizations...")
    
    # Plot forecasts
    forecasts_dict = {
        'ARIMA': (arima_forecast, arima_lower, arima_upper),
        'SARIMA': (sarima_forecast, sarima_lower, sarima_upper),
        'ABC': (abc_forecast, abc_lower, abc_upper),
    }
    
    all_data = np.concatenate([train_data, test_data])
    plot_forecasts(
        actual=all_data,
        forecasts=forecasts_dict,
        title='Inflation Forecasts Comparison',
        save_path='results/figures/forecasts_comparison.png'
    )
    print("   Saved: results/figures/forecasts_comparison.png")
    
    # Plot comparison
    plot_comparison(
        metrics_dict=metrics,
        metric_names=['RMSE', 'MAE', 'MAPE'],
        title='Model Performance Comparison',
        save_path='results/figures/metrics_comparison.png'
    )
    print("   Saved: results/figures/metrics_comparison.png")
    
    # Save results table
    metrics_df.to_csv('results/tables/forecast_metrics.csv')
    print("   Saved: results/tables/forecast_metrics.csv")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
