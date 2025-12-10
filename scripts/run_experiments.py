"""
Example script to run ABC estimation and forecasting.

This script demonstrates the complete workflow:
1. Load and preprocess data
2. Estimate model parameters using ABC-SMC
3. Generate forecasts
4. Evaluate performance
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data_processing import load_inflation_data, load_features
from src.models.state_space import UnobservedComponentsModel
from src.models.abc import ABCSMC, summary_statistics
from src.models.forecasting import point_forecast, interval_forecast
from src.utils import load_config, plot_forecast, plot_components


def main():
    """Run the complete ABC inflation forecasting pipeline."""
    
    print("=" * 80)
    print("ABC Inflation Forecasting - Example Experiment")
    print("=" * 80)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    config = load_config('config/model_config.yaml')
    data_config = load_config('config/data_config.yaml')
    
    # 2. Load data (placeholder - would load actual data)
    print("\n2. Loading data...")
    print("   (Using simulated data for demonstration)")
    
    # Simulate some data for demonstration
    np.random.seed(42)
    T = 240  # 20 years of monthly data
    
    # Simulate inflation with trend and cycle
    trend = 2.0 + 0.1 * np.arange(T) / T + np.random.normal(0, 0.1, T).cumsum()
    cycle = 0.8 * np.sin(2 * np.pi * np.arange(T) / 48) + np.random.normal(0, 0.2, T)
    inflation = trend + cycle + np.random.normal(0, 0.3, T)
    
    # Simulate features
    n_features = 5
    features = np.random.randn(T, n_features)
    
    print(f"   Data shape: {inflation.shape}")
    print(f"   Features shape: {features.shape}")
    
    # 3. Split data
    print("\n3. Splitting data...")
    train_size = int(0.7 * T)
    val_size = int(0.15 * T)
    
    train_data = inflation[:train_size]
    val_data = inflation[train_size:train_size + val_size]
    test_data = inflation[train_size + val_size:]
    
    print(f"   Train: {len(train_data)} observations")
    print(f"   Validation: {len(val_data)} observations")
    print(f"   Test: {len(test_data)} observations")
    
    # 4. Define model
    print("\n4. Setting up state-space model...")
    model = UnobservedComponentsModel(
        n_features=n_features,
        include_cycle=True,
        seasonal_period=None
    )
    print("   Model initialized")
    
    # 5. Define ABC simulation function
    print("\n5. Setting up ABC simulation...")
    
    def simulate_model(params):
        """Simulate data from the model."""
        model.build_matrices(params, X=features[:train_size])
        
        # Simulate (simplified - would use proper state-space simulation)
        simulated = np.random.normal(
            params.get('mu', 2.0),
            params.get('sigma_eps', 0.5),
            size=len(train_data)
        )
        return simulated
    
    # Define priors
    priors = {
        'sigma_eps': lambda: np.abs(np.random.normal(0.5, 0.2)),
        'sigma_mu': lambda: np.abs(np.random.normal(0.3, 0.1)),
        'sigma_nu': lambda: np.abs(np.random.normal(0.1, 0.05)),
        'sigma_gamma': lambda: np.abs(np.random.normal(0.5, 0.2)),
        'rho': lambda: np.random.uniform(0.5, 0.95),
    }
    
    # 6. Run ABC-SMC
    print("\n6. Running ABC-SMC estimation...")
    print("   (Using small settings for demonstration)")
    
    abc = ABCSMC(
        model=simulate_model,
        prior=priors,
        distance_metric='euclidean',
        summary_func=summary_statistics
    )
    
    epsilon_schedule = [2.0, 1.5, 1.0]  # Small for demo
    
    results = abc.run(
        observed_data=train_data,
        epsilon_schedule=epsilon_schedule,
        n_particles=100,  # Small for demo
        verbose=True
    )
    
    print(f"\n   ABC-SMC completed!")
    print(f"   Final population size: {len(results['particles'][-1])}")
    
    # 7. Extract parameter estimates
    print("\n7. Parameter estimates (posterior means):")
    for param in priors.keys():
        samples = abc.get_posterior_samples(param)
        mean = np.mean(samples)
        std = np.std(samples)
        print(f"   {param}: {mean:.4f} ± {std:.4f}")
    
    # 8. Fit model with estimated parameters
    print("\n8. Fitting model with estimated parameters...")
    param_estimates = {
        param: np.mean(abc.get_posterior_samples(param))
        for param in priors.keys()
    }
    
    model.build_matrices(param_estimates)
    model.fit(train_data, param_estimates)
    print("   Model fitted")
    
    # 9. Generate forecasts
    print("\n9. Generating forecasts...")
    forecast_horizon = 12
    
    forecasts = model.forecast(forecast_horizon)
    print(f"   Generated {forecast_horizon}-step ahead forecasts")
    
    # 10. Evaluate
    print("\n10. Forecast evaluation:")
    actual_test = test_data[:forecast_horizon]
    
    if len(actual_test) == forecast_horizon:
        mse = np.mean((actual_test - forecasts.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_test - forecasts.flatten()))
        
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
    else:
        print("   (Test data too short for full evaluation)")
    
    # 11. Save results
    print("\n11. Saving results...")
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save parameter estimates
    param_df = pd.DataFrame({
        'parameter': list(param_estimates.keys()),
        'estimate': list(param_estimates.values())
    })
    param_df.to_csv(results_dir / 'parameter_estimates.csv', index=False)
    print("   Saved parameter estimates")
    
    # Save forecasts
    forecast_df = pd.DataFrame({
        'horizon': range(1, forecast_horizon + 1),
        'forecast': forecasts.flatten()
    })
    forecast_df.to_csv(results_dir / 'forecasts.csv', index=False)
    print("   Saved forecasts")
    
    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
