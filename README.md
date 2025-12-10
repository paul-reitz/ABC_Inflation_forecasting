# ABC Inflation Forecasting

Approximate Bayesian Computation (ABC) for Inflation Forecasting - Beating SARB Forecasts

## Overview

This project implements a comprehensive Approximate Bayesian Computation (ABC) framework for inflation forecasting, with a specific focus on comparing performance against South African Reserve Bank (SARB) inflation forecasts. The implementation demonstrates how ABC methods can provide competitive or superior forecasts compared to traditional econometric approaches.

## Key Features

- **ABC Sequential Monte Carlo (SMC)**: Advanced ABC implementation with adaptive tolerance schedules
- **Baseline Models**: ARIMA and SARIMA models for comparison
- **Comprehensive Metrics**: Multiple evaluation metrics including MAE, RMSE, MAPE, coverage probability, and Diebold-Mariano tests
- **Visualization Tools**: Rich plotting functions for forecasts, residuals, and posterior distributions
- **SARB Comparison**: Framework for comparing against official SARB forecasts
- **Extensible Design**: Modular architecture for easy extension and customization

## Project Structure

```
ABC_Inflation_forecasting/
├── src/
│   └── abc_inflation/
│       ├── models/              # Model implementations
│       │   ├── abc_model.py     # ABC-SMC model
│       │   └── baseline_models.py  # ARIMA/SARIMA models
│       ├── utils/               # Utility functions
│       │   ├── metrics.py       # Forecast evaluation metrics
│       │   ├── data_loader.py   # Data loading and preprocessing
│       │   └── visualization.py # Plotting functions
│       └── data/                # Data handling
├── notebooks/                   # Jupyter notebooks
│   └── demo_abc_forecasting.ipynb  # Complete demonstration
├── tests/                       # Unit tests
├── data/                        # Data directory
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data
├── results/                     # Output directory
│   ├── figures/                 # Generated plots
│   └── tables/                  # Result tables
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/paul-reitz/ABC_Inflation_forecasting.git
cd ABC_Inflation_forecasting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Using the Jupyter Notebook

The easiest way to get started is with the demonstration notebook:

```bash
jupyter notebook notebooks/demo_abc_forecasting.ipynb
```

This notebook provides a complete walkthrough including:
- Data loading and preprocessing
- Fitting ABC and baseline models
- Generating and comparing forecasts
- Statistical evaluation
- Visualization

### Using the Python API

```python
import numpy as np
from scipy import stats
from abc_inflation.models import ABCInflationModel
from abc_inflation.utils import generate_summary_statistics

# Define prior distributions
prior_distributions = {
    'phi': stats.uniform(0.3, 0.6),      # AR coefficient
    'sigma': stats.uniform(0.1, 1.0),    # Noise std
    'mu': stats.norm(4.5, 1.0),          # Mean
}

# Initialize ABC model
model = ABCInflationModel(
    prior_distributions=prior_distributions,
    summary_stats_fn=generate_summary_statistics,
    n_particles=500,
    epsilon_schedule=[2.0, 1.0, 0.5, 0.25]
)

# Fit to data
model.fit(inflation_data, method='smc', verbose=True)

# Generate forecasts
forecast, lower, upper = model.forecast(n_steps=12, n_simulations=200)
```

## Methodology

### ABC Framework

The ABC approach works by:

1. **Prior Specification**: Define prior distributions for model parameters
2. **Simulation**: Generate synthetic data from the model with sampled parameters
3. **Summary Statistics**: Compute summary statistics that capture key features of the data
4. **Distance Calculation**: Measure distance between observed and simulated summary statistics
5. **Acceptance/Rejection**: Accept parameters that produce simulations close to observed data
6. **Sequential Refinement**: Use ABC-SMC to adaptively refine the posterior

### Model Structure

The implementation uses an AR(1) process as the base model, but the framework is extensible to:
- ARMA/ARIMA models
- State-space models
- Structural inflation models
- Models incorporating macroeconomic indicators

### Comparison with SARB

The framework evaluates ABC forecasts against:
- SARB official inflation forecasts
- ARIMA baseline models
- SARIMA seasonal models

## Evaluation Metrics

The package computes comprehensive forecast evaluation metrics:

- **Point Forecast Accuracy**: MAE, RMSE, MAPE, SMAPE
- **Interval Forecast Accuracy**: Coverage probability, interval score
- **Statistical Tests**: Diebold-Mariano test for forecast comparison
- **Model Selection**: AIC, BIC for baseline models

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=abc_inflation --cov-report=html
```

## Examples

### Example 1: Basic ABC Forecasting

```python
from abc_inflation.models import ABCInflationModel
from abc_inflation.utils import forecast_metrics

# Fit model
model = ABCInflationModel(prior_distributions, summary_stats_fn)
model.fit(train_data, method='smc')

# Forecast
forecast, lower, upper = model.forecast(n_steps=12)

# Evaluate
metrics = forecast_metrics(test_data, forecast, model_name="ABC")
print(f"RMSE: {metrics['RMSE']:.4f}")
```

### Example 2: Model Comparison

```python
from abc_inflation.utils import compare_forecasts, plot_comparison

# Generate forecasts from multiple models
forecasts_dict = {
    'ABC': abc_forecast,
    'ARIMA': arima_forecast,
    'SARIMA': sarima_forecast,
    'SARB': sarb_forecast,
}

# Compare
metrics = compare_forecasts(test_data, forecasts_dict)
plot_comparison(metrics, metric_names=['RMSE', 'MAE', 'MAPE'])
```

### Example 3: Posterior Analysis

```python
# Get posterior summary
summary = model.get_posterior_summary()
print(summary)

# Plot posterior distributions
from abc_inflation.utils import plot_posterior_distributions
plot_posterior_distributions(model.posterior_samples)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Data Sources

To replicate the analysis with actual data:

1. **South African Inflation Data**: Available from Stats SA or SARB
2. **SARB Forecasts**: Published in SARB Monetary Policy Review and Quarterly Bulletin

Place data files in the `data/raw/` directory and use the data loading utilities.

## References

### ABC Methodology
- Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian computation in population genetics.
- Sisson, S. A., Fan, Y., & Beaumont, M. (2018). Handbook of approximate Bayesian computation.

### Inflation Forecasting
- Stock, J. H., & Watson, M. W. (2007). Why has US inflation become harder to forecast?
- Faust, J., & Wright, J. H. (2013). Forecasting inflation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Paul Reitz

## Acknowledgments

- South African Reserve Bank for providing forecast data
- The ABC research community for methodological developments
