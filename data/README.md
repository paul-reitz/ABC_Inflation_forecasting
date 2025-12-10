# Data Directory

This directory contains the data used for inflation forecasting analysis.

## Structure

- `raw/`: Original, unprocessed data files
- `processed/`: Cleaned and preprocessed data ready for analysis

## Data Sources

### Inflation Data
- Consumer Price Index (CPI) data
- Producer Price Index (PPI) data
- PCE Price Index data
- Source: Federal Reserve Economic Data (FRED), BLS, etc.

### Multivariate Features

1. **Macroeconomic Indicators**
   - GDP growth rate
   - Unemployment rate
   - Industrial production
   - Capacity utilization

2. **Monetary Indicators**
   - Money supply (M1, M2)
   - Interest rates (Federal Funds Rate, 10-year Treasury)
   - Credit aggregates

3. **External Sector**
   - Exchange rates (Trade-weighted dollar index)
   - Oil prices (WTI, Brent)
   - Commodity price indices

4. **Sentiment Indicators**
   - Consumer confidence index
   - Business sentiment surveys
   - Market-based inflation expectations

## Data Preprocessing

The data processing pipeline includes:

1. **Data Collection**: Download from public APIs/databases
2. **Cleaning**: Handle missing values, outliers
3. **Transformation**: Compute growth rates, seasonal adjustment
4. **Feature Engineering**: Create lags, moving averages, volatility measures
5. **Normalization**: Standardize features for model input

## Usage

Raw data should be placed in the `raw/` directory. Run the preprocessing scripts in `src/data_processing/` to generate processed data in the `processed/` directory.

```python
from src.data_processing import load_data, preprocess_data

# Load raw data
raw_data = load_data('raw/inflation_data.csv')

# Preprocess
processed_data = preprocess_data(raw_data)

# Save
processed_data.to_csv('processed/inflation_data_processed.csv')
```

## Data Availability

For replication purposes, processed data will be made available. Raw data sources are publicly accessible from the sources mentioned above.

## Notes

- Large data files (>100MB) are not committed to the repository
- Use the data download scripts in `scripts/` to obtain the data
- Ensure proper citation of data sources in the paper
