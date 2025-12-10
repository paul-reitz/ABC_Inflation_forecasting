# ABC Inflation Forecasting

A research project on inflation forecasting using Approximate Bayesian Computation (ABC) and structural Bayesian state-space models with multivariate features.

## Project Overview

This repository contains the code, data, and analysis for a research paper on advanced inflation forecasting methods. The project combines:

- **Approximate Bayesian Computation (ABC)**: A simulation-based inference method for parameter estimation
- **Structural Bayesian State-Space Models**: Dynamic models that capture the temporal evolution of inflation
- **Multivariate Features**: Incorporation of multiple economic indicators to improve forecast accuracy

## Repository Structure

```
ABC_Inflation_forecasting/
├── data/                          # Data storage
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed/cleaned data
│   └── README.md                  # Data documentation
├── src/                           # Source code
│   ├── data_processing/           # Data loading and preprocessing
│   ├── models/                    # Model implementations
│   │   ├── state_space/          # State-space model components
│   │   ├── abc/                  # ABC algorithms
│   │   └── forecasting/          # Forecasting methods
│   ├── utils/                     # Utility functions
│   └── visualization/             # Plotting and visualization
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_abc_estimation.ipynb
│   └── 04_forecasting_analysis.ipynb
├── tests/                         # Unit tests
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_forecasting.py
├── paper/                         # Research paper
│   ├── main.tex                   # Main paper file
│   ├── sections/                  # Paper sections
│   ├── figures/                   # Paper figures
│   ├── tables/                    # Paper tables
│   └── references.bib             # Bibliography
├── results/                       # Analysis results
│   ├── figures/                   # Generated figures
│   ├── tables/                    # Generated tables
│   └── model_outputs/             # Model results
├── config/                        # Configuration files
│   ├── model_config.yaml
│   └── data_config.yaml
├── scripts/                       # Standalone scripts
│   ├── run_experiments.py
│   └── generate_results.py
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## Research Focus

### 1. State-Space Models
- Dynamic linear models (DLM)
- Unobserved components models (UCM)
- Structural time series models

### 2. ABC Methods
- Rejection ABC
- Sequential Monte Carlo ABC (SMC-ABC)
- Distance metrics for inflation forecasting

### 3. Multivariate Features
- GDP growth
- Unemployment rate
- Money supply
- Exchange rates
- Commodity prices
- Interest rates
- Consumer sentiment indices

### 4. Forecasting Evaluation
- Point forecasts
- Interval forecasts
- Probabilistic forecasts
- Comparison with benchmark models

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/paul-reitz/ABC_Inflation_forecasting.git
cd ABC_Inflation_forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Data Preparation**: See `notebooks/01_data_exploration.ipynb`
2. **Model Development**: See `notebooks/02_model_development.ipynb`
3. **ABC Estimation**: See `notebooks/03_abc_estimation.ipynb`
4. **Forecasting**: See `notebooks/04_forecasting_analysis.ipynb`

## Key Components

### State-Space Model Specification

The general state-space model has the form:

**Observation equation:**
```
y_t = Z_t * α_t + ε_t,  ε_t ~ N(0, H_t)
```

**State equation:**
```
α_t = T_t * α_{t-1} + R_t * η_t,  η_t ~ N(0, Q_t)
```

Where:
- `y_t` is the observed inflation
- `α_t` is the latent state vector (trend, cycle, seasonal components)
- `Z_t`, `T_t`, `R_t` are system matrices
- Multivariate features enter through the observation or state equation

### ABC Algorithm

The ABC framework estimates model parameters θ by:

1. Simulate data from the model with candidate parameters
2. Compare simulated data to observed data using summary statistics
3. Accept parameters if distance is below threshold ε
4. Iterate until convergence

## Paper Outline

The research paper is organized as follows:

1. **Introduction**: Motivation and research questions
2. **Literature Review**: ABC methods, state-space models, inflation forecasting
3. **Methodology**: Model specification, ABC algorithm, multivariate features
4. **Data**: Data sources, preprocessing, descriptive statistics
5. **Empirical Results**: Parameter estimates, forecast performance
6. **Discussion**: Interpretation and policy implications
7. **Conclusion**: Summary and future research

See `paper/` directory for LaTeX source files.

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{abc_inflation_forecasting,
  title={Inflation Forecasting with Approximate Bayesian Computation and Structural State-Space Models},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  note={In preparation}
}
```

## Contact

For questions or feedback, please contact: [Your Email]
