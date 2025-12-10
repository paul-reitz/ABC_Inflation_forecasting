"""
Visualization Utilities

This module provides functions for visualizing inflation forecasts and
comparing model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_forecasts(
    actual: np.ndarray,
    forecasts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Inflation Forecasts",
    ylabel: str = "Inflation Rate (%)",
    save_path: Optional[str] = None
):
    """
    Plot actual values and multiple forecasts with confidence intervals.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    forecasts : dict
        Dictionary mapping model names to (mean, lower, upper) tuples
    dates : pd.DatetimeIndex, optional
        Date index for x-axis
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Determine x-axis
    if dates is None:
        x_actual = np.arange(len(actual))
    else:
        x_actual = dates[:len(actual)]
    
    # Plot actual values
    ax.plot(x_actual, actual, 'ko-', label='Actual', linewidth=2, markersize=4)
    
    # Colors for different models
    colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))
    
    for (model_name, (mean, lower, upper)), color in zip(forecasts.items(), colors):
        # Determine x-axis for forecast
        if dates is None:
            x_forecast = np.arange(len(actual), len(actual) + len(mean))
        else:
            # Assume forecasts start after actual data
            last_date = dates[len(actual)-1] if len(actual) <= len(dates) else dates[-1]
            x_forecast = pd.date_range(
                start=last_date,
                periods=len(mean)+1,
                freq=dates.freq
            )[1:]
        
        # Plot forecast
        ax.plot(x_forecast, mean, 'o-', label=f'{model_name}', 
                color=color, linewidth=2, markersize=4)
        ax.fill_between(x_forecast, lower, upper, alpha=0.2, color=color)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None
):
    """
    Plot comparison of forecast metrics across models.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping model names to their metrics
    metric_names : list, optional
        List of metric names to plot. If None, plots all metrics.
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    # Select metrics to plot
    if metric_names is None:
        metric_names = ['RMSE', 'MAE', 'MAPE', 'R2']
    
    # Filter to selected metrics
    df_plot = df[[m for m in metric_names if m in df.columns]]
    
    # Create subplots
    n_metrics = len(df_plot.columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, df_plot.columns):
        df_plot[metric].plot(kind='bar', ax=ax, color=plt.cm.Set2(np.arange(len(df_plot))))
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Plot residual diagnostics.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual observed values
    predicted : np.ndarray
        Predicted values
    model_name : str, optional
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(residuals, 'ko-', markersize=4)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'{model_name}: Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f'{model_name}: Residuals Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name}: Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    axes[1, 1].scatter(predicted, actual, alpha=0.6, s=50)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_title(f'{model_name}: Actual vs Predicted', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_time_series(
    data: pd.Series,
    title: str = "Inflation Time Series",
    ylabel: str = "Inflation Rate (%)",
    save_path: Optional[str] = None
):
    """
    Plot a time series.
    
    Parameters
    ----------
    data : pd.Series
        Time series data with DatetimeIndex
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data.index, data.values, 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_posterior_distributions(
    posterior_samples: List[Dict[str, float]],
    param_names: Optional[List[str]] = None,
    true_values: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot posterior distributions from ABC.
    
    Parameters
    ----------
    posterior_samples : list
        List of parameter dictionaries
    param_names : list, optional
        Parameter names to plot. If None, plots all.
    true_values : dict, optional
        True parameter values (for simulation studies)
    save_path : str, optional
        Path to save the figure
    """
    if param_names is None:
        param_names = list(posterior_samples[0].keys())
    
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, param_names):
        values = [sample[param] for sample in posterior_samples]
        
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, density=True)
        
        # Plot KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values)
        x_range = np.linspace(min(values), max(values), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Plot true value if available
        if true_values and param in true_values:
            ax.axvline(true_values[param], color='g', linestyle='--', 
                      linewidth=2, label='True Value')
        
        ax.set_title(f'Posterior: {param}', fontweight='bold')
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a formatted table of forecast metrics.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping model names to their metrics
    save_path : str, optional
        Path to save the table (CSV or LaTeX)
        
    Returns
    -------
    pd.DataFrame
        Formatted metrics table
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    if save_path:
        if save_path.endswith('.csv'):
            df.to_csv(save_path)
        elif save_path.endswith('.tex'):
            df.to_latex(save_path)
        else:
            df.to_csv(save_path)
    
    return df
