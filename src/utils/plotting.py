"""
Visualization utilities for inflation forecasting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_forecast(
    actual: np.ndarray,
    forecast: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Inflation Forecast",
    save_path: Optional[str] = None
) -> None:
    """
    Plot forecast with prediction intervals.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecast values
    lower : np.ndarray, optional
        Lower prediction interval
    upper : np.ndarray, optional
        Upper prediction interval
    dates : pd.DatetimeIndex, optional
        Date index
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots()
    
    if dates is None:
        dates = np.arange(len(actual))
    
    # Plot actual
    ax.plot(dates[:len(actual)], actual, 'k-', label='Actual', linewidth=2)
    
    # Plot forecast
    forecast_dates = dates[len(actual):len(actual)+len(forecast)]
    ax.plot(forecast_dates, forecast, 'r--', label='Forecast', linewidth=2)
    
    # Plot prediction interval
    if lower is not None and upper is not None:
        ax.fill_between(
            forecast_dates,
            lower,
            upper,
            alpha=0.3,
            color='red',
            label='95% PI'
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Inflation (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_components(
    trend: np.ndarray,
    cycle: np.ndarray,
    irregular: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot state-space components.
    
    Parameters:
    -----------
    trend : np.ndarray
        Trend component
    cycle : np.ndarray
        Cyclical component
    irregular : np.ndarray
        Irregular component
    dates : pd.DatetimeIndex, optional
        Date index
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    if dates is None:
        dates = np.arange(len(trend))
    
    # Trend
    axes[0].plot(dates, trend, 'b-', linewidth=2)
    axes[0].set_ylabel('Trend')
    axes[0].set_title('Trend Component')
    axes[0].grid(True, alpha=0.3)
    
    # Cycle
    axes[1].plot(dates, cycle, 'g-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Cycle')
    axes[1].set_title('Cyclical Component')
    axes[1].grid(True, alpha=0.3)
    
    # Irregular
    axes[2].plot(dates, irregular, 'r-', linewidth=1, alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Irregular')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Irregular Component')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence(
    distances: List[np.ndarray],
    epsilons: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot ABC-SMC convergence.
    
    Parameters:
    -----------
    distances : List[np.ndarray]
        Distances for each population
    epsilons : List[float]
        Tolerance levels
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots()
    
    for i, (dist, eps) in enumerate(zip(distances, epsilons)):
        ax.violinplot(
            [dist],
            positions=[i],
            showmeans=True,
            showmedians=True
        )
        ax.axhline(y=eps, xmin=i/len(distances), xmax=(i+1)/len(distances),
                  color='r', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Population')
    ax.set_ylabel('Distance')
    ax.set_title('ABC-SMC Convergence')
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([f'Pop {i+1}' for i in range(len(distances))])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_posterior(
    samples: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot posterior distributions for parameters.
    
    Parameters:
    -----------
    samples : Dict[str, np.ndarray]
        Dictionary of parameter samples
    save_path : str, optional
        Path to save figure
    """
    n_params = len(samples)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i, (param, values) in enumerate(samples.items()):
        axes[i].hist(values, bins=30, density=True, alpha=0.7, edgecolor='black')
        axes[i].axvline(np.mean(values), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(values):.3f}')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Posterior: {param}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
