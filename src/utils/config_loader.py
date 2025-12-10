"""
Configuration file loader.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    config : Dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
    --------
    root : Path
        Project root path
    """
    return Path(__file__).parent.parent.parent


def get_data_path(filename: str) -> Path:
    """
    Get full path to data file.
    
    Parameters:
    -----------
    filename : str
        Data filename
        
    Returns:
    --------
    path : Path
        Full path to data file
    """
    return get_project_root() / 'data' / filename


def get_results_path(filename: str) -> Path:
    """
    Get full path to results file.
    
    Parameters:
    -----------
    filename : str
        Results filename
        
    Returns:
    --------
    path : Path
        Full path to results file
    """
    return get_project_root() / 'results' / filename
