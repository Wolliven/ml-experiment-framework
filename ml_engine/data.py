"""
This module contains functions for loading data for machine learning experiments. It includes functionality to read datasets from specified paths and validate the loaded data.
"""
import pandas as pd
import logging
from pathlib import Path

def load_data(data_config: dict) -> pd.DataFrame:
    """
    Load and validate the dataset specified in the experiment configuration.

    Parameters
    ----------
    data_config : dict
        Configuration dictionary containing:
        - data_path
        - target_column

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    data_path = data_config.get("data_path")
    if not data_path:
        raise ValueError("Data path is missing in the experiment configuration")
    
    target_variable = data_config.get('target_column')
    if not target_variable:
        raise ValueError("Target column is missing in the experiment configuration.")

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")
        
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in the dataset.")
    if df[target_variable].isna().all():
        raise ValueError("Target column contains only missing values.")

    missing_rows = df.isna().any(axis=1).sum()

    if missing_rows > 0:
        logging.warning("%d rows contain missing values.", missing_rows)

    return df