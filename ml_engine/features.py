"""
This module contains functions for preparing the data for the model use, both in dropping missing rows (if specified by user) and in separating the training data from the target data.
"""
from ml_engine.data import load_data
import pandas as pd
import logging

def drop_missing(df : pd.DataFrame) -> pd.DataFrame:
    missing_rows = df.isna().any(axis=1).sum()
    if missing_rows > 0:
        logging.warning("%d row/s with missing values will be dropped.", missing_rows)
    df = df.dropna()
    return df

def prepare_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the data for training. Calls data validation, reads data, checks if user specified to drop missing values and does so.
    Separates training data from target data.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - experiment parameters
    Returns
    -------
    X, y
        Dataframes containing training data and target data, respectively.
    """
    data_config = config.get("data", {})
    df = load_data(data_config)

    strategy = config.get("preprocessing", {}).get("missing_strategy", None)
    if strategy == "drop":
        df = drop_missing(df)
    
    target_column = data_config.get("target_column", None)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    X = df.drop(columns=[target_column])
    if X.empty:
        raise ValueError("No feature columns remain after removing target column.")
    y = df[target_column]
    return X, y