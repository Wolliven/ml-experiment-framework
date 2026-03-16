"""
This module contains the preprocessing logic for the model pipeline formation. It selects an appropriate strategy for processing the data based on the configuration.
"""
from sklearn.impute import SimpleImputer


def preprocess_data(config: dict):
    """
    Preprocess the data for training. This function can be expanded to include various preprocessing steps such as handling missing values, encoding categorical variables, and feature scaling.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - preprocessing
        - experiment parameters
    Returns
    -------
    Transformer or None
        A preprocessing transformer (e.g., SimpleImputer) or None if no preprocessing is required.
    """
    preprocessing_config = config.get('preprocessing', {})
    strategy = preprocessing_config.get('missing_strategy', "none")
    if strategy == "drop" or strategy == "none":
        return None
    else:
        return SimpleImputer(strategy=strategy)