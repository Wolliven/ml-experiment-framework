"""
This module contains the preprocessing logic for the model pipeline formation. It selects an appropriate strategy for processing the data based on the configuration.
"""
from sklearn.impute import SimpleImputer
from ml_engine.constants import IMPUTER_STRATEGIES

def preprocess_data(config: dict):
    """
    Selects and returns the preprocessing transformer based on configuration.
    """
    preprocessing_config = config.get('preprocessing', {})
    strategy = preprocessing_config.get("missing_strategy") or "none"
    if strategy not in IMPUTER_STRATEGIES:
        raise ValueError("Unsupported preprocessing strategy. Please refer to configs/README.md for details")
    if strategy in ("drop", "none"):
        return "passthrough"
    return SimpleImputer(strategy=IMPUTER_STRATEGIES[strategy])