"""
Model construction utilities for regression experiments.

This module builds supported regression models from configuration
and provides a selector function to choose the requested model.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from ml_engine.preprocess import preprocess_data
from sklearn.model_selection import GridSearchCV


def build_linear(config: dict) -> Pipeline:
    """Build a linear regression pipeline with preprocessing and scaling."""
    pipeline = Pipeline([
        ("imputer", preprocess_data()),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    return pipeline

def build_ridge(config: dict) -> GridSearchCV:
    """Build a ridge regression pipeline wrapped in GridSearchCV."""
    model_data = config.get("model", {})
    validation_data = config.get("training", {})
    scoring = validation_data.get("scoring", "r2")
    alpha_grid = model_data.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_pipeline = Pipeline([
        ("imputer", preprocess_data()),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])
    search = GridSearchCV(
        ridge_pipeline,
        param_grid={"model__alpha": alpha_grid},
        cv=5,
        scoring=scoring
    )
    return search

def build_tree(config: dict) -> Pipeline:
    """Build a decision tree regressor from configuration."""
    model_data = config.get("model", {})
    random_state = model_data.get("random_state", 42)
    max_depth = model_data.get("max_depth", 5)
    model_tree = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state
    )
    pipeline = Pipeline([
        ("imputer", preprocess_data())
        ("model", model_tree)
    ])
    return pipeline

def build_forest(config: dict) -> Pipeline:
    """Build a random forest regressor from configuration."""
    model_data = config.get("model", {})
    n_estimators = model_data.get("n_estimators", 100)
    max_depth = model_data.get("max_depth", 10)
    random_state = model_data.get("random_state", 42)
    n_jobs = model_data.get("n_jobs", -1)
    model_forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    pipeline = Pipeline([
        ("imputer", preprocess_data())
        ("model", model_forest)
    ])
    return pipeline

def build_boosting(config: dict) -> Pipeline:
    """Build a gradient boosting regressor from configuration."""
    model_data = config.get("model", {})
    n_estimators = model_data.get("n_estimators", 100)
    max_depth = model_data.get("max_depth", 3)
    random_state = model_data.get("random_state", 42)
    learning_rate = model_data.get("learning_rate", 0.1)
    model_boosting = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        learning_rate=learning_rate
    )
    pipeline = Pipeline([
        ("imputer", preprocess_data())
        ("model", model_boosting)
    ])
    return pipeline

MODEL_REGISTRY = {
        "linear": build_linear,
        "ridge": build_ridge,
        "tree": build_tree,
        "forest": build_forest,
        "boosting": build_boosting
    }

def select_model(config: dict):
    """Select and build the model specified in the configuration."""
    model_data = config.get("model", {})
    model_type = model_data.get("model_type")
    if not model_type:
        raise ValueError("Missing 'model_type' in config['model'].")
    if model_type not in MODEL_REGISTRY:
        raise ValueError("The specified model is not available. Refer to configs/README.md for model input instructions.")    
    return MODEL_REGISTRY[model_type](config)