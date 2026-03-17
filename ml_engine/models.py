
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from ml_engine.preprocess import preprocess_data
from sklearn.model_selection import cross_val_score, GridSearchCV


def build_linear() -> Pipeline:
    model_linear = Pipeline([
        ("imputer", preprocess_data),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    return model_linear

def build_ridge(alpha_grid: list) -> GridSearchCV:
    ridge_pipeline = Pipeline([
        ("imputer", preprocess_data),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])
    model_ridge = GridSearchCV(
        ridge_pipeline,
        param_grid={"model__alpha": alpha_grid},
        cv=5,
        scoring="r2"
    )
    return model_ridge

def build_tree(random_state: int, max_depth: int) -> DecisionTreeRegressor:
    model_tree = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state
    )
    return model_tree


def select_model(config: dict):
    model_data = config.get("model")
    model_type = model_data.get("model_type")
    model_selection = ["linear", "ridge", "tree", "forest", "boosting"]
    if model_type not in model_selection:
        raise ValueError("The specified model is not available. Refer to README.md for model input instructions.")
    if model_type == "linear":
        model = build_linear()
    elif model_type == "ridge":
        alpha_grid = model_data.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])
        model = build_ridge(alpha_grid=alpha_grid)
    elif model_type == "tree":
        random_state = model_data.get("random_state", 42)
        max_depth = model_data.get("max_depth", 5)
        model = build_tree(random_state=random_state, max_depth=max_depth)
    
    return model