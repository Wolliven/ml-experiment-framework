"""
This module contains the training logic for the machine learning model. It calls functions to load data, preprocess it, and train the model using the specified parameters. 
"""
import yaml
from ml_engine.features import prepare_data
from ml_engine.models import select_model
from sklearn.model_selection import cross_val_score
import pickle as pkl


MODES = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

def train_model(experiment_path : str = None) -> dict:
    if not experiment_path:
        raise ValueError("Experiment file path is required.")
    if not experiment_path.endswith(".yaml"):
        raise ValueError("Invalid experiment file format. Please provide a YAML file.")
    
    with open(experiment_path, 'r') as file:
        config = yaml.safe_load(file)
    
    cv_folds = config.get("training", {}).get("cv_folds", 5)
    scoring = config.get("training", {}).get("scoring", "r2")
    if scoring not in MODES:
        raise ValueError("The specified scoring mode is not available. Refer to configs/README.md for model input instructions.")
    model_path = "artifacts/models/test.pkl"
    
    X, y = prepare_data(config)
    model = select_model(config)

    scores = cross_val_score(model, X, y, cv=cv_folds, scoring=MODES[scoring])
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pkl.dump(model, f)

    return {"model_path": model_path}