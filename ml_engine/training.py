"""
This module contains the training logic for the machine learning model. It calls functions to load data, preprocess it, and train the model using the specified parameters. 
"""
import yaml
from ml_engine.features import prepare_data
from ml_engine.models import select_model
from ml_engine.tracking import save_experiment
from ml_engine.constants import MODES, DEFAULT_CV_FOLDS, DEFAULT_SCORING
from sklearn.model_selection import cross_val_score
import pickle as pkl
from datetime import datetime

def train_model(experiment_path : str = None) -> dict:
    if not experiment_path:
        raise ValueError("Experiment file path is required.")
    if not experiment_path.endswith(".yaml"):
        raise ValueError("Invalid experiment file format. Please provide a YAML file.")
    
    with open(experiment_path, 'r') as file:
        config = yaml.safe_load(file)
    
    training = config.get("training") or {}
    cv_folds = training.get("cv_folds", DEFAULT_CV_FOLDS)
    scoring = training.get("scoring", DEFAULT_SCORING)
    if scoring not in MODES:
        raise ValueError("The specified scoring mode is not available. Refer to configs/README.md for model input instructions.")
    
    X, y = prepare_data(config)
    model = select_model(config)

    scores = cross_val_score(model, X, y, cv=cv_folds, scoring=MODES[scoring])

    model.fit(X, y)

    model_type = config.get("model").get("model_type")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"artifacts/models/{model_type}_{timestamp}.pkl"
    save_experiment(scores=scores, timestamp=timestamp, config=config, model=model, model_path=model_path)

    with open(model_path, "wb") as f:
        pkl.dump(model, f)

    return {"model_path": model_path}