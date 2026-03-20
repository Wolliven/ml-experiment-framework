"""
Handles experiment tracking by saving model configuration, metrics,
and artifact information to a JSON file.
"""
import numpy as np
import json
from ml_engine.constants import TRACKED_PARAMS, DEFAULT_CV_FOLDS, DEFAULT_SCORING
from pathlib import Path

def save_experiment(timestamp: str, scores: np.ndarray, config: dict, model, model_path: str, config_path: str):
    """
    Saves experiment results including model parameters, training setup,
    metrics, and artifact metadata.

    Args:
        timestamp (str): Unique experiment timestamp.
        scores (np.ndarray): Cross-validation scores.
        config (dict): Experiment configuration.
        model: Trained model pipeline.
        model_path (str): Path where the model artifact is saved.
        config_path (str): Path to the configuration file.
    """
    results = {
        "model": {},
        "training": {},
        "preprocessing": {},
        "metrics": {},
        "artifact": {},
        "config_path": {}
    }

    params = model.named_steps["model"].get_params()
    model_type = config.get("model").get("model_type")

    results["model"] = {
        "model_type": model_type,
        **{k: params[k] for k in TRACKED_PARAMS[model_type]}
        }
    results["training"] = {
        "cv_folds": DEFAULT_CV_FOLDS,
        "scoring": DEFAULT_SCORING
    }
    results["preprocessing"] = config.get("preprocessing", {}).get("missing_strategy") or "none"
    results["metrics"] = {
        "cv_scores": scores.tolist(),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std())
    }
    results["artifact"] = {
        "model_path": model_path,
        "timestamp": timestamp
    }
    results["config_path"] = config_path

    Path("artifacts/experiments").mkdir(parents=True, exist_ok=True)
    experiment_path = f"artifacts/experiments/{model_type}_{timestamp}.json"
    with open(experiment_path, "w") as f:
        json.dump(results, f, indent=4)