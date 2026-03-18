
import numpy as np

TRACKED_PARAMS = {
    "linear": [],
    
    "ridge": [
        "alpha"  # selected best alpha after GridSearchCV
    ],
    
    "tree": [
        "max_depth",
        "random_state"
    ],
    
    "forest": [
        "n_estimators",
        "max_depth",
        "random_state",
        "n_jobs"
    ],
    
    "boosting": [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "random_state"
    ],
}

def save_experiment(timestamp: str, scores: np.ndarray, config: dict, model):
    model_data = config.get("model")
    training_data = config.get("training")
    training = config.get("training") or {}
    cv_folds = training.get("cv_folds", 5)
    scoring = training.get("scoring", "r2")
    results = {
        "model": {},
        "training": {},
        "preprocessing": {},
        "metrics": {},
        "artifact": {}
    }

    params = model.named_steps["model"].get_params()
    model_name = config.get("model").get("model_type")
    results["model"] = {
        "model_type": model_name,
        **{k: params[k] for k in TRACKED_PARAMS[model_name]}
        }
    results["preprocessing"] = config.get("preprocessing", {}).get("missing_strategy") or "none"
    print(results)