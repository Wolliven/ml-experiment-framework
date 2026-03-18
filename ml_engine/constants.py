DEFAULT_CV_FOLDS = 5
DEFAULT_SCORING = "r2"

IMPUTER_STRATEGIES = {
    "mean": "mean",
    "median": "median",
    "mode": "most_frequent",  # sklearn name
    "drop": None,
    "none": None,
}

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

MODES = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}