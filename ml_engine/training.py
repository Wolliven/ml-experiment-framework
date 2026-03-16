"""
This module contains the training logic for the machine learning model. It calls functions to load data, preprocess it, and train the model using the specified parameters. 
"""
import yaml
from ml_engine.data import load_data

def train_model(experiment_path : str = None) -> dict:
    if not experiment_path.endswith(".yaml"):
        raise ValueError("Invalid experiment file format. Please provide a YAML file.")
    if not experiment_path:
        raise ValueError("Experiment file path is required.")
    
    with open(experiment_path, 'r') as file:
        config = yaml.safe_load(file)
    
    data_config = config.get('data', {})
    df = load_data(data_config)

    model = config['model']['model_type']
    return {"model": model, "data_path": data_config.get('data_path')}