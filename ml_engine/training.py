"""
This module contains the training logic for the machine learning model. It calls functions to load data, preprocess it, and train the model using the specified parameters. 
"""
import yaml

def train_model(experiment_path : str = None) -> dict:
    with open(experiment_path, 'r') as file:
        config = yaml.safe_load(file)
        model = config['model']['model_type']
    return {"model": model}