# ML Experiment Framework

A small machine learning system designed to manage **reproducible training experiments, model artifacts, and evaluation results**.

The goal of this project is to move beyond individual model scripts and build a **structured ML experimentation pipeline** similar to those used in real machine learning workflows.

Instead of manually modifying training scripts for each experiment, this system allows experiments to be defined through **configuration files**, executed through a consistent training pipeline, and automatically logged for later analysis.

The project focuses on **ML engineering practices** rather than studying new algorithms.

---

# Project Objectives

This project aims to implement a simple but structured ML experimentation system that can:

1. Load and validate datasets
2. Build machine learning models from configuration files
3. Train and evaluate models using a consistent pipeline
4. Automatically log experiment parameters and results
5. Save trained models as reusable artifacts
6. Maintain a simple model registry for experiment tracking
7. Perform inference using previously trained models

The goal is to simulate the **core workflow of real ML systems** in a simplified environment.

---

# Motivation

Previous projects focused on understanding machine learning models and their behavior.

This project focuses on the **engineering side of machine learning**:

* reproducible experiments
* configuration-driven training
* structured pipelines
* artifact management
* experiment tracking

This project implements a **minimal version of these ideas** for educational purposes.

---

# Key Concepts

The system introduces several important ML engineering concepts.

### Design Principles

- separation of concerns (data, features, models, training, inference)
- reproducibility via configuration
- artifact-based workflows

### Configuration-Driven Experiments

Experiments are defined through configuration files instead of modifying training code.

Example:

```
configs/gradient_boosting.yaml
```

```
model: gradient_boosting
learning_rate: 0.05
n_estimators: 200
max_depth: 5

...
```

Training is executed through:

```
python run_experiment.py configs/gradient_boosting.yaml
```

This approach improves **reproducibility and experimentation discipline**.

---

### Experiment Tracking

Each experiment automatically records:

* model type
* hyperparameters
* evaluation metrics
* timestamp
* saved model path
* configuration path

Example:

```
artifacts/experiments/
linear_2026-03-18_16-03-40.json
```

These records allow experiments to be **reproduced and compared**.

---

### Model Artifacts

Trained models are saved as reusable artifacts.

Example:

```
artifacts/models/
linear_2026-03-18_16-03-40.pkl
```

Each model is associated with the experiment that produced it. Models are saved together with required metadata (e.g., feature names) to ensure safe and consistent inference.

---

### Reproducible Training Pipeline

The system separates the main stages of machine learning into reusable modules:

* data loading
* feature preparation
* model construction
* training
* evaluation
* experiment logging

This structure mirrors how real ML training pipelines are organized.

---

# Project Structure

```
ml-experiment-framework/
│
├── ml_engine/                 # core ML system code
│   ├── data.py                # dataset loading
│   ├── features.py            # preprocessing
│   ├── models.py              # model construction
│   ├── training.py            # training & evaluation
│   ├── tracking.py            # experiment logging
│   ├── preprocess.py          # preprocessing pipeline
│   ├── constants.py           # project constants
│   └── predict.py             # inference logic
│
├── scripts/                   # CLI entrypoints
│   ├── run_experiment.py
│   └── predict.py
│
├── configs/                   # experiment definitions
│   └── README.md              # experiment configuration explanation
│
├── artifacts/                 # generated outputs
│   ├── models/                # generated models
│   ├── predictions/           # generated predictions
│   └── experiments/           # experiment metadata
│
├── data/                      # datasets
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Example Workflow

Run a training experiment:

```
python scripts/run_experiment.py configs/experiment_01.yaml
```

The system will automatically:

1. Load dataset
2. Build the model from configuration
3. Train and evaluate the model
4. Save evaluation metrics
5. Save the trained model
6. Record the experiment

### Inference Input / Output

- Input: CSV or JSON
- Output: CSV (default) or JSON

Inference can then be performed with:

```
python -m scripts.predict input.json --model artifacts/models/model_v3.pkl --out predictions.csv
```

The input format is inferred from file extensions.

---

# Scope

This project focuses on **ML experimentation infrastructure**.

Out of scope:

* deep learning
* web frontends
* cloud deployment
* distributed training

The goal is to build a **clean and reproducible ML experimentation workflow**.

---

# Requirements

Python 3.9+

Libraries:

```
pandas
numpy
scikit-learn
pyyaml
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Philosophy

This project prioritizes:

* reproducibility
* clean pipeline structure
* experiment discipline
* simple but realistic ML workflows

The goal is to practice **building machine learning systems**, not just training models.