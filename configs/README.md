Good instinct. When configs start growing, **a README in `/configs` prevents chaos** later.

Here’s a **clean, practical README.md** you can drop into the `configs/` folder.

---

# Experiment Configuration

This folder contains **YAML configuration files** used to run machine learning experiments.

Each configuration file defines:

* The **model and its hyperparameters**
* The **training procedure**
* The **dataset location**
* Optional **preprocessing steps**

Using YAML configs allows experiments to be **reproducible and easy to modify without changing code**.

---

# File Format

Each configuration file follows this structure:

```yaml
model:
  model_type: gradient_boosting
  learning_rate: 0.1
  n_estimators: 200
  max_depth: 5

training:
  cv_folds: 5
  scoring: r2

data:
  data_path: data/california_housing.csv
  target_column: MedHouseVal

preprocessing:
  missing_strategy: mean
```

---

# Sections

## model

Defines the model type and its hyperparameters.

Example:

```yaml
model:
  model_type: gradient_boosting
  learning_rate: 0.1
  n_estimators: 200
  max_depth: 5
```

### Supported Models

The following models can be specified in the configuration file under the `model` section.

```

Supported Models
│
├── Linear Models
│   ├── Linear Regression
│   └── Ridge Regression
│
└── Tree Models
    ├── Decision Tree
    ├── Random Forest
    └── Gradient Boosting

```

#### Linear Regression

```yaml
model:
  model_type: linear
```

No additional hyperparameters.


#### Ridge Regression

```yaml
model:
  model_type: ridge
  alpha_grid: [0.01, 0.1, 1.0, 10.0, 100.0]
```

Available hyperparameters:

* `alpha_grid`
  List of alpha values used for cross-validation. The best alpha is selected automatically.
  If not provided, the default grid `[0.01, 0.1, 1.0, 10.0, 100.0]` is used.


#### Decision Tree Regressor

```yaml
model:
  model_type: tree
  random_state: 42
  max_depth: 5
```

Available hyperparameters:

* `random_state`
  Seed used to initialize the random number generator.
  If not provided, the default state `42` is used.

* `max_depth`
  Maximum depth of the decision tree. Controls model complexity.
  If not provided, the default depth `5` is used.


#### Random Forest Regressor

```yaml
model:
  model_type: forest
  n_estimators: 100
  max_depth: 10
  random_state: 42
  n_jobs: -1
```

Available hyperparameters:

* `n_estimators`
  Number of trees in the forest.
  If not provided, the default number `100` is used.


* `max_depth`
  Maximum depth of each tree.
  If not provided, the default depth `10` is used.


* `random_state`
  Seed used to initialize the random number generator.
  If not provided, the default state `42` is used.


* `n_jobs`
  Number of CPU cores used for training.
  Use `-1` to use all available cores.
  If not provided, the default number `-1` is used.


#### Gradient Boosting Regressor

```yaml
model:
  model_type: boosting
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 3
  random_state: 42
```

Available hyperparameters:

* `n_estimators`
  Number of boosting stages.
  If not provided, the default number `100` is used.

* `learning_rate`
  Step size applied to each boosting stage.
  If not provided, the default rate `0.1` is used.


* `max_depth`
  Maximum depth of individual regression trees.
  If not provided, the default depth `3` is used.


* `random_state`
  Seed used to initialize the random number generator.
  If not provided, the default number `42` is used.


---

## training

Controls how the model is trained and evaluated.

Example:

```yaml
training:
  cv_folds: 5
  scoring: r2
```

Fields:

| Field      | Description                                    |
| ---------- | ---------------------------------------------- |
| `cv_folds` | Number of cross-validation folds               |
| `scoring`  | Evaluation metric used during cross-validation |

---

## data

Defines where the dataset is located and what column is the prediction target.

Example:

```yaml
data:
  data_path: data/california_housing.csv
  target_column: MedHouseVal
```

Fields:

| Field           | Description              |
| --------------- | ------------------------ |
| `data_path`     | Path to the dataset file |
| `target_column` | Column to be predicted   |

---

## preprocessing

Optional preprocessing steps applied before training.

Example:

```yaml
preprocessing:
  missing_strategy: mean
```

Fields:

| Field              | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `missing_strategy` | Strategy used for imputation (`mean`, `median`, `mode`, `drop`, `none`)     |

---

# Adding New Configurations

To create a new experiment:

1. Copy an existing config file.
2. Modify the model parameters or training settings.
3. Run the training script using the new config.

Example:

```
configs/
  baseline.yaml
  gradient_boosting_v1.yaml
  gradient_boosting_deep.yaml
```