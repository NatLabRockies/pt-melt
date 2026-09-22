# pt-melt

PT-MELT (PyTorch Machine Learning Toolbox) is a collection of architectures,
processing utilities, and machine-learning workflows built on PyTorch.

The goal of PT-MELT is to provide researchers with a flexible toolbox for
rapidly developing, training, evaluating, and deploying machine-learning
models across a range of applications.

## Installation

PT-MELT requires Python 3.11 or newer.

### Python package

Install the core package from a local checkout:

```bash
python -m pip install .
```

Install PT-MELT with optional hyperparameter-tuning support:

```bash
python -m pip install ".[hpo]"
```

PT-MELT can also be installed directly from GitHub:

```bash
python -m pip install \
  "ptmelt @ git+https://github.com/NatLabRockies/pt-melt.git"
```

For Ray Tune and Optuna support:

```bash
python -m pip install \
  "ptmelt[hpo] @ git+https://github.com/NatLabRockies/pt-melt.git"
```

These installation methods use the dependencies declared in `pyproject.toml`.
They do not use the repository's Pixi lock file.

## Reproducible development environment

PT-MELT uses Pixi for reproducible development and testing environments.

Install the complete development environment from the committed lock file:

```bash
pixi install -e dev --locked
```

Run the regression tests:

```bash
pixi run -e test tests
```

Run lint and formatting checks:

```bash
pixi run -e dev lint
pixi run -e dev format-check
```

Build the documentation:

```bash
pixi run -e doc docs
```

The available Pixi environments are:

- `default`: core PT-MELT runtime
- `hpo`: core runtime plus Ray Tune and Optuna
- `test`: test tooling plus HPO dependencies
- `doc`: Sphinx documentation environment
- `dev`: combined development environment

Pixi uses the committed `pixi.lock` to reproduce reviewed dependency
resolutions across supported platforms.

## Hyperparameter tuning

PT-MELT includes native helpers for Ray Tune workflows in
`ptmelt.utils.hp_tuning`. The builder supports `ann`, `resnet`, `bnn`, `rnn`,
`temporal_transformer`, and `vae` model configurations.

```python
from ray import tune

from ptmelt.utils.hp_tuning import run_ray_tune

result = run_ray_tune(
    train_dl=train_dl,
    val_dl=val_dl,
    base_config={
        "arch_type": "ann",
        "num_features": num_features,
        "num_outputs": num_outputs,
        "epochs": 25,
        "learning_rate": 1e-3,
        "loss_fn": "mse",
    },
    search_space={
        "width": tune.choice([32, 64, 128]),
        "depth": tune.randint(1, 5),
    },
    metric="val_loss",
    mode="min",
    num_samples=20,
)

print(result.best_config)
print(result.best_hyperparameters)
print(result.metric_details)
```

`run_ray_tune` returns the best configuration, searched hyperparameters,
selected metric details, and per-trial histories.
