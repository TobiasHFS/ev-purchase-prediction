# EV Purchase Prediction

A tabular classification study for [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9). It contains feature engineering, boosting models and ensemble experiments for predicting electric vehicle purchases.

The project is an archived competition prototype. The script named `train_production_ensemble.py` is a training experiment, not a deployed service. Ensemble weights are selected using out-of-fold labels, so the resulting score is a model-selection score rather than an independent test result.

## Setup

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.lock.txt
```

Accept the competition rules on Kaggle and download `train.csv`, `test.csv` and `sample_submission.csv` into `playground-series-s6e9/` at the repository root.

```sh
python src/train_production_ensemble.py
```

The current XGBoost and CatBoost configurations expect a CUDA GPU. Adjust those model settings for CPU use before starting a full run. Training data, prediction files and submissions are excluded from Git.

## Layout

- `src/features.py`: feature construction
- `src/models.py`: model configurations
- `src/train_production_ensemble.py`: training and blending
- Other `src/` scripts: exploratory runs and diagnostics
- `tests/`: acceptance checks that require local competition data and generated predictions

The full training run and data-dependent acceptance checks have not been rerun for publication. MIT applies to the original code; Kaggle data has separate terms.

## Code sharing on Kaggle

The competition rules require publicly shared competition code to also be shared through its Kaggle discussion forum or notebooks. The repository link should be posted there by the owner. Competition data is excluded.

The dependency lock records the versions resolved for Python 3.12 on Windows on 2026-10-03. Other platforms may need a compatible local environment.
