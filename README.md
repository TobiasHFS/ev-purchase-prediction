# EV Purchase Prediction

Feature engineering, boosting models and ensemble experiments for [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9).

## Training

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.lock.txt
```

Download `train.csv`, `test.csv` and `sample_submission.csv` from Kaggle into `playground-series-s6e9/` at the repository root.

```sh
python src/train_production_ensemble.py
```

XGBoost and CatBoost currently use CUDA. Change their settings in `src/models.py` before training on a CPU.

`src/features.py` contains feature construction, and `src/models.py` contains model settings. The other scripts cover exploratory runs and blending. Tests in `tests/` need the competition data and generated predictions.

Ensemble weights are fitted on out-of-fold predictions. Scores from that step include the effect of selecting the blend weights.

[MIT license](LICENSE).
