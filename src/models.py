"""
src/models.py
=============
Production Model Architectures & Configurations for Kaggle Playground Series S6E9
(Predicting Electric Vehicle Purchases).

Exposes model builders and configurations utilizing RTX 3060 Ti GPU (CUDA)
and multi-core Ryzen CPU (16 threads):
1. XGBoost GPU Hist (Multi-seed bagging across seeds 42 and 2026, depth 5)
2. LightGBM CPU (Multi-threaded n_jobs=16, multi-seed bagging across seeds 42 and 2026, depth 7, num_leaves 45)
3. CatBoost GPU (Native CUDA, depth=7, l2_leaf_reg=8, learning_rate=0.06)
4. HistGradientBoosting (CPU, max_iter=300, learning_rate=0.05, max_depth=7)
"""

from __future__ import annotations
from typing import Dict, Any, List
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingClassifier


def get_lgb_configs() -> List[Dict[str, Any]]:
    """Return tuned LightGBM parameter configurations for multi-seed bagging."""
    return [
        {
            'name': 'lgb_s42_d7',
            'params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.040,
                'max_depth': 7,
                'num_leaves': 45,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_samples': 80,
                'reg_alpha': 0.2,
                'reg_lambda': 2.0,
                'verbose': -1,
                'n_jobs': 16,
                'random_state': 42
            },
            'num_boost_round': 2500,
            'early_stopping_rounds': 50
        },
        {
            'name': 'lgb_s2026_d7',
            'params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.035,
                'max_depth': 7,
                'num_leaves': 45,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_samples': 80,
                'reg_alpha': 0.3,
                'reg_lambda': 2.5,
                'verbose': -1,
                'n_jobs': 16,
                'random_state': 2026
            },
            'num_boost_round': 2500,
            'early_stopping_rounds': 50
        }
    ]


def get_xgb_configs() -> List[Dict[str, Any]]:
    """Return tuned XGBoost GPU configurations for multi-seed bagging."""
    return [
        {
            'name': 'xgb_s42_d5',
            'params': {
                'n_estimators': 2500,
                'learning_rate': 0.040,
                'max_depth': 5,
                'min_child_weight': 5,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'device': 'cuda',
                'enable_categorical': True,
                'early_stopping_rounds': 50,
                'random_state': 42
            }
        },
        {
            'name': 'xgb_s2026_d5',
            'params': {
                'n_estimators': 2500,
                'learning_rate': 0.035,
                'max_depth': 5,
                'min_child_weight': 5,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.2,
                'reg_lambda': 1.5,
                'tree_method': 'hist',
                'device': 'cuda',
                'enable_categorical': True,
                'early_stopping_rounds': 50,
                'random_state': 2026
            }
        }
    ]


def get_catboost_config(cat_features: List[str]) -> Dict[str, Any]:
    """Return tuned CatBoost GPU configuration."""
    return {
        'name': 'cb_gpu_d7',
        'params': {
            'iterations': 2000,
            'learning_rate': 0.06,
            'depth': 7,
            'l2_leaf_reg': 8,
            'eval_metric': 'AUC',
            'task_type': 'GPU',
            'cat_features': cat_features,
            'early_stopping_rounds': 50,
            'random_seed': 42,
            'verbose': 0
        }
    }


def get_hgb_config(cat_features: List[str]) -> Dict[str, Any]:
    """Return tuned HistGradientBoosting CPU configuration."""
    return {
        'name': 'hgb_cpu_d7',
        'params': {
            'max_iter': 300,
            'learning_rate': 0.05,
            'max_depth': 7,
            'categorical_features': cat_features,
            'early_stopping': True,
            'n_iter_no_change': 25,
            'random_state': 42
        }
    }
