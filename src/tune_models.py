import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    train = pd.read_csv("playground-series-s6e9/train.csv")
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    base_cols = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    is_sub = (train['Subsidy_Available'] == 'Yes').astype(float)
    inc_s = train['Annual_Income_USD'] / 100000.0

    df = train[base_cols].copy()
    df['income_per_car'] = train['Annual_Income_USD'] / train['Number_of_Cars_Owned']
    df['income_x_subsidy'] = inc_s * is_sub
    df['income_x_concern'] = inc_s * train['Environmental_Concern_Level']
    df['subsidy_x_concern'] = is_sub * train['Environmental_Concern_Level']
    
    city_mean_inc = train.groupby('City_Type')['Annual_Income_USD'].transform('mean')
    df['income_diff_city'] = train['Annual_Income_USD'] - city_mean_inc

    city_mean_comm = train.groupby('City_Type')['Daily_Commute_km'].transform('mean')
    df['commute_diff_city'] = train['Daily_Commute_km'] - city_mean_comm

    return df, y

def tune_lgb(df, y, tr_idx, va_idx):
    print("\n--- Tuning LightGBM on Validation Fold ---")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df_lgb = df.copy()
    for c in cat_cols:
        df_lgb[c] = df_lgb[c].astype('category')

    X_tr, y_tr = df_lgb.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = df_lgb.iloc[va_idx], y.iloc[va_idx]

    configs = [
        {'num_leaves': 31, 'max_depth': -1, 'colsample_bytree': 0.8, 'subsample': 0.8, 'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'lr': 0.03},
        {'num_leaves': 63, 'max_depth': 8, 'colsample_bytree': 0.8, 'subsample': 0.8, 'min_child_samples': 100, 'reg_alpha': 0.5, 'reg_lambda': 3.0, 'lr': 0.03},
        {'num_leaves': 127, 'max_depth': 9, 'colsample_bytree': 0.7, 'subsample': 0.8, 'min_child_samples': 200, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'lr': 0.03},
        {'num_leaves': 45, 'max_depth': 7, 'colsample_bytree': 0.85, 'subsample': 0.85, 'min_child_samples': 80, 'reg_alpha': 0.2, 'reg_lambda': 2.0, 'lr': 0.03},
    ]

    best_auc = 0
    best_cfg = None

    for i, cfg in enumerate(configs):
        tr_data = lgb.Dataset(X_tr, label=y_tr)
        va_data = lgb.Dataset(X_va, label=y_va, reference=tr_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': cfg['lr'],
            'num_leaves': cfg['num_leaves'],
            'max_depth': cfg['max_depth'],
            'colsample_bytree': cfg['colsample_bytree'],
            'subsample': cfg['subsample'],
            'min_child_samples': cfg['min_child_samples'],
            'reg_alpha': cfg['reg_alpha'],
            'reg_lambda': cfg['reg_lambda'],
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 4
        }
        model = lgb.train(params, tr_data, num_boost_round=1500, valid_sets=[va_data], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_va)
        auc = roc_auc_score(y_va, preds)
        print(f"LGB Config {i+1}: AUC={auc:.6f} (best_iter={model.best_iteration}) -> {cfg}")
        if auc > best_auc:
            best_auc = auc
            best_cfg = cfg
    print(f"Best LGB: AUC={best_auc:.6f}")
    return best_cfg

def tune_xgb(df, y, tr_idx, va_idx):
    print("\n--- Tuning XGBoost (GPU) on Validation Fold ---")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df_xgb = df.copy()
    for c in cat_cols:
        df_xgb[c] = df_xgb[c].astype('category')

    X_tr, y_tr = df_xgb.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = df_xgb.iloc[va_idx], y.iloc[va_idx]

    configs = [
        {'max_depth': 6, 'colsample_bytree': 0.8, 'subsample': 0.8, 'min_child_weight': 10, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'lr': 0.03},
        {'max_depth': 7, 'colsample_bytree': 0.8, 'subsample': 0.8, 'min_child_weight': 20, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'lr': 0.03},
        {'max_depth': 8, 'colsample_bytree': 0.7, 'subsample': 0.85, 'min_child_weight': 30, 'reg_alpha': 1.0, 'reg_lambda': 10.0, 'lr': 0.03},
        {'max_depth': 5, 'colsample_bytree': 0.85, 'subsample': 0.85, 'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'lr': 0.03},
    ]

    best_auc = 0
    best_cfg = None

    for i, cfg in enumerate(configs):
        m = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=cfg['lr'],
            max_depth=cfg['max_depth'],
            colsample_bytree=cfg['colsample_bytree'],
            subsample=cfg['subsample'],
            min_child_weight=cfg['min_child_weight'],
            reg_alpha=cfg['reg_alpha'],
            reg_lambda=cfg['reg_lambda'],
            tree_method="hist",
            device="cuda",
            enable_categorical=True,
            early_stopping_rounds=50,
            random_state=42
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, preds)
        print(f"XGB Config {i+1}: AUC={auc:.6f} (best_iter={m.best_iteration}) -> {cfg}")
        if auc > best_auc:
            best_auc = auc
            best_cfg = cfg
    print(f"Best XGB: AUC={best_auc:.6f}")
    return best_cfg

def tune_cb(df, y, tr_idx, va_idx):
    print("\n--- Tuning CatBoost (GPU) on Validation Fold ---")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    X_tr, y_tr = df.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = df.iloc[va_idx], y.iloc[va_idx]

    configs = [
        {'depth': 6, 'l2_leaf_reg': 5, 'lr': 0.05},
        {'depth': 7, 'l2_leaf_reg': 10, 'lr': 0.05},
        {'depth': 8, 'l2_leaf_reg': 15, 'lr': 0.04},
    ]

    best_auc = 0
    best_cfg = None

    for i, cfg in enumerate(configs):
        m = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=cfg['lr'],
            depth=cfg['depth'],
            l2_leaf_reg=cfg['l2_leaf_reg'],
            task_type="GPU",
            cat_features=cat_cols,
            random_seed=42,
            early_stopping_rounds=50,
            verbose=0
        )
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        preds = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, preds)
        print(f"CB Config {i+1}: AUC={auc:.6f} (best_iter={m.get_best_iteration()}) -> {cfg}")
        if auc > best_auc:
            best_auc = auc
            best_cfg = cfg
    print(f"Best CB: AUC={best_auc:.6f}")
    return best_cfg

def run():
    df, y = prepare_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(df, y))

    best_lgb = tune_lgb(df, y, tr_idx, va_idx)
    best_xgb = tune_xgb(df, y, tr_idx, va_idx)
    best_cb = tune_cb(df, y, tr_idx, va_idx)

    print("\n=== TUNING SUMMARY ===")
    print("Best LGB:", best_lgb)
    print("Best XGB:", best_xgb)
    print("Best CB:", best_cb)

if __name__ == "__main__":
    run()
