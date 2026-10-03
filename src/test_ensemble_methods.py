import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def run_ensemble_test():
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

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(df, y))

    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 1. Train Tuned LightGBM
    print("Training Tuned LightGBM...")
    df_lgb = df.copy()
    for c in cat_cols:
        df_lgb[c] = df_lgb[c].astype('category')
    tr_data = lgb.Dataset(df_lgb.iloc[tr_idx], label=y_tr)
    va_data = lgb.Dataset(df_lgb.iloc[va_idx], label=y_va, reference=tr_data)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 45,
        'max_depth': 7,
        'colsample_bytree': 0.85,
        'subsample': 0.85,
        'min_child_samples': 80,
        'reg_alpha': 0.2,
        'reg_lambda': 2.0,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    m_lgb = lgb.train(lgb_params, tr_data, num_boost_round=1500, valid_sets=[va_data], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred_lgb = m_lgb.predict(df_lgb.iloc[va_idx])
    print(f"LGB AUC: {roc_auc_score(y_va, pred_lgb):.6f}")

    # 2. Train Tuned XGBoost (GPU)
    print("Training Tuned XGBoost (GPU)...")
    m_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        colsample_bytree=0.85,
        subsample=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=42
    )
    m_xgb.fit(df_lgb.iloc[tr_idx], y_tr, eval_set=[(df_lgb.iloc[va_idx], y_va)], verbose=False)
    pred_xgb = m_xgb.predict_proba(df_lgb.iloc[va_idx])[:, 1]
    print(f"XGB AUC: {roc_auc_score(y_va, pred_xgb):.6f}")

    # 3. Train Tuned CatBoost (GPU)
    print("Training Tuned CatBoost (GPU)...")
    m_cb = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        task_type="GPU",
        cat_features=cat_cols,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0
    )
    m_cb.fit(df.iloc[tr_idx], y_tr, eval_set=(df.iloc[va_idx], y_va), verbose=False)
    pred_cb = m_cb.predict_proba(df.iloc[va_idx])[:, 1]
    print(f"CB AUC:  {roc_auc_score(y_va, pred_cb):.6f}")

    # 4. Evaluate Ensembles
    print("\n--- Ensembling Comparisons ---")
    # Probability average
    prob_blend = (pred_lgb + pred_xgb + pred_cb) / 3.0
    print(f"Equal Probability Average AUC: {roc_auc_score(y_va, prob_blend):.6f}")

    # Percentile Rank Average
    rank_lgb = rankdata(pred_lgb) / len(pred_lgb)
    rank_xgb = rankdata(pred_xgb) / len(pred_xgb)
    rank_cb = rankdata(pred_cb) / len(pred_cb)
    rank_blend = (rank_lgb + rank_xgb + rank_cb) / 3.0
    print(f"Equal Rank Average AUC:        {roc_auc_score(y_va, rank_blend):.6f}")

    # Optimal weighted rank average (grid search)
    best_w_auc = 0
    best_weights = None
    for w_lgb in np.linspace(0.2, 0.6, 9):
        for w_xgb in np.linspace(0.2, 0.6, 9):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb >= 0.1:
                w_blend = w_lgb * rank_lgb + w_xgb * rank_xgb + w_cb * rank_cb
                sc = roc_auc_score(y_va, w_blend)
                if sc > best_w_auc:
                    best_w_auc = sc
                    best_weights = (w_lgb, w_xgb, w_cb)
    print(f"Optimal Weighted Rank AUC:     {best_w_auc:.6f} with weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")

if __name__ == "__main__":
    run_ensemble_test()
