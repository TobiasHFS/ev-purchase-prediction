import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def evaluate_cv(df_features, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df_features))
    
    X = df_features.copy()
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype('category')

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        tr_data = lgb.Dataset(X_tr, label=y_tr)
        va_data = lgb.Dataset(X_va, label=y_va, reference=tr_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42 + fold,
            'n_jobs': 4
        }

        model = lgb.train(
            params,
            tr_data,
            num_boost_round=1000,
            valid_sets=[va_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof[va_idx] = model.predict(X_va)

    auc = roc_auc_score(y, oof)
    print(f"[{name:<40}] 5-Fold OOF AUC: {auc:.6f} | Delta: {auc - 0.941578:+.6f}")
    return auc

def run_refinement():
    train = pd.read_csv("playground-series-s6e9/train.csv")
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    base_cols = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    is_sub = (train['Subsidy_Available'] == 'Yes').astype(float)
    inc_s = train['Annual_Income_USD'] / 100000.0

    # 1. Test each economic feature alone
    # 1a. income_per_car
    df_ipc = train[base_cols].copy()
    df_ipc['income_per_car'] = train['Annual_Income_USD'] / train['Number_of_Cars_Owned']
    evaluate_cv(df_ipc, y, "Economic 1: income_per_car")

    # 1b. income_x_subsidy
    df_ixs = train[base_cols].copy()
    df_ixs['income_x_subsidy'] = inc_s * is_sub
    evaluate_cv(df_ixs, y, "Economic 2: income_x_subsidy")

    # 1c. income_x_concern
    df_ixc = train[base_cols].copy()
    df_ixc['income_x_concern'] = inc_s * train['Environmental_Concern_Level']
    evaluate_cv(df_ixc, y, "Economic 3: income_x_concern")

    # 1d. subsidy_x_concern
    df_sxc = train[base_cols].copy()
    df_sxc['subsidy_x_concern'] = is_sub * train['Environmental_Concern_Level']
    evaluate_cv(df_sxc, y, "Economic 4: subsidy_x_concern")

    # 2. Test combinations
    # 2a. All 4 economic features
    df_all_econ = train[base_cols].copy()
    df_all_econ['income_per_car'] = train['Annual_Income_USD'] / train['Number_of_Cars_Owned']
    df_all_econ['income_x_subsidy'] = inc_s * is_sub
    df_all_econ['income_x_concern'] = inc_s * train['Environmental_Concern_Level']
    df_all_econ['subsidy_x_concern'] = is_sub * train['Environmental_Concern_Level']
    evaluate_cv(df_all_econ, y, "All 4 Economic Features")

    # 2b. All 4 Economic + City mean income
    df_econ_city = df_all_econ.copy()
    city_mean_inc = train.groupby('City_Type')['Annual_Income_USD'].transform('mean')
    df_econ_city['income_diff_city'] = train['Annual_Income_USD'] - city_mean_inc
    evaluate_cv(df_econ_city, y, "All 4 Econ + City Income Diff")

if __name__ == "__main__":
    run_refinement()
