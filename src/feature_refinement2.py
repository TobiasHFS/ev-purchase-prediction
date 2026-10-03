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
    print(f"[{name:<45}] 5-Fold OOF AUC: {auc:.6f} | Delta: {auc - 0.941578:+.6f}")
    return auc

def run():
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

    evaluate_cv(df, y, "Current Best: 4 Econ + City Income Diff")

    # Try adding city charging station differences
    df_charg = df.copy()
    city_mean_chg_h = train.groupby('City_Type')['Charging_Stations_Near_Home'].transform('mean')
    city_mean_chg_w = train.groupby('City_Type')['Charging_Stations_Near_Work'].transform('mean')
    df_charg['home_chg_diff_city'] = train['Charging_Stations_Near_Home'] - city_mean_chg_h
    df_charg['work_chg_diff_city'] = train['Charging_Stations_Near_Work'] - city_mean_chg_w
    evaluate_cv(df_charg, y, "+ City Charging Station Diffs")

    # Try adding commute difference by city
    df_comm = df.copy()
    city_mean_comm = train.groupby('City_Type')['Daily_Commute_km'].transform('mean')
    df_comm['commute_diff_city'] = train['Daily_Commute_km'] - city_mean_comm
    evaluate_cv(df_comm, y, "+ City Commute Diff")

    # Try adding age difference by city
    df_age = df.copy()
    city_mean_age = train.groupby('City_Type')['Age'].transform('mean')
    df_age['age_diff_city'] = train['Age'] - city_mean_age
    evaluate_cv(df_age, y, "+ City Age Diff")

    # Try combining all city relative features
    df_all_rel = df.copy()
    df_all_rel['commute_diff_city'] = train['Daily_Commute_km'] - city_mean_comm
    df_all_rel['age_diff_city'] = train['Age'] - city_mean_age
    df_all_rel['home_chg_diff_city'] = train['Charging_Stations_Near_Home'] - city_mean_chg_h
    evaluate_cv(df_all_rel, y, "+ All Relative City Demographics")

if __name__ == "__main__":
    run()
