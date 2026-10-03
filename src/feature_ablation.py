import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def evaluate_cv(df_features, y, name, cat_cols=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df_features))
    
    X = df_features.copy()
    if cat_cols is None:
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
    print(f"[{name:<35}] 5-Fold OOF AUC: {auc:.6f} | Delta: {auc - 0.941610:+.6f}")
    return auc

def run_ablation():
    train = pd.read_csv("playground-series-s6e9/train.csv")
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    base_cols = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    
    print("Baseline...")
    base_auc = evaluate_cv(train[base_cols], y, "Baseline (Raw Features)")

    # Group 1: Simpson's Paradox & Charging Interactions
    df1 = train[base_cols].copy()
    df1['no_home_charge'] = (df1['Home_Charging_Possible'] == 'No').astype(int)
    df1['total_chargers'] = df1['Charging_Stations_Near_Home'] + df1['Charging_Stations_Near_Work']
    df1['chargers_if_no_home'] = df1['no_home_charge'] * df1['total_chargers']
    df1['home_chargers_if_no_home'] = df1['no_home_charge'] * df1['Charging_Stations_Near_Home']
    df1['commute_per_charger'] = df1['Daily_Commute_km'] / (df1['total_chargers'] + 1.0)
    evaluate_cv(df1, y, "Group 1: Charging/Simpson")

    # Group 2: Boundary / Clipping Flags
    df2 = train[base_cols].copy()
    df2['is_income_min'] = (df2['Annual_Income_USD'] == 30000.0).astype(int)
    df2['is_commute_min'] = (df2['Daily_Commute_km'] == 5.0).astype(int)
    df2['is_zero_chargers_home'] = (df2['Charging_Stations_Near_Home'] == 0).astype(int)
    df2['is_zero_chargers_work'] = (df2['Charging_Stations_Near_Work'] == 0).astype(int)
    df2['is_zero_both'] = ((df2['Charging_Stations_Near_Home'] == 0) & (df2['Charging_Stations_Near_Work'] == 0)).astype(int)
    evaluate_cv(df2, y, "Group 2: Boundary/Clipping Flags")

    # Group 3: Economic Interactions
    df3 = train[base_cols].copy()
    is_sub = (df3['Subsidy_Available'] == 'Yes').astype(float)
    df3['income_per_car'] = df3['Annual_Income_USD'] / df3['Number_of_Cars_Owned']
    df3['income_x_subsidy'] = (df3['Annual_Income_USD'] / 100000.0) * is_sub
    df3['income_x_concern'] = (df3['Annual_Income_USD'] / 100000.0) * df3['Environmental_Concern_Level']
    df3['subsidy_x_concern'] = is_sub * df3['Environmental_Concern_Level']
    evaluate_cv(df3, y, "Group 3: Economic Interactions")

    # Group 4: Range Anxiety Interactions
    df4 = train[base_cols].copy()
    anx_map = {'Low': 0, 'Medium': 1, 'High': 3}
    df4['anxiety_num'] = df4['Range_Anxiety_Level'].map(anx_map)
    no_home = (df4['Home_Charging_Possible'] == 'No').astype(int)
    df4['anxiety_x_commute'] = df4['anxiety_num'] * df4['Daily_Commute_km']
    df4['anxiety_x_no_home'] = df4['anxiety_num'] * no_home
    evaluate_cv(df4, y, "Group 4: Range Anxiety Interactions")

    # Group 5: Group Statistics (Mean/Diff by City_Type)
    df5 = train[base_cols].copy()
    for col in ['Annual_Income_USD', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Daily_Commute_km']:
        city_mean = df5.groupby('City_Type')[col].transform('mean')
        city_std = df5.groupby('City_Type')[col].transform('std')
        df5[f'{col}_diff_city_mean'] = df5[col] - city_mean
        df5[f'{col}_z_city'] = (df5[col] - city_mean) / city_std
    evaluate_cv(df5, y, "Group 5: Group Aggregations (City)")

    # Group 6: Categorical Cross-Products
    df6 = train[base_cols].copy()
    df6['city_home_charge'] = df6['City_Type'] + '_' + df6['Home_Charging_Possible']
    df6['city_car_type'] = df6['City_Type'] + '_' + df6['Current_Car_Type']
    df6['subsidy_worry'] = df6['Subsidy_Available'] + '_' + df6['Range_Anxiety_Level']
    df6['home_charge_worry'] = df6['Home_Charging_Possible'] + '_' + df6['Range_Anxiety_Level']
    evaluate_cv(df6, y, "Group 6: Categorical Cross-Products")

if __name__ == "__main__":
    run_ablation()
