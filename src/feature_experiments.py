import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def get_base_features(df):
    f_df = df.copy()
    f_df['is_subsidy'] = (f_df['Subsidy_Available'] == 'Yes').astype(float)
    f_df['is_home_charge'] = (f_df['Home_Charging_Possible'] == 'Yes').astype(float)
    f_df['med_worry'] = (f_df['Range_Anxiety_Level'] == 'Medium').astype(float)
    f_df['high_worry'] = (f_df['Range_Anxiety_Level'] == 'High').astype(float)
    f_df['low_worry'] = (f_df['Range_Anxiety_Level'] == 'Low').astype(float)
    f_df['range_worry_score'] = f_df['high_worry'] * 3.0 + f_df['med_worry'] * 1.0
    f_df['income_scaled'] = f_df['Annual_Income_USD'] / 100000.0

    # Chris Deotte / Fable 5.1 recipe score
    f_df['recipe_buy_score'] = (
        1.2 * f_df['income_scaled'] +
        0.6 * f_df['Environmental_Concern_Level'] +
        2.0 * f_df['is_subsidy'] -
        1.0 * f_df['med_worry'] -
        3.0 * f_df['high_worry']
    )
    return f_df

def get_advanced_features(df):
    f_df = get_base_features(df)

    # 1. Total Charging & Ratios
    f_df['total_chargers'] = f_df['Charging_Stations_Near_Home'] + f_df['Charging_Stations_Near_Work']
    f_df['home_work_charger_ratio'] = f_df['Charging_Stations_Near_Home'] / (f_df['Charging_Stations_Near_Work'] + 1.0)
    
    # 2. Simpson's Paradox features: Home Charging vs Public Charging
    # If no home charging, public stations are essential
    f_df['no_home_charge'] = 1.0 - f_df['is_home_charge']
    f_df['dep_public_home'] = f_df['no_home_charge'] * f_df['Charging_Stations_Near_Home']
    f_df['dep_public_total'] = f_df['no_home_charge'] * f_df['total_chargers']
    
    # 3. Commute vs Charging / Range Anxiety
    f_df['commute_per_total_charger'] = f_df['Daily_Commute_km'] / (f_df['total_chargers'] + 1.0)
    f_df['commute_x_worry'] = f_df['Daily_Commute_km'] * f_df['range_worry_score']
    f_df['commute_x_no_home'] = f_df['Daily_Commute_km'] * f_df['no_home_charge']

    # 4. Economic Interactions
    f_df['income_per_car'] = f_df['Annual_Income_USD'] / f_df['Number_of_Cars_Owned']
    f_df['income_x_subsidy'] = f_df['income_scaled'] * f_df['is_subsidy']
    f_df['income_x_concern'] = f_df['income_scaled'] * f_df['Environmental_Concern_Level']
    f_df['subsidy_x_concern'] = f_df['is_subsidy'] * f_df['Environmental_Concern_Level']
    
    # 5. Non-linear distance to threshold (5.5)
    f_df['recipe_margin_to_threshold'] = f_df['recipe_buy_score'] - 5.5
    f_df['recipe_prob_sigmoid'] = 1.0 / (1.0 + np.exp(-1.5 * f_df['recipe_margin_to_threshold']))

    # 6. Categorical interactions
    f_df['city_home_charge'] = f_df['City_Type'] + '_' + f_df['Home_Charging_Possible']
    f_df['city_car_type'] = f_df['City_Type'] + '_' + f_df['Current_Car_Type']
    f_df['subsidy_worry'] = f_df['Subsidy_Available'] + '_' + f_df['Range_Anxiety_Level']

    return f_df

def evaluate_lgbm(X, y, feature_cols, cat_cols, name="Experiment", base_margin=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    
    X_proc = X[feature_cols].copy()
    for c in cat_cols:
        if c in X_proc.columns:
            X_proc[c] = X_proc[c].astype('category')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
        X_tr, y_tr = X_proc.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_proc.iloc[val_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

        if base_margin is not None:
            train_data.set_init_score(base_margin.iloc[train_idx])
            val_data.set_init_score(base_margin.iloc[val_idx])

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 4
        }

        eval_res = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(eval_res)]
        )

        val_pred = model.predict(X_va)
        oof[val_idx] = val_pred

    auc = roc_auc_score(y, oof)
    print(f"[{name}] 5-Fold OOF AUC: {auc:.6f}")
    return oof, auc

def run():
    print("Loading train.csv...")
    train = pd.read_csv("playground-series-s6e9/train.csv")
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    raw_features = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    raw_cats = [c for c in raw_features if not pd.api.types.is_numeric_dtype(train[c])]

    print("\n--- Exp 1: Raw features baseline ---")
    evaluate_lgbm(train, y, raw_features, raw_cats, name="Raw Features")

    print("\n--- Exp 2: Raw features + Recipe Clue ---")
    df_base = get_base_features(train)
    base_features = raw_features + ['recipe_buy_score']
    evaluate_lgbm(df_base, y, base_features, raw_cats, name="Raw + Recipe Clue")

    print("\n--- Exp 3: Advanced Engineered Features ---")
    df_adv = get_advanced_features(train)
    adv_features = [c for c in df_adv.columns if c not in ['id', 'Will_Buy_EV']]
    adv_cats = [c for c in adv_features if df_adv[c].dtype == 'object' or df_adv[c].dtype == 'str']
    evaluate_lgbm(df_adv, y, adv_features, adv_cats, name="Advanced Features")

    print("\n--- Exp 4: Advanced Features + Base Margin ---")
    # Base margin as logit of recipe prob
    margin = (df_adv['recipe_buy_score'] - 5.5) * 1.2
    evaluate_lgbm(df_adv, y, adv_features, adv_cats, name="Advanced + Base Margin", base_margin=margin)

if __name__ == "__main__":
    run()
