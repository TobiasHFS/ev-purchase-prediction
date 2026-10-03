import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("playground-series-s6e9/train.csv")
y = (train['Will_Buy_EV'] == 'Yes').astype(int)

# Load existing OOF to check correlation
oof_df = pd.read_csv("src/oof_predictions.csv")

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
for c in cat_cols:
    df[c] = df[c].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(skf.split(df, y))

X_tr, y_tr = df.iloc[tr_idx], y.iloc[tr_idx]
X_va, y_va = df.iloc[va_idx], y.iloc[va_idx]

tr_data = lgb.Dataset(X_tr, label=y_tr)
va_data = lgb.Dataset(X_va, label=y_va, reference=tr_data)

# Test ExtraTrees mode in LightGBM
params_et = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'extra_trees': True,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'verbose': -1,
    'random_state': 123,
    'n_jobs': 4
}
m_et = lgb.train(params_et, tr_data, num_boost_round=1500, valid_sets=[va_data], callbacks=[lgb.early_stopping(50, verbose=False)])
p_et = m_et.predict(X_va)
print(f"LGBM ExtraTrees AUC: {roc_auc_score(y_va, p_et):.6f}")

# Check correlation with standard LGB and XGB
p_lgb = oof_df['lgb'].iloc[va_idx]
p_xgb = oof_df['xgb'].iloc[va_idx]
print(f"Correlation with standard LGB: {np.corrcoef(p_et, p_lgb)[0, 1]:.5f}")
print(f"Correlation with standard XGB: {np.corrcoef(p_et, p_xgb)[0, 1]:.5f}")

# Test 50/50 blend of standard LGB + ExtraTrees LGB
blend_et = 0.5 * p_lgb + 0.5 * p_et
print(f"Standard LGB alone: {roc_auc_score(y_va, p_lgb):.6f}")
print(f"Standard LGB + ExtraTrees LGB blend: {roc_auc_score(y_va, blend_et):.6f}")

# Test 3-way blend: standard LGB + XGB + ExtraTrees LGB
from scipy.stats import rankdata
r_lgb = rankdata(p_lgb) / len(p_lgb)
r_xgb = rankdata(p_xgb) / len(p_xgb)
r_et = rankdata(p_et) / len(p_et)
r_3way = (r_lgb + r_xgb + r_et) / 3.0
print(f"3-Way Rank Blend (LGB + XGB + ExtraTrees): {roc_auc_score(y_va, r_3way):.6f}")
