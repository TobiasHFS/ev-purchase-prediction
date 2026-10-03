import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

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
for c in cat_cols:
    df[c] = df[c].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(skf.split(df, y))

X_tr, y_tr = df.iloc[tr_idx], y.iloc[tr_idx]
X_va, y_va = df.iloc[va_idx], y.iloc[va_idx]

# Model A: LightGBM (depth 7, leaves 45, seed 42)
tr_data = lgb.Dataset(X_tr, label=y_tr)
va_data = lgb.Dataset(X_va, label=y_va, reference=tr_data)
p_lgb1 = lgb.train({'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03, 'num_leaves': 45, 'max_depth': 7, 'colsample_bytree': 0.85, 'subsample': 0.85, 'min_child_samples': 80, 'reg_alpha': 0.2, 'reg_lambda': 2.0, 'verbose': -1, 'random_state': 42, 'n_jobs': 4}, tr_data, 1500, [va_data], callbacks=[lgb.early_stopping(50, False)]).predict(X_va)

# Model B: LightGBM (depth 8, leaves 63, seed 2026)
p_lgb2 = lgb.train({'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 8, 'colsample_bytree': 0.80, 'subsample': 0.80, 'min_child_samples': 100, 'reg_alpha': 0.5, 'reg_lambda': 3.0, 'verbose': -1, 'random_state': 2026, 'n_jobs': 4}, tr_data, 1500, [va_data], callbacks=[lgb.early_stopping(50, False)]).predict(X_va)

# Model C: XGBoost GPU (depth 5, seed 42)
m_xgb1 = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, colsample_bytree=0.85, subsample=0.85, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", device="cuda", enable_categorical=True, early_stopping_rounds=50, random_state=42)
m_xgb1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p_xgb1 = m_xgb1.predict_proba(X_va)[:, 1]

# Model D: XGBoost GPU (depth 6, seed 2026)
m_xgb2 = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=6, colsample_bytree=0.80, subsample=0.80, min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0, tree_method="hist", device="cuda", enable_categorical=True, early_stopping_rounds=50, random_state=2026)
m_xgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p_xgb2 = m_xgb2.predict_proba(X_va)[:, 1]

print(f"LGB1 (d7, l45): AUC={roc_auc_score(y_va, p_lgb1):.6f}")
print(f"LGB2 (d8, l63): AUC={roc_auc_score(y_va, p_lgb2):.6f}")
print(f"XGB1 (d5):      AUC={roc_auc_score(y_va, p_xgb1):.6f}")
print(f"XGB2 (d6):      AUC={roc_auc_score(y_va, p_xgb2):.6f}")

# Rank average of the 4 models
r1 = rankdata(p_lgb1) / len(p_lgb1)
r2 = rankdata(p_lgb2) / len(p_lgb2)
r3 = rankdata(p_xgb1) / len(p_xgb1)
r4 = rankdata(p_xgb2) / len(p_xgb2)

blend_4 = (r1 + r2 + r3 + r4) / 4.0
print(f">> 4-Model Multi-Depth Multi-Seed Rank Blend AUC: {roc_auc_score(y_va, blend_4):.6f}")
