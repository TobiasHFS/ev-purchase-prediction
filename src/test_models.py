import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import sklearn
import numpy as np

print("Imports succeeded!")
X = np.random.randn(100, 5)
y = (np.random.randn(100) > 0).astype(int)

# Test LightGBM
m_lgb = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
m_lgb.fit(X, y)
print("LightGBM fit succeeded!")

# Test XGBoost
m_xgb = xgb.XGBClassifier(n_estimators=10, verbosity=0)
m_xgb.fit(X, y)
print("XGBoost fit succeeded!")

try:
    m_xgb_gpu = xgb.XGBClassifier(n_estimators=10, tree_method="hist", device="cuda", verbosity=0)
    m_xgb_gpu.fit(X, y)
    print("XGBoost GPU fit succeeded!")
except Exception as e:
    print(f"XGBoost GPU failed: {e}")

# Test CatBoost
m_cb = cb.CatBoostClassifier(iterations=10, verbose=0)
m_cb.fit(X, y)
print("CatBoost fit succeeded!")

try:
    m_cb_gpu = cb.CatBoostClassifier(iterations=10, task_type="GPU", verbose=0)
    m_cb_gpu.fit(X, y)
    print("CatBoost GPU fit succeeded!")
except Exception as e:
    print(f"CatBoost GPU failed: {e}")
