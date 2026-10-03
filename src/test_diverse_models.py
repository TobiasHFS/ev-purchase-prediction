import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def test_models():
    train = pd.read_csv("playground-series-s6e9/train.csv")
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    features = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    cat_features = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]

    # Use 1 fold of StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(train, y))

    # Preprocessing for sklearn models (MLP, HGBC)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ]
    )
    X_train_dense = preprocessor.fit_transform(train.iloc[train_idx][features])
    X_val_dense = preprocessor.transform(train.iloc[val_idx][features])

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    preds = {}

    # 1. LightGBM
    print("\n--- Testing LightGBM ---")
    t0 = time.time()
    train_lgb = train[features].copy()
    for col in cat_features:
        train_lgb[col] = train_lgb[col].astype('category')
    
    m_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=4, verbose=-1)
    m_lgb.fit(train_lgb.iloc[train_idx], y_train, eval_set=[(train_lgb.iloc[val_idx], y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds['lgb'] = m_lgb.predict_proba(train_lgb.iloc[val_idx])[:, 1]
    print(f"LGBM AUC: {roc_auc_score(y_val, preds['lgb']):.6f} (time: {time.time() - t0:.1f}s)")

    # 2. XGBoost (GPU)
    print("\n--- Testing XGBoost (GPU) ---")
    t0 = time.time()
    train_xgb = train[features].copy()
    for col in cat_features:
        train_xgb[col] = train_xgb[col].astype('category')
    m_xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, tree_method="hist", device="cuda", enable_categorical=True, random_state=42, early_stopping_rounds=50)
    m_xgb.fit(train_xgb.iloc[train_idx], y_train, eval_set=[(train_xgb.iloc[val_idx], y_val)], verbose=False)
    preds['xgb'] = m_xgb.predict_proba(train_xgb.iloc[val_idx])[:, 1]
    print(f"XGB GPU AUC: {roc_auc_score(y_val, preds['xgb']):.6f} (time: {time.time() - t0:.1f}s)")

    # 3. CatBoost (GPU)
    print("\n--- Testing CatBoost (GPU) ---")
    t0 = time.time()
    m_cb = cb.CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, task_type="GPU", cat_features=cat_features, random_seed=42, early_stopping_rounds=50, verbose=0)
    m_cb.fit(train.iloc[train_idx][features], y_train, eval_set=(train.iloc[val_idx][features], y_val), verbose=False)
    preds['cb'] = m_cb.predict_proba(train.iloc[val_idx][features])[:, 1]
    print(f"CatBoost GPU AUC: {roc_auc_score(y_val, preds['cb']):.6f} (time: {time.time() - t0:.1f}s)")

    # 4. HistGradientBoosting
    print("\n--- Testing HistGradientBoosting ---")
    t0 = time.time()
    m_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42, categorical_features=[f in cat_features for f in features])
    # For HGBC, label encode categoricals
    train_hgb = train[features].copy()
    for col in cat_features:
        train_hgb[col] = train_hgb[col].astype('category').cat.codes
    m_hgb.fit(train_hgb.iloc[train_idx], y_train)
    preds['hgb'] = m_hgb.predict_proba(train_hgb.iloc[val_idx])[:, 1]
    print(f"HGBC AUC: {roc_auc_score(y_val, preds['hgb']):.6f} (time: {time.time() - t0:.1f}s)")

    # 5. Fast MLP (e.g. 128-64 hidden units)
    print("\n--- Testing MLPClassifier ---")
    t0 = time.time()
    m_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42, early_stopping=True)
    m_mlp.fit(X_train_dense, y_train)
    preds['mlp'] = m_mlp.predict_proba(X_val_dense)[:, 1]
    print(f"MLP AUC: {roc_auc_score(y_val, preds['mlp']):.6f} (time: {time.time() - t0:.1f}s)")

    # Correlations between models
    print("\n--- Predictions Rank Correlation Matrix ---")
    pred_df = pd.DataFrame(preds)
    print(pred_df.corr(method='spearman'))

    # Blends
    print("\n--- Simple Equal Blend (LGB + XGB + CB) ---")
    blend_trees = (preds['lgb'] + preds['xgb'] + preds['cb']) / 3.0
    print(f"Tree Ensemble AUC: {roc_auc_score(y_val, blend_trees):.6f}")

    print("\n--- All 5 Models Blend (LGB + XGB + CB + HGB + MLP) ---")
    blend_all = (preds['lgb'] + preds['xgb'] + preds['cb'] + preds['hgb'] + preds['mlp']) / 5.0
    print(f"All 5 Blend AUC: {roc_auc_score(y_val, blend_all):.6f}")

if __name__ == "__main__":
    test_models()
