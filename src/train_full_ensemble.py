import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def build_features(train_df, test_df):
    print("Building verified feature engineering pipeline...")
    
    # Preserve original copies untouched
    train = train_df.copy()
    test = test_df.copy()

    base_cols = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    
    # Binary helpers
    is_sub_tr = (train['Subsidy_Available'] == 'Yes').astype(float)
    is_sub_te = (test['Subsidy_Available'] == 'Yes').astype(float)
    
    inc_s_tr = train['Annual_Income_USD'] / 100000.0
    inc_s_te = test['Annual_Income_USD'] / 100000.0

    # 1. Economic interactions
    train['income_per_car'] = train['Annual_Income_USD'] / train['Number_of_Cars_Owned']
    test['income_per_car'] = test['Annual_Income_USD'] / test['Number_of_Cars_Owned']

    train['income_x_subsidy'] = inc_s_tr * is_sub_tr
    test['income_x_subsidy'] = inc_s_te * is_sub_te

    train['income_x_concern'] = inc_s_tr * train['Environmental_Concern_Level']
    test['income_x_concern'] = inc_s_te * test['Environmental_Concern_Level']

    train['subsidy_x_concern'] = is_sub_tr * train['Environmental_Concern_Level']
    test['subsidy_x_concern'] = is_sub_te * test['Environmental_Concern_Level']

    # 2. Relative City demographic statistics (computed on train to prevent leakage)
    city_income_map = train.groupby('City_Type')['Annual_Income_USD'].mean().to_dict()
    city_commute_map = train.groupby('City_Type')['Daily_Commute_km'].mean().to_dict()

    train['income_diff_city'] = train['Annual_Income_USD'] - train['City_Type'].map(city_income_map)
    test['income_diff_city'] = test['Annual_Income_USD'] - test['City_Type'].map(city_income_map)

    train['commute_diff_city'] = train['Daily_Commute_km'] - train['City_Type'].map(city_commute_map)
    test['commute_diff_city'] = test['Daily_Commute_km'] - test['City_Type'].map(city_commute_map)

    feature_cols = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]

    print(f"Total features ({len(feature_cols)}): {feature_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    return train, test, feature_cols, cat_cols

def train_and_predict():
    t_start = time.time()
    
    # 1. Load Data
    train_path = "playground-series-s6e9/train.csv"
    test_path = "playground-series-s6e9/test.csv"
    sample_sub_path = "playground-series-s6e9/sample_submission.csv"

    print("Loading data...")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    y = (train_raw['Will_Buy_EV'] == 'Yes').astype(int)
    n_train = len(train_raw)
    n_test = len(test_raw)

    train, test, feature_cols, cat_cols = build_features(train_raw, test_raw)

    # Prepare datasets for different frameworks
    # LightGBM / XGBoost categorical encoding
    train_cat = train[feature_cols].copy()
    test_cat = test[feature_cols].copy()
    for c in cat_cols:
        train_cat[c] = train_cat[c].astype('category')
        test_cat[c] = test_cat[c].astype('category')

    # HGBC integer categorical encoding
    train_hgb = train[feature_cols].copy()
    test_hgb = test[feature_cols].copy()
    for c in cat_cols:
        train_hgb[c] = train_hgb[c].astype('category').cat.codes
        test_hgb[c] = test_hgb[c].astype('category').cat.codes

    # 5-Fold Stratified Split
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)
    oof_cb = np.zeros(n_train)
    oof_hgb = np.zeros(n_train)

    test_pred_lgb = np.zeros(n_test)
    test_pred_xgb = np.zeros(n_test)
    test_pred_cb = np.zeros(n_test)
    test_pred_hgb = np.zeros(n_test)

    # -------------------------------------------------------------
    # Model 1: Tuned LightGBM
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING MODEL 1: Tuned LightGBM (5 Folds)")
    print("="*60)
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
        'n_jobs': 4
    }

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_cat, y)):
        f_t0 = time.time()
        X_tr, y_tr = train_cat.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = train_cat.iloc[va_idx], y.iloc[va_idx]

        tr_data = lgb.Dataset(X_tr, label=y_tr)
        va_data = lgb.Dataset(X_va, label=y_va, reference=tr_data)

        params = lgb_params.copy()
        params['random_state'] = 42 + fold

        m_lgb = lgb.train(
            params,
            tr_data,
            num_boost_round=2000,
            valid_sets=[va_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        va_pred = m_lgb.predict(X_va)
        oof_lgb[va_idx] = va_pred
        test_pred_lgb += m_lgb.predict(test_cat) / n_splits

        fold_auc = roc_auc_score(y_va, va_pred)
        print(f"LGB Fold {fold+1} AUC: {fold_auc:.6f} (trees: {m_lgb.best_iteration}, time: {time.time()-f_t0:.1f}s)")

    auc_lgb = roc_auc_score(y, oof_lgb)
    print(f">> FULL 5-FOLD OOF LightGBM AUC: {auc_lgb:.6f} (Total time: {time.time()-t0:.1f}s)")

    # -------------------------------------------------------------
    # Model 2: Tuned XGBoost (GPU Hist)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING MODEL 2: Tuned XGBoost GPU Hist (5 Folds)")
    print("="*60)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_cat, y)):
        f_t0 = time.time()
        X_tr, y_tr = train_cat.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = train_cat.iloc[va_idx], y.iloc[va_idx]

        m_xgb = xgb.XGBClassifier(
            n_estimators=2000,
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
            random_state=42 + fold
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        va_pred = m_xgb.predict_proba(X_va)[:, 1]
        oof_xgb[va_idx] = va_pred
        test_pred_xgb += m_xgb.predict_proba(test_cat)[:, 1] / n_splits

        fold_auc = roc_auc_score(y_va, va_pred)
        print(f"XGB Fold {fold+1} AUC: {fold_auc:.6f} (trees: {m_xgb.best_iteration}, time: {time.time()-f_t0:.1f}s)")

    auc_xgb = roc_auc_score(y, oof_xgb)
    print(f">> FULL 5-FOLD OOF XGBoost AUC: {auc_xgb:.6f} (Total time: {time.time()-t0:.1f}s)")

    # -------------------------------------------------------------
    # Model 3: Tuned CatBoost (GPU)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING MODEL 3: Tuned CatBoost GPU (5 Folds)")
    print("="*60)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        f_t0 = time.time()
        X_tr, y_tr = train.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_va, y_va = train.iloc[va_idx][feature_cols], y.iloc[va_idx]

        m_cb = cb.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            task_type="GPU",
            cat_features=cat_cols,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=0
        )
        m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        va_pred = m_cb.predict_proba(X_va)[:, 1]
        oof_cb[va_idx] = va_pred
        test_pred_cb += m_cb.predict_proba(test[feature_cols])[:, 1] / n_splits

        fold_auc = roc_auc_score(y_va, va_pred)
        print(f"CB Fold {fold+1} AUC: {fold_auc:.6f} (trees: {m_cb.get_best_iteration()}, time: {time.time()-f_t0:.1f}s)")

    auc_cb = roc_auc_score(y, oof_cb)
    print(f">> FULL 5-FOLD OOF CatBoost AUC: {auc_cb:.6f} (Total time: {time.time()-t0:.1f}s)")

    # -------------------------------------------------------------
    # Model 4: Tuned HistGradientBoostingClassifier
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING MODEL 4: Tuned HistGradientBoosting (5 Folds)")
    print("="*60)
    t0 = time.time()
    cat_mask = [f in cat_cols for f in feature_cols]
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_hgb, y)):
        f_t0 = time.time()
        X_tr, y_tr = train_hgb.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = train_hgb.iloc[va_idx], y.iloc[va_idx]

        m_hgb = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=45,
            min_samples_leaf=60,
            l2_regularization=1.5,
            categorical_features=cat_mask,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42 + fold
        )
        m_hgb.fit(X_tr, y_tr)

        va_pred = m_hgb.predict_proba(X_va)[:, 1]
        oof_hgb[va_idx] = va_pred
        test_pred_hgb += m_hgb.predict_proba(test_hgb)[:, 1] / n_splits

        fold_auc = roc_auc_score(y_va, va_pred)
        print(f"HGBC Fold {fold+1} AUC: {fold_auc:.6f} (trees: {m_hgb.n_iter_}, time: {time.time()-f_t0:.1f}s)")

    auc_hgb = roc_auc_score(y, oof_hgb)
    print(f">> FULL 5-FOLD OOF HGBC AUC: {auc_hgb:.6f} (Total time: {time.time()-t0:.1f}s)")

    # Save OOF and raw test predictions to disk
    oof_df = pd.DataFrame({
        'id': train_raw['id'],
        'Will_Buy_EV': y,
        'lgb': oof_lgb,
        'xgb': oof_xgb,
        'cb': oof_cb,
        'hgb': oof_hgb
    })
    oof_df.to_csv("src/oof_predictions.csv", index=False)

    test_pred_df = pd.DataFrame({
        'id': test_raw['id'],
        'lgb': test_pred_lgb,
        'xgb': test_pred_xgb,
        'cb': test_pred_cb,
        'hgb': test_pred_hgb
    })
    test_pred_df.to_csv("src/test_predictions.csv", index=False)
    print("\nSaved OOF and Test predictions to src/ directory.")

    # -------------------------------------------------------------
    # 5. ENSEMBLE OPTIMIZATION
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("ENSEMBLE OPTIMIZATION & EVALUATION")
    print("="*60)
    print(f"Individual Model 5-Fold OOF AUCs:")
    print(f"  Model 1 (LightGBM):           {auc_lgb:.6f}")
    print(f"  Model 2 (XGBoost GPU):        {auc_xgb:.6f}")
    print(f"  Model 3 (CatBoost GPU):       {auc_cb:.6f}")
    print(f"  Model 4 (HistGradientBoost):  {auc_hgb:.6f}")

    # Percentile Rank transformation
    rank_oof_lgb = rankdata(oof_lgb) / n_train
    rank_oof_xgb = rankdata(oof_xgb) / n_train
    rank_oof_cb = rankdata(oof_cb) / n_train
    rank_oof_hgb = rankdata(oof_hgb) / n_train

    rank_test_lgb = rankdata(test_pred_lgb) / n_test
    rank_test_xgb = rankdata(test_pred_xgb) / n_test
    rank_test_cb = rankdata(test_pred_cb) / n_test
    rank_test_hgb = rankdata(test_pred_hgb) / n_test

    # Simple Equal Rank Average
    equal_rank_oof = (rank_oof_lgb + rank_oof_xgb + rank_oof_cb + rank_oof_hgb) / 4.0
    auc_equal_rank = roc_auc_score(y, equal_rank_oof)
    print(f"\nEqual Rank Average (all 4) OOF AUC: {auc_equal_rank:.6f}")

    # Equal Probability Average
    equal_prob_oof = (oof_lgb + oof_xgb + oof_cb + oof_hgb) / 4.0
    auc_equal_prob = roc_auc_score(y, equal_prob_oof)
    print(f"Equal Probability Average OOF AUC:  {auc_equal_prob:.6f}")

    # Optimal Weight Optimization via Scipy
    print("\nOptimizing ensemble weights using Nelder-Mead to maximize OOF AUC...")
    OOF_ranks = np.column_stack([rank_oof_lgb, rank_oof_xgb, rank_oof_cb, rank_oof_hgb])

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(0, w)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        blend = OOF_ranks @ w
        # Negative AUC to minimize
        return -roc_auc_score(y, blend)

    init_weights = [0.35, 0.35, 0.15, 0.15]
    opt_res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4})
    best_w = np.maximum(0, opt_res.x)
    best_w = best_w / best_w.sum()

    opt_oof_blend = OOF_ranks @ best_w
    auc_opt_rank = roc_auc_score(y, opt_oof_blend)

    print("\n--- OPTIMAL WEIGHT RESULTS ---")
    print(f"LightGBM weight:           {best_w[0]:.4f}")
    print(f"XGBoost weight:            {best_w[1]:.4f}")
    print(f"CatBoost weight:           {best_w[2]:.4f}")
    print(f"HistGradientBoost weight:  {best_w[3]:.4f}")
    print(f">> OPTIMIZED ENSEMBLE 5-FOLD OOF AUC: {auc_opt_rank:.6f}")
    print(f"Total CV Improvement over single best model: {auc_opt_rank - max(auc_lgb, auc_xgb, auc_cb, auc_hgb):+.6f}")

    # -------------------------------------------------------------
    # 6. GENERATE FINAL PREDICTIONS
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPUTING FINAL TEST PREDICTIONS & SUBMISSION")
    print("="*60)
    TEST_ranks = np.column_stack([rank_test_lgb, rank_test_xgb, rank_test_cb, rank_test_hgb])
    final_test_preds = TEST_ranks @ best_w

    # Hard Zero-Boundary Post-Processing
    hard_zero_mask = (test_raw['Subsidy_Available'] == 'No') & (test_raw['Range_Anxiety_Level'] == 'High')
    n_hard_zero = hard_zero_mask.sum()
    print(f"Applying hard zero boundary constraint to {n_hard_zero} test samples (Subsidy=No & Worry=High)...")
    min_prob = final_test_preds.min()
    # Set them to 0.0 or lowest epsilon
    final_test_preds[hard_zero_mask] = np.minimum(final_test_preds[hard_zero_mask], min_prob * 0.1)

    # -------------------------------------------------------------
    # 7. WRITE AND VALIDATE SUBMISSION.CSV
    # -------------------------------------------------------------
    sub = pd.DataFrame({
        'id': test_raw['id'],
        'Will_Buy_EV': final_test_preds
    })

    sub_filename = "submission.csv"
    sub.to_csv(sub_filename, index=False)
    print(f"\nSaved submission to '{sub_filename}'")

    # Verification checks
    print("\n--- VERIFYING SUBMISSION FILE ---")
    assert os.path.exists(sub_filename), "Submission file does not exist!"
    assert sub.shape == sample_sub.shape, f"Shape mismatch! Expected {sample_sub.shape}, got {sub.shape}"
    assert list(sub.columns) == ['id', 'Will_Buy_EV'], f"Columns mismatch: {list(sub.columns)}"
    assert sub['id'].equals(test_raw['id']), "ID column does not match test set IDs exactly!"
    assert sub['Will_Buy_EV'].isnull().sum() == 0, "Submission contains NULL values!"
    assert not np.isinf(sub['Will_Buy_EV']).any(), "Submission contains INF values!"
    
    print(f"Row count: {len(sub):,} rows (matches sample_submission)")
    print(f"Will_Buy_EV summary stats:")
    print(sub['Will_Buy_EV'].describe())
    print("\nTop 5 predictions:")
    print(sub.head())
    print("\nBottom 5 predictions:")
    print(sub.tail())

    total_time = time.time() - t_start
    print(f"\n>>> ENTIRE PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES! <<<")

if __name__ == "__main__":
    train_and_predict()
