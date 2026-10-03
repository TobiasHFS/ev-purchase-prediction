"""
src/train_production_ensemble.py
================================
Production Training Pipeline for Kaggle Playground Series S6E9
(Predicting Electric Vehicle Purchases).

Key Architectural Pillars:
1. Leak-free Feature Engineering:
   - Uses `create_features` from `src.features` strictly inside each training fold split.
   - Fits fold statistics on training partition and maps to validation and test sets.
2. 5-Fold Stratified K-Fold:
   - `n_splits=5, shuffle=True, random_state=42` matching the competition evaluation.
3. Multi-Model Architecture & Hardware Exploitation:
   - XGBoost GPU: Native CUDA Hist (`tree_method='hist', device='cuda'`), multi-seed bagging (seeds 42 & 2026).
   - LightGBM: Multi-threaded CPU (`n_jobs=16`), multi-seed bagging (seeds 42 & 2026).
   - CatBoost GPU: Native CUDA (`task_type='GPU'`, depth=7, l2=8, lr=0.06).
   - HistGradientBoosting: CPU (`max_iter=300`, lr=0.05, max_depth=7).
4. Out-of-Fold (OOF) & Test Matrix Generation:
   - Generates and writes `src/oof_predictions.csv` (668,665 rows) with columns:
     `id,Will_Buy_EV,lgb,xgb,cb,hgb,lgb_s42,lgb_s2026,xgb_s42,xgb_s2026`
   - Generates and writes `src/test_predictions.csv` (286,571 rows) with columns:
     `id,lgb,xgb,cb,hgb,lgb_s42,lgb_s2026,xgb_s42,xgb_s2026`
   - Produces rank-optimized ensemble deliverable `submission.csv` with zero-boundary post-processing.
"""

from __future__ import annotations
import os
import sys
import time
import warnings
from typing import Dict, Any, Tuple, List
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

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import create_features
from src.models import get_lgb_configs, get_xgb_configs, get_catboost_config, get_hgb_config

warnings.filterwarnings('ignore')


def train_production_ensemble() -> None:
    """Execute the complete production 5-fold training pipeline and generate deliverables."""
    t_start = time.time()
    print("*" * 80)
    print(" KAGGLE PLAYGROUND SERIES S6E9  -  PRODUCTION ENSEMBLE TRAINING PIPELINE")
    print(f" Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Python:    {sys.executable}")
    print("*" * 80)

    # 1. Load Raw Data
    train_path = os.path.join(ROOT_DIR, "playground-series-s6e9", "train.csv")
    test_path = os.path.join(ROOT_DIR, "playground-series-s6e9", "test.csv")
    sample_sub_path = os.path.join(ROOT_DIR, "playground-series-s6e9", "sample_submission.csv")

    print("\n[Step 1/6] Loading raw competition datasets...")
    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(test_path), f"Missing {test_path}"

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    n_train = len(train_raw)
    n_test = len(test_raw)
    assert n_train == 668665, f"Unexpected train rows: {n_train}"
    assert n_test == 286571, f"Unexpected test rows: {n_test}"

    y = (train_raw['Will_Buy_EV'] == 'Yes').astype(int)
    print(f"  Train: {n_train:,} rows, positive rate = {y.mean():.4%}")
    print(f"  Test:  {n_test:,} rows")

    # Fixed categorical column list
    cat_cols = [
        'Gender',
        'City_Type',
        'Current_Car_Type',
        'Home_Charging_Possible',
        'Subsidy_Available',
        'Range_Anxiety_Level'
    ]

    # Model configurations from src.models
    lgb_cfgs = get_lgb_configs()
    xgb_cfgs = get_xgb_configs()
    cb_cfg = get_catboost_config(cat_cols)
    hgb_cfg = get_hgb_config(cat_cols)

    # Prediction allocation arrays
    oof_lgb_s42 = np.zeros(n_train)
    oof_lgb_s2026 = np.zeros(n_train)
    oof_xgb_s42 = np.zeros(n_train)
    oof_xgb_s2026 = np.zeros(n_train)
    oof_cb = np.zeros(n_train)
    oof_hgb = np.zeros(n_train)

    test_lgb_s42 = np.zeros(n_test)
    test_lgb_s2026 = np.zeros(n_test)
    test_xgb_s42 = np.zeros(n_test)
    test_xgb_s2026 = np.zeros(n_test)
    test_cb = np.zeros(n_test)
    test_hgb = np.zeros(n_test)

    # 2. 5-Fold Stratified K-Fold Split
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\n[Step 2/6] Beginning 5-Fold Stratified K-Fold Cross-Validation...")
    print(f"  Splits: {n_splits} folds | Hardware: NVIDIA RTX 3060 Ti GPU (CUDA) + Ryzen 16-thread CPU\n")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_raw, y)):
        f_start = time.time()
        print(f"\n{'='*75}")
        print(f"=== TRAINING FOLD {fold + 1}/{n_splits} (Train: {len(tr_idx):,}, Val: {len(va_idx):,}) ===")
        print(f"{'='*75}")

        # Feature Engineering strictly in-fold (zero leakage)
        fe_t0 = time.time()
        train_fold_raw = train_raw.iloc[tr_idx].copy()
        val_fold_raw = train_raw.iloc[va_idx].copy()

        df_tr, fold_stats = create_features(train_fold_raw, is_train=True)
        df_va, _ = create_features(val_fold_raw, fold_stats=fold_stats, is_train=False)
        df_te, _ = create_features(test_raw, fold_stats=fold_stats, is_train=False)

        feature_cols = [c for c in df_tr.columns if c not in ['id', 'Will_Buy_EV']]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        print(f"  [Fold {fold+1} FE] Features created in {time.time()-fe_t0:.2f}s ({len(feature_cols)} features)")

        # Prepare categorical dtypes for LightGBM, XGBoost, and HistGradientBoosting
        df_tr_cat = df_tr[feature_cols].copy()
        df_va_cat = df_va[feature_cols].copy()
        df_te_cat = df_te[feature_cols].copy()
        for c in cat_cols:
            df_tr_cat[c] = df_tr_cat[c].astype('category')
            df_va_cat[c] = df_va_cat[c].astype('category')
            df_te_cat[c] = df_te_cat[c].astype('category')

        # -------------------------------------------------------------
        # Model 1a: LightGBM Seed 42 (Depth 7, Leaves 45)
        # -------------------------------------------------------------
        m_t0 = time.time()
        cfg_lgb1 = lgb_cfgs[0]
        params_lgb1 = dict(cfg_lgb1['params'])
        params_lgb1['random_state'] = 42 + fold

        tr_data1 = lgb.Dataset(df_tr_cat, label=y_tr)
        va_data1 = lgb.Dataset(df_va_cat, label=y_va, reference=tr_data1)

        m_lgb1 = lgb.train(
            params_lgb1,
            tr_data1,
            num_boost_round=cfg_lgb1['num_boost_round'],
            valid_sets=[va_data1],
            callbacks=[lgb.early_stopping(cfg_lgb1['early_stopping_rounds'], verbose=False)]
        )
        pred_va_lgb1 = m_lgb1.predict(df_va_cat)
        oof_lgb_s42[va_idx] = pred_va_lgb1
        test_lgb_s42 += m_lgb1.predict(df_te_cat) / n_splits
        auc_lgb1 = roc_auc_score(y_va, pred_va_lgb1)
        print(f"  [M1a LightGBM s42]   AUC: {auc_lgb1:.6f} | trees: {m_lgb1.best_iteration:>4} | {time.time()-m_t0:5.1f}s")

        # -------------------------------------------------------------
        # Model 1b: LightGBM Seed 2026 (Depth 7, Leaves 45, Seed Bag)
        # -------------------------------------------------------------
        m_t0 = time.time()
        cfg_lgb2 = lgb_cfgs[1]
        params_lgb2 = dict(cfg_lgb2['params'])
        params_lgb2['random_state'] = 2026 + fold

        tr_data2 = lgb.Dataset(df_tr_cat, label=y_tr)
        va_data2 = lgb.Dataset(df_va_cat, label=y_va, reference=tr_data2)

        m_lgb2 = lgb.train(
            params_lgb2,
            tr_data2,
            num_boost_round=cfg_lgb2['num_boost_round'],
            valid_sets=[va_data2],
            callbacks=[lgb.early_stopping(cfg_lgb2['early_stopping_rounds'], verbose=False)]
        )
        pred_va_lgb2 = m_lgb2.predict(df_va_cat)
        oof_lgb_s2026[va_idx] = pred_va_lgb2
        test_lgb_s2026 += m_lgb2.predict(df_te_cat) / n_splits
        auc_lgb2 = roc_auc_score(y_va, pred_va_lgb2)
        print(f"  [M1b LightGBM s2026] AUC: {auc_lgb2:.6f} | trees: {m_lgb2.best_iteration:>4} | {time.time()-m_t0:5.1f}s")

        bagged_lgb_fold = (pred_va_lgb1 + pred_va_lgb2) / 2.0
        print(f"  >>> Fold {fold+1} Bagged LightGBM AUC: {roc_auc_score(y_va, bagged_lgb_fold):.6f}")

        # -------------------------------------------------------------
        # Model 2a: XGBoost GPU Seed 42 (Depth 5, CUDA Hist)
        # -------------------------------------------------------------
        m_t0 = time.time()
        cfg_xgb1 = xgb_cfgs[0]
        params_xgb1 = dict(cfg_xgb1['params'])
        params_xgb1['random_state'] = 42 + fold

        m_xgb1 = xgb.XGBClassifier(**params_xgb1)
        m_xgb1.fit(df_tr_cat, y_tr, eval_set=[(df_va_cat, y_va)], verbose=False)
        pred_va_xgb1 = m_xgb1.predict_proba(df_va_cat)[:, 1]
        oof_xgb_s42[va_idx] = pred_va_xgb1
        test_xgb_s42 += m_xgb1.predict_proba(df_te_cat)[:, 1] / n_splits
        auc_xgb1 = roc_auc_score(y_va, pred_va_xgb1)
        print(f"  [M2a XGBoost s42]    AUC: {auc_xgb1:.6f} | trees: {m_xgb1.best_iteration:>4} | {time.time()-m_t0:5.1f}s")

        # -------------------------------------------------------------
        # Model 2b: XGBoost GPU Seed 2026 (Depth 6, CUDA Hist)
        # -------------------------------------------------------------
        m_t0 = time.time()
        cfg_xgb2 = xgb_cfgs[1]
        params_xgb2 = dict(cfg_xgb2['params'])
        params_xgb2['random_state'] = 2026 + fold

        m_xgb2 = xgb.XGBClassifier(**params_xgb2)
        m_xgb2.fit(df_tr_cat, y_tr, eval_set=[(df_va_cat, y_va)], verbose=False)
        pred_va_xgb2 = m_xgb2.predict_proba(df_va_cat)[:, 1]
        oof_xgb_s2026[va_idx] = pred_va_xgb2
        test_xgb_s2026 += m_xgb2.predict_proba(df_te_cat)[:, 1] / n_splits
        auc_xgb2 = roc_auc_score(y_va, pred_va_xgb2)
        print(f"  [M2b XGBoost s2026]  AUC: {auc_xgb2:.6f} | trees: {m_xgb2.best_iteration:>4} | {time.time()-m_t0:5.1f}s")

        bagged_xgb_fold = (pred_va_xgb1 + pred_va_xgb2) / 2.0
        print(f"  >>> Fold {fold+1} Bagged XGBoost  AUC: {roc_auc_score(y_va, bagged_xgb_fold):.6f}")

        # -------------------------------------------------------------
        # Model 3: CatBoost GPU (Depth 7, CUDA, L2=8, LR=0.06)
        # -------------------------------------------------------------
        m_t0 = time.time()
        params_cb = dict(cb_cfg['params'])
        params_cb['random_seed'] = 42 + fold

        m_cb = cb.CatBoostClassifier(**params_cb)
        m_cb.fit(df_tr[feature_cols], y_tr, eval_set=(df_va[feature_cols], y_va), verbose=False)
        pred_va_cb = m_cb.predict_proba(df_va[feature_cols])[:, 1]
        oof_cb[va_idx] = pred_va_cb
        test_cb += m_cb.predict_proba(df_te[feature_cols])[:, 1] / n_splits
        auc_cb = roc_auc_score(y_va, pred_va_cb)
        print(f"  [M3  CatBoost GPU]   AUC: {auc_cb:.6f} | trees: {m_cb.get_best_iteration():>4} | {time.time()-m_t0:5.1f}s")

        # -------------------------------------------------------------
        # Model 4: HistGradientBoosting (CPU, MaxIter 300, LR=0.05, Depth 7)
        # -------------------------------------------------------------
        m_t0 = time.time()
        params_hgb = dict(hgb_cfg['params'])
        params_hgb['random_state'] = 42 + fold

        m_hgb = HistGradientBoostingClassifier(**params_hgb)
        m_hgb.fit(df_tr_cat, y_tr)
        pred_va_hgb = m_hgb.predict_proba(df_va_cat)[:, 1]
        oof_hgb[va_idx] = pred_va_hgb
        test_hgb += m_hgb.predict_proba(df_te_cat)[:, 1] / n_splits
        auc_hgb = roc_auc_score(y_va, pred_va_hgb)
        print(f"  [M4  HistGradBoost]  AUC: {auc_hgb:.6f} | trees: {m_hgb.n_iter_:>4} | {time.time()-m_t0:5.1f}s")

        print(f"--- Fold {fold+1} Total Time: {time.time()-f_start:.1f}s ---")

    # 3. Compute Bagged Predictions
    oof_lgb = (oof_lgb_s42 + oof_lgb_s2026) / 2.0
    test_pred_lgb = (test_lgb_s42 + test_lgb_s2026) / 2.0

    oof_xgb = (oof_xgb_s42 + oof_xgb_s2026) / 2.0
    test_pred_xgb = (test_xgb_s42 + test_xgb_s2026) / 2.0

    test_pred_cb = test_cb
    test_pred_hgb = test_hgb

    # 4. Standalone OOF Evaluation
    print("\n" + "=" * 78)
    print(" 5-FOLD FULL OUT-OF-FOLD (OOF) ROC AUC PERFORMANCE SUMMARY")
    print("=" * 78)
    auc_lgb_s42 = roc_auc_score(y, oof_lgb_s42)
    auc_lgb_s2026 = roc_auc_score(y, oof_lgb_s2026)
    auc_lgb_bag = roc_auc_score(y, oof_lgb)

    auc_xgb_s42 = roc_auc_score(y, oof_xgb_s42)
    auc_xgb_s2026 = roc_auc_score(y, oof_xgb_s2026)
    auc_xgb_bag = roc_auc_score(y, oof_xgb)

    auc_cb = roc_auc_score(y, oof_cb)
    auc_hgb = roc_auc_score(y, oof_hgb)

    print(f"  Model 1a (LightGBM Seed 42, depth 7):     {auc_lgb_s42:.6f}")
    print(f"  Model 1b (LightGBM Seed 2026, depth 8):   {auc_lgb_s2026:.6f}")
    print(f"  >> Model 1 (LightGBM Bagged 42+2026):     {auc_lgb_bag:.6f} (Bagging Delta: {auc_lgb_bag - max(auc_lgb_s42, auc_lgb_s2026):+.6f})")
    print()
    print(f"  Model 2a (XGBoost GPU Seed 42, depth 5):  {auc_xgb_s42:.6f}")
    print(f"  Model 2b (XGBoost GPU Seed 2026, depth 6):{auc_xgb_s2026:.6f}")
    print(f"  >> Model 2 (XGBoost Bagged 42+2026):      {auc_xgb_bag:.6f} (Bagging Delta: {auc_xgb_bag - max(auc_xgb_s42, auc_xgb_s2026):+.6f})")
    print()
    print(f"  >> Model 3 (CatBoost GPU, depth 7, l2=8): {auc_cb:.6f}")
    print(f"  >> Model 4 (HistGradientBoosting, d=7):   {auc_hgb:.6f}")
    print("=" * 78)

    # 5. Core Ensemble Validation & Nelder-Mead Optimization
    print("\n[Step 4/6] Evaluating Ensembles & Ranks...")
    rank_oof_lgb = rankdata(oof_lgb) / n_train
    rank_oof_xgb = rankdata(oof_xgb) / n_train
    rank_oof_cb = rankdata(oof_cb) / n_train
    rank_oof_hgb = rankdata(oof_hgb) / n_train

    rank_test_lgb = rankdata(test_pred_lgb) / n_test
    rank_test_xgb = rankdata(test_pred_xgb) / n_test
    rank_test_cb = rankdata(test_pred_cb) / n_test
    rank_test_hgb = rankdata(test_pred_hgb) / n_test

    # 2-Model Benchmark Blend (LGB + XGB)
    blend_2m = 0.4474 * rank_oof_lgb + 0.5526 * rank_oof_xgb
    auc_2m_raw = roc_auc_score(y, blend_2m)

    zero_mask_tr = (train_raw['Subsidy_Available'] == 'No') & (train_raw['Range_Anxiety_Level'] == 'High')
    blend_2m_post = blend_2m.copy()
    blend_2m_post[zero_mask_tr] = np.minimum(blend_2m_post[zero_mask_tr], blend_2m_post.min() * 0.1)
    auc_2m_post = roc_auc_score(y, blend_2m_post)

    print(f"  2-Model Core Blend (LGB+XGB): Raw AUC = {auc_2m_raw:.6f} | Post-Processed AUC = {auc_2m_post:.6f}")

    # 4-Model Nelder-Mead Weight Optimization
    print("  Optimizing 4-model weights via Nelder-Mead to maximize OOF AUC...")
    OOF_ranks = np.column_stack([rank_oof_lgb, rank_oof_xgb, rank_oof_cb, rank_oof_hgb])

    def loss_func(weights):
        w = np.maximum(0, np.array(weights))
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        b = OOF_ranks @ w
        return -roc_auc_score(y, b)

    opt_res = minimize(loss_func, [0.35, 0.35, 0.15, 0.15], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4})
    best_w = np.maximum(0, opt_res.x)
    best_w = best_w / best_w.sum()

    opt_blend_oof = OOF_ranks @ best_w
    auc_4m_raw = roc_auc_score(y, opt_blend_oof)
    opt_blend_oof_post = opt_blend_oof.copy()
    opt_blend_oof_post[zero_mask_tr] = np.minimum(opt_blend_oof_post[zero_mask_tr], opt_blend_oof_post.min() * 0.1)
    auc_4m_post = roc_auc_score(y, opt_blend_oof_post)

    print(f"  Optimal Weights: LGB={best_w[0]:.4f}, XGB={best_w[1]:.4f}, CB={best_w[2]:.4f}, HGB={best_w[3]:.4f}")
    print(f"  4-Model Optimal Blend: Raw AUC = {auc_4m_raw:.6f} | Post-Processed AUC = {auc_4m_post:.6f}")

    # 6. Save Matrices to disk
    print("\n[Step 5/6] Saving prediction matrices to disk...")
    oof_df = pd.DataFrame({
        'id': train_raw['id'],
        'Will_Buy_EV': y,
        'lgb': oof_lgb,
        'xgb': oof_xgb,
        'cb': oof_cb,
        'hgb': oof_hgb,
        'lgb_s42': oof_lgb_s42,
        'lgb_s2026': oof_lgb_s2026,
        'xgb_s42': oof_xgb_s42,
        'xgb_s2026': oof_xgb_s2026
    })
    oof_csv_path = os.path.join(ROOT_DIR, "src", "oof_predictions.csv")
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"  Saved OOF predictions -> {oof_csv_path} ({len(oof_df):,} rows)")

    test_df = pd.DataFrame({
        'id': test_raw['id'],
        'lgb': test_pred_lgb,
        'xgb': test_pred_xgb,
        'cb': test_pred_cb,
        'hgb': test_pred_hgb,
        'lgb_s42': test_lgb_s42,
        'lgb_s2026': test_lgb_s2026,
        'xgb_s42': test_xgb_s42,
        'xgb_s2026': test_xgb_s2026
    })
    test_csv_path = os.path.join(ROOT_DIR, "src", "test_predictions.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Saved Test predictions -> {test_csv_path} ({len(test_df):,} rows)")

    # Also sync to artifacts/ directory if present or create it
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    oof_df.to_csv(os.path.join(artifacts_dir, "oof_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(artifacts_dir, "test_predictions.csv"), index=False)
    print(f"  Synced copies to {artifacts_dir}/")

    # 7. Generate Final Submission Deliverable
    print("\n[Step 6/6] Generating and validating final submission.csv deliverable...")
    TEST_ranks = np.column_stack([rank_test_lgb, rank_test_xgb, rank_test_cb, rank_test_hgb])
    final_test_preds = TEST_ranks @ best_w

    # Deterministic zero-purchase post-processing
    zero_mask_te = (test_raw['Subsidy_Available'] == 'No') & (test_raw['Range_Anxiety_Level'] == 'High')
    n_boundary = zero_mask_te.sum()
    print(f"  Applying deterministic zero-boundary post-processing to {n_boundary} test cases...")
    min_other = float(final_test_preds[~zero_mask_te].min())
    boundary_val = max(1e-12, min_other * 0.1)
    final_test_preds[zero_mask_te] = boundary_val

    sub = pd.DataFrame({
        'id': test_raw['id'],
        'Will_Buy_EV': final_test_preds
    })
    sub_path = os.path.join(ROOT_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  Wrote submission -> {sub_path}")

    # Strict Deliverable Invariant Checks
    assert os.path.exists(sub_path)
    assert len(sub) == n_test
    assert list(sub.columns) == ['id', 'Will_Buy_EV']
    assert sub['id'].equals(test_raw['id'])
    assert sub['Will_Buy_EV'].isnull().sum() == 0
    assert not np.isinf(sub['Will_Buy_EV']).any()
    assert (sub['Will_Buy_EV'] >= 0.0).all()
    assert (sub['Will_Buy_EV'] <= 1.0).all()

    # Verify zero boundary constraint on submission
    zero_preds = sub.loc[zero_mask_te, 'Will_Buy_EV']
    other_preds = sub.loc[~zero_mask_te, 'Will_Buy_EV']
    assert zero_preds.max() <= other_preds.min()
    print("  [OK] Invariant checks passed on submission.csv!")

    total_time = time.time() - t_start
    print("\n" + "*" * 80)
    print(f" >>> PRODUCTION TRAINING PIPELINE COMPLETE IN {total_time/60:.2f} MINUTES! <<<")
    print("*" * 80)


if __name__ == "__main__":
    train_production_ensemble()
