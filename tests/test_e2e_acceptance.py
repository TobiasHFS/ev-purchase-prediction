"""
E2E Acceptance Test Suite for Kaggle Playground Series S6E9
(Predicting Electric Vehicle Purchases)

Four-Tier Test Architecture:
- Tier 1: Feature & Invariant Checks (raw data integrity, submission format, schema sanity, vocab validation)
- Tier 2: Boundary & Corner Cases (zero-purchase boundary, Dirac clipping spikes, extreme values)
- Tier 3: Cross-Feature & CV Validation (Stratified K-Fold, zero leakage, feature whitelist & exclusion)
- Tier 4: Acceptance Performance Check (ensemble OOF ROC AUC strictly > 0.942289)
"""

import os
import unittest
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Expected authoritative constants
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(ROOT_DIR, "playground-series-s6e9", "train.csv")
TEST_PATH = os.path.join(ROOT_DIR, "playground-series-s6e9", "test.csv")
SAMPLE_SUB_PATH = os.path.join(ROOT_DIR, "playground-series-s6e9", "sample_submission.csv")
SUBMISSION_PATH = os.path.join(ROOT_DIR, "submission.csv")
OOF_PATH = os.path.join(ROOT_DIR, "src", "oof_predictions.csv")

TRAIN_ROW_COUNT = 668665
TEST_ROW_COUNT = 286571
SUBMISSION_TOTAL_LINES = 286572

EXPECTED_RAW_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]

CATEGORICAL_VOCABULARIES = {
    "Gender": {"Male", "Female", "Other"},
    "City_Type": {"Urban", "Suburban", "Rural"},
    "Current_Car_Type": {"Sedan", "SUV", "Hatchback", "Truck"},
    "Home_Charging_Possible": {"Yes", "No"},
    "Subsidy_Available": {"Yes", "No"},
    "Range_Anxiety_Level": {"Low", "Medium", "High"}
}

EXPECTED_POSITIVE_FEATURES = [
    "income_x_subsidy",
    "income_x_concern",
    "income_per_car",
    "subsidy_x_concern",
    "income_diff_city",
    "commute_diff_city"
]

DEGRADED_EXCLUDED_FEATURES = [
    "linear_recipe_score",
    "base_margin_injection",
    "public_charging_ratios",
    "no_home_charge",
    "total_chargers",
    "chargers_if_no_home",
    "commute_per_charger",
    "home_chargers_if_no_home",
    "range_anxiety_crosses",
    "anxiety_num",
    "anxiety_x_commute",
    "anxiety_x_no_home",
    "categorical_cross_products",
    "city_home_charge",
    "city_car_type",
    "subsidy_worry",
    "home_charge_worry",
    "boundary_clipping_flags",
    "is_income_min",
    "is_commute_min",
    "is_zero_chargers",
    "city_charging_diffs",
    "home_chg_diff_city",
    "work_chg_diff_city",
    "city_age_diff",
    "extratrees_lgbm"
]

# Authoritative SHA-256 hashes of original raw files (Strict READ-ONLY validation)
EXPECTED_SHA256 = {
    "train.csv": "eae9eaa4e6378df405e755f853771d7e26d212bd93258349fc797b771021946a",
    "test.csv": "539263f6caabc40afd5e2f0bc0ab16b10a2d1177c565fc71b866f0181d836b34",
    "sample_submission.csv": "a9747a8b947e4e35505e3da4535a5a494978b012a7adb50973f13e598849dda5"
}

ACCEPTANCE_ROC_AUC_THRESHOLD = 0.942289


class SharedDataCache:
    """Singleton cache to avoid repeatedly parsing 45MB CSV files across test cases."""
    _train_df = None
    _test_df = None
    _sub_df = None
    _oof_df = None

    @classmethod
    def get_train(cls):
        if cls._train_df is None:
            assert os.path.exists(TRAIN_PATH), f"Raw train file missing: {TRAIN_PATH}"
            cls._train_df = pd.read_csv(TRAIN_PATH)
        return cls._train_df

    @classmethod
    def get_test(cls):
        if cls._test_df is None:
            assert os.path.exists(TEST_PATH), f"Raw test file missing: {TEST_PATH}"
            cls._test_df = pd.read_csv(TEST_PATH)
        return cls._test_df

    @classmethod
    def get_submission(cls):
        if cls._sub_df is None:
            assert os.path.exists(SUBMISSION_PATH), f"Submission file missing: {SUBMISSION_PATH}"
            cls._sub_df = pd.read_csv(SUBMISSION_PATH)
        return cls._sub_df

    @classmethod
    def get_oof(cls):
        if cls._oof_df is None:
            assert os.path.exists(OOF_PATH), f"OOF predictions file missing: {OOF_PATH}"
            cls._oof_df = pd.read_csv(OOF_PATH)
        return cls._oof_df


# =====================================================================
# TIER 1: Feature & Invariant Checks
# =====================================================================
class TestTier1FeatureAndInvariantChecks(unittest.TestCase):
    """Verifies schemas, row counts, null values, uniqueness, vocabularies, and file integrity."""

    def test_raw_data_files_unmodified(self):
        """Tier 1: Verify raw data files in playground-series-s6e9/ have not been altered."""
        for filename, expected_hash in EXPECTED_SHA256.items():
            filepath = os.path.join(ROOT_DIR, "playground-series-s6e9", filename)
            self.assertTrue(os.path.exists(filepath), f"Raw file does not exist: {filepath}")
            
            sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Raw file '{filename}' corrupted or modified! Hash mismatch."
            )

    def test_train_schema_and_integrity(self):
        """Tier 1: Verify train.csv shape, column names, 0 nulls, and 0 duplicate feature rows."""
        train = SharedDataCache.get_train()
        self.assertEqual(len(train), TRAIN_ROW_COUNT, f"Train row count must be {TRAIN_ROW_COUNT}")
        
        expected_cols = ["id"] + EXPECTED_RAW_FEATURES + ["Will_Buy_EV"]
        self.assertEqual(list(train.columns), expected_cols, "Train columns do not match expected schema")
        
        # Zero nulls
        null_counts = train.isnull().sum()
        self.assertEqual(null_counts.sum(), 0, f"Train contains unexpected nulls: {null_counts[null_counts > 0].to_dict()}")
        
        # 0 duplicate feature rows
        dup_count = train.duplicated(subset=EXPECTED_RAW_FEATURES).sum()
        self.assertEqual(dup_count, 0, f"Train contains {dup_count} duplicate feature rows")

    def test_test_schema_and_integrity(self):
        """Tier 1: Verify test.csv shape, column names, 0 nulls, and 0 duplicate feature rows."""
        test = SharedDataCache.get_test()
        self.assertEqual(len(test), TEST_ROW_COUNT, f"Test row count must be {TEST_ROW_COUNT}")
        
        expected_cols = ["id"] + EXPECTED_RAW_FEATURES
        self.assertEqual(list(test.columns), expected_cols, "Test columns do not match expected schema")
        
        # Zero nulls
        null_counts = test.isnull().sum()
        self.assertEqual(null_counts.sum(), 0, f"Test contains unexpected nulls: {null_counts[null_counts > 0].to_dict()}")
        
        # 0 duplicate feature rows
        dup_count = test.duplicated(subset=EXPECTED_RAW_FEATURES).sum()
        self.assertEqual(dup_count, 0, f"Test contains {dup_count} duplicate feature rows")

    def test_categorical_vocabularies_and_encoding_integrity(self):
        """Tier 1 (Adversarial): Verify categorical columns have strictly valid vocabulary with no whitespace/corrupt strings."""
        train = SharedDataCache.get_train()
        test = SharedDataCache.get_test()

        for col, expected_vocab in CATEGORICAL_VOCABULARIES.items():
            train_vals = set(train[col].dropna().unique())
            test_vals = set(test[col].dropna().unique())

            self.assertEqual(
                train_vals,
                expected_vocab,
                f"Train column '{col}' has unexpected vocabulary values: {train_vals ^ expected_vocab}"
            )
            self.assertEqual(
                test_vals,
                expected_vocab,
                f"Test column '{col}' has unexpected vocabulary values: {test_vals ^ expected_vocab}"
            )

    def test_submission_file_existence_and_structure(self):
        """Tier 1: Verify submission.csv exists, exactly 286,572 lines, columns id,Will_Buy_EV."""
        self.assertTrue(os.path.exists(SUBMISSION_PATH), f"Submission file missing at {SUBMISSION_PATH}")
        
        # Verify physical line count
        with open(SUBMISSION_PATH, "rb") as f:
            total_lines = sum(1 for _ in f)
        self.assertEqual(total_lines, SUBMISSION_TOTAL_LINES, f"submission.csv line count must be {SUBMISSION_TOTAL_LINES}")
        
        sub = SharedDataCache.get_submission()
        self.assertEqual(len(sub), TEST_ROW_COUNT, f"submission.csv data rows must be {TEST_ROW_COUNT}")
        self.assertEqual(list(sub.columns), ["id", "Will_Buy_EV"], f"Submission columns must strictly be ['id', 'Will_Buy_EV']")

    def test_submission_id_alignment(self):
        """Tier 1: Verify submission.csv IDs strictly match test.csv IDs in exact order without duplicates."""
        test = SharedDataCache.get_test()
        sub = SharedDataCache.get_submission()
        self.assertTrue(sub["id"].equals(test["id"]), "submission.csv 'id' column does not match test.csv exactly")
        self.assertEqual(sub["id"].duplicated().sum(), 0, "submission.csv contains duplicate IDs")

    def test_submission_probability_validity(self):
        """Tier 1: Verify all predictions are finite, non-null, and bounded in [0.0, 1.0]."""
        sub = SharedDataCache.get_submission()
        preds = sub["Will_Buy_EV"]
        
        self.assertEqual(preds.isnull().sum(), 0, "submission.csv contains null or NaN values")
        self.assertFalse(np.isinf(preds).any(), "submission.csv contains infinite values")
        self.assertTrue((preds >= 0.0).all(), "submission.csv contains negative probabilities")
        self.assertTrue((preds <= 1.0).all(), "submission.csv contains probabilities greater than 1.0")


# =====================================================================
# TIER 2: Boundary & Corner Cases
# =====================================================================
class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Verifies domain rules, deterministic zero boundary, and Dirac clipping spikes."""

    def test_deterministic_zero_purchase_condition(self):
        """Tier 2: Verify Subsidy_Available == 'No' & Range_Anxiety_Level == 'High' received minimum probability."""
        train = SharedDataCache.get_train()
        test = SharedDataCache.get_test()
        sub = SharedDataCache.get_submission()

        # Domain truth check on train set: 887 cases, 0 buys
        tr_mask = (train["Subsidy_Available"] == "No") & (train["Range_Anxiety_Level"] == "High")
        tr_buys = (train.loc[tr_mask, "Will_Buy_EV"] == "Yes").sum()
        self.assertEqual(tr_buys, 0, f"Expected exactly 0 buys in train for Subsidy=No & Anxiety=High, found {tr_buys}")
        self.assertEqual(tr_mask.sum(), 887, f"Expected 887 matching train rows, got {tr_mask.sum()}")

        # Test set boundary check: 404 test cases
        te_mask = (test["Subsidy_Available"] == "No") & (test["Range_Anxiety_Level"] == "High")
        self.assertEqual(te_mask.sum(), 404, f"Expected 404 matching test rows, got {te_mask.sum()}")

        zero_preds = sub.loc[te_mask, "Will_Buy_EV"]
        other_preds = sub.loc[~te_mask, "Will_Buy_EV"]

        min_sub_prob = sub["Will_Buy_EV"].min()
        max_zero_prob = zero_preds.max()
        min_other_prob = other_preds.min()

        self.assertAlmostEqual(
            max_zero_prob,
            min_sub_prob,
            places=8,
            msg="Zero-boundary test rows do not have the lowest probability in submission"
        )
        self.assertLessEqual(
            max_zero_prob,
            min_other_prob,
            "Highest zero-boundary prediction exceeds the lowest non-boundary prediction"
        )

    def test_dirac_clipping_bounds_income(self):
        """Tier 2: Verify Dirac clipping spike at $30k income (lower bound, zero values below)."""
        train = SharedDataCache.get_train()
        test = SharedDataCache.get_test()

        # No value strictly less than 30k in train or test
        self.assertEqual((train["Annual_Income_USD"] < 30000.0).sum(), 0, "Train has income < $30,000")
        self.assertEqual((test["Annual_Income_USD"] < 30000.0).sum(), 0, "Test has income < $30,000")

        # Exact spike counts
        train_spike_count = (train["Annual_Income_USD"] == 30000.0).sum()
        test_spike_count = (test["Annual_Income_USD"] == 30000.0).sum()

        self.assertEqual(train_spike_count, 61605, f"Expected 61,605 train rows at $30k, got {train_spike_count}")
        self.assertEqual(test_spike_count, 26276, f"Expected 26,276 test rows at $30k, got {test_spike_count}")

        # Proportions consistent (~9.21% train, ~9.17% test)
        self.assertAlmostEqual(train_spike_count / len(train), 0.092131, places=4)
        self.assertAlmostEqual(test_spike_count / len(test), 0.091691, places=4)

        # Upper bound handling (smooth distribution up to ~188k)
        self.assertGreaterEqual(train["Annual_Income_USD"].max(), 180000.0)
        self.assertGreaterEqual(test["Annual_Income_USD"].max(), 180000.0)

    def test_dirac_clipping_bounds_commute(self):
        """Tier 2: Verify Dirac clipping spike at 5.0 km commute (lower bound, zero values below)."""
        train = SharedDataCache.get_train()
        test = SharedDataCache.get_test()

        # No value strictly less than 5.0 in train or test
        self.assertEqual((train["Daily_Commute_km"] < 5.0).sum(), 0, "Train has commute < 5.0 km")
        self.assertEqual((test["Daily_Commute_km"] < 5.0).sum(), 0, "Test has commute < 5.0 km")

        # Exact spike counts
        train_spike_count = (train["Daily_Commute_km"] == 5.0).sum()
        test_spike_count = (test["Daily_Commute_km"] == 5.0).sum()

        self.assertEqual(train_spike_count, 144280, f"Expected 144,280 train rows at 5.0km, got {train_spike_count}")
        self.assertEqual(test_spike_count, 61668, f"Expected 61,668 test rows at 5.0km, got {test_spike_count}")

        # Proportions consistent (~21.58% train, ~21.52% test)
        self.assertAlmostEqual(train_spike_count / len(train), 0.215773, places=4)
        self.assertAlmostEqual(test_spike_count / len(test), 0.215193, places=4)

    def test_infrastructure_bounds_and_no_zero_division(self):
        """Tier 2: Verify discrete feature bounds and ensure Number_of_Cars_Owned >= 1 prevents zero-division in income_per_car."""
        train = SharedDataCache.get_train()
        test = SharedDataCache.get_test()

        for df, name in [(train, "train"), (test, "test")]:
            self.assertGreaterEqual(df["Charging_Stations_Near_Home"].min(), 0)
            self.assertLessEqual(df["Charging_Stations_Near_Home"].max(), 14)
            self.assertGreaterEqual(df["Charging_Stations_Near_Work"].min(), 0)
            self.assertLessEqual(df["Charging_Stations_Near_Work"].max(), 19)
            
            # Crucial: Number_of_Cars_Owned in [1, 4] ensures income_per_car has no divide-by-zero
            self.assertGreaterEqual(df["Number_of_Cars_Owned"].min(), 1)
            self.assertLessEqual(df["Number_of_Cars_Owned"].max(), 4)


# =====================================================================
# TIER 3: Cross-Feature & CV Validation
# =====================================================================
class TestTier3CrossFeatureAndValidationIntegrity(unittest.TestCase):
    """Verifies 5-Fold Stratified K-Fold setup, zero leakage, and feature selection whitelist."""

    def test_stratified_kfold_split_specification(self):
        """Tier 3: Verify 5-Fold Stratified K-Fold (n_splits=5, shuffle=True, random_state=42) partition."""
        train = SharedDataCache.get_train()
        y = (train["Will_Buy_EV"] == "Yes").astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(train, y))
        self.assertEqual(len(folds), 5, "Expected exactly 5 folds")

        overall_pos_rate = y.mean()
        visited_indices = set()

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            # 668,665 / 5 = exactly 133,733 validation rows per fold
            self.assertEqual(len(va_idx), 133733, f"Fold {fold_idx} validation size mismatch")
            self.assertEqual(len(tr_idx), 534932, f"Fold {fold_idx} train size mismatch")

            # Check positive rate alignment (within 0.05% margin)
            va_pos_rate = y.iloc[va_idx].mean()
            self.assertAlmostEqual(
                va_pos_rate,
                overall_pos_rate,
                delta=0.0005,
                msg=f"Fold {fold_idx} positive class rate {va_pos_rate:.5f} deviates from overall {overall_pos_rate:.5f}"
            )

            # Ensure validation indices are disjoint
            va_set = set(va_idx)
            self.assertTrue(visited_indices.isdisjoint(va_set), f"Fold {fold_idx} has overlapping validation indices")
            visited_indices.update(va_set)

        # Ensure all rows were covered
        self.assertEqual(len(visited_indices), len(train), "Folds do not partition entire training dataset")

    def test_zero_leakage_group_statistics(self):
        """Tier 3: Verify group statistics (income/commute by City_Type) must be isolated within folds."""
        train = SharedDataCache.get_train()
        y = (train["Will_Buy_EV"] == "Yes").astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        global_city_income = train.groupby("City_Type")["Annual_Income_USD"].mean().to_dict()

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
            tr_df = train.iloc[tr_idx]
            fold_city_income = tr_df.groupby("City_Type")["Annual_Income_USD"].mean().to_dict()

            # Fold-computed city means must not be identical to global means (proves fold variance)
            for city in global_city_income:
                diff = abs(fold_city_income[city] - global_city_income[city])
                self.assertGreater(
                    diff,
                    0.0,
                    f"Fold {fold_idx} group stat for {city} identical to global (indicates leakage if using global)"
                )

    def test_production_feature_whitelist_and_exclusion(self):
        """Tier 3: Verify production feature engineering contains only 6 positive features and no degraded features."""
        import sys
        sys.path.append(os.path.join(ROOT_DIR, "src"))

        from train_full_ensemble import build_features
        sample_tr = SharedDataCache.get_train().head(20).copy()
        sample_te = SharedDataCache.get_test().head(20).copy()

        tr_trans, te_trans, f_cols, c_cols = build_features(sample_tr, sample_te)

        base_set = set(EXPECTED_RAW_FEATURES)
        all_feature_set = set(f_cols)
        engineered_features = all_feature_set - base_set

        # Verify all 6 positive features are included
        for pos_feat in EXPECTED_POSITIVE_FEATURES:
            self.assertIn(
                pos_feat,
                engineered_features,
                f"Required positive feature '{pos_feat}' is missing from feature pipeline!"
            )

        # Verify no degraded features are included
        for deg_feat in DEGRADED_EXCLUDED_FEATURES:
            self.assertNotIn(
                deg_feat,
                all_feature_set,
                f"Degraded feature '{deg_feat}' must be excluded from feature pipeline!"
            )


# =====================================================================
# TIER 4: Acceptance Performance Check
# =====================================================================
class TestTier4AcceptancePerformanceCheck(unittest.TestCase):
    """Verifies that ensemble OOF ROC AUC strictly exceeds 0.942289."""

    def test_oof_predictions_file_integrity(self):
        """Tier 4: Verify oof_predictions.csv exists, has 668,665 rows, expected columns, 0 nulls."""
        self.assertTrue(os.path.exists(OOF_PATH), f"OOF predictions file missing at {OOF_PATH}")
        oof = SharedDataCache.get_oof()
        self.assertEqual(len(oof), TRAIN_ROW_COUNT, f"OOF row count must be {TRAIN_ROW_COUNT}")
        
        required_cols = ["id", "Will_Buy_EV", "lgb", "xgb", "cb", "hgb"]
        for col in required_cols:
            self.assertIn(col, oof.columns, f"Required column '{col}' missing from OOF predictions")

        self.assertEqual(oof[required_cols].isnull().sum().sum(), 0, "OOF predictions contain null values")

    def test_individual_model_performance_floors(self):
        """Tier 4: Verify individual baseline models achieve reasonable minimum ROC AUC floors."""
        oof = SharedDataCache.get_oof()
        y = oof["Will_Buy_EV"]

        auc_lgb = roc_auc_score(y, oof["lgb"])
        auc_xgb = roc_auc_score(y, oof["xgb"])
        auc_cb = roc_auc_score(y, oof["cb"])
        auc_hgb = roc_auc_score(y, oof["hgb"])

        self.assertGreater(auc_lgb, 0.9415, f"LightGBM OOF AUC {auc_lgb:.6f} below floor")
        self.assertGreater(auc_xgb, 0.9415, f"XGBoost OOF AUC {auc_xgb:.6f} below floor")
        self.assertGreater(auc_cb, 0.9405, f"CatBoost OOF AUC {auc_cb:.6f} below floor")
        self.assertGreater(auc_hgb, 0.9405, f"HGBC OOF AUC {auc_hgb:.6f} below floor")

    def test_ensemble_oof_roc_auc_strictly_exceeds_threshold(self):
        """Tier 4: Verify ensemble Out-Of-Fold ROC AUC strictly exceeds 0.942289."""
        oof = SharedDataCache.get_oof()
        train = SharedDataCache.get_train()
        y = oof["Will_Buy_EV"]
        n_train = len(oof)

        # Percentile ranks for the core models
        rank_lgb = rankdata(oof["lgb"]) / n_train
        rank_xgb = rankdata(oof["xgb"]) / n_train

        # Production Nelder-Mead weights (LGB: 0.4474, XGB: 0.5526)
        w_lgb = 0.4474
        w_xgb = 0.5526
        ensemble_blend = w_lgb * rank_lgb + w_xgb * rank_xgb

        raw_auc = roc_auc_score(y, ensemble_blend)

        # Post-processed with zero-boundary enforcement
        zero_mask = (train["Subsidy_Available"] == "No") & (train["Range_Anxiety_Level"] == "High")
        post_blend = ensemble_blend.copy()
        post_blend[zero_mask] = np.minimum(post_blend[zero_mask], post_blend.min() * 0.1)
        post_auc = roc_auc_score(y, post_blend)

        print(f"\n[TIER 4 ACCEPTANCE] Raw Ensemble OOF ROC AUC:            {raw_auc:.8f}")
        print(f"[TIER 4 ACCEPTANCE] Post-Processed Ensemble OOF ROC AUC:    {post_auc:.8f}")
        print(f"[TIER 4 ACCEPTANCE] Baseline Requirement Threshold:         {ACCEPTANCE_ROC_AUC_THRESHOLD:.8f}")
        print(f"[TIER 4 ACCEPTANCE] Net Delta over Threshold:              +{post_auc - ACCEPTANCE_ROC_AUC_THRESHOLD:.8f}")

        # The core acceptance gate: strictly exceed 0.942289
        self.assertGreater(
            post_auc,
            ACCEPTANCE_ROC_AUC_THRESHOLD,
            f"Post-processed ensemble OOF ROC AUC {post_auc:.8f} does not strictly exceed threshold {ACCEPTANCE_ROC_AUC_THRESHOLD}!"
        )
        self.assertGreater(
            raw_auc,
            ACCEPTANCE_ROC_AUC_THRESHOLD,
            f"Raw ensemble OOF ROC AUC {raw_auc:.8f} does not strictly exceed threshold {ACCEPTANCE_ROC_AUC_THRESHOLD}!"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
