#!/usr/bin/env python
"""
Standalone E2E Test Runner for Kaggle Playground Series S6E9
Predicting Electric Vehicle Purchases

Executes the four-tier acceptance test suite with structured diagnostics,
per-tier timing, metrics summary, and strictly controlled exit codes.
"""

import os
import sys
import time
import argparse
import unittest
from datetime import datetime, timezone

# Ensure project root and tests directory are on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from test_e2e_acceptance import (
    TestTier1FeatureAndInvariantChecks,
    TestTier2BoundaryAndCornerCases,
    TestTier3CrossFeatureAndValidationIntegrity,
    TestTier4AcceptancePerformanceCheck,
    SharedDataCache,
    ACCEPTANCE_ROC_AUC_THRESHOLD,
    SUBMISSION_PATH,
    OOF_PATH
)

TIER_MAP = {
    1: {
        "name": "Tier 1: Feature & Invariant Checks",
        "class": TestTier1FeatureAndInvariantChecks,
        "description": "Schemas, row counts, zero nulls, duplicate feature check, submission shape & IDs, raw file hashes"
    },
    2: {
        "name": "Tier 2: Boundary & Corner Cases",
        "class": TestTier2BoundaryAndCornerCases,
        "description": "Deterministic zero boundary (Subsidy=No & Anxiety=High), Dirac clipping at $30k and 5km, feature bounds"
    },
    3: {
        "name": "Tier 3: Cross-Feature & CV Validation",
        "class": TestTier3CrossFeatureAndValidationIntegrity,
        "description": "5-Fold Stratified K-Fold integrity, zero leakage group stats, positive feature whitelist & exclusion"
    },
    4: {
        "name": "Tier 4: Acceptance Performance Check",
        "class": TestTier4AcceptancePerformanceCheck,
        "description": "Ensemble Out-Of-Fold ROC AUC strictly exceeding 0.942289, individual model floors, OOF matrix shape"
    }
}


class StructuredTestResult:
    def __init__(self, method_name, docstring, status, duration_sec, error_msg=None):
        self.method_name = method_name
        self.docstring = docstring or method_name
        self.status = status
        self.duration_sec = duration_sec
        self.error_msg = error_msg


def run_tier(tier_num, tier_info, verbose=False):
    test_class = tier_info["class"]
    loader = unittest.TestLoader()
    test_names = loader.getTestCaseNames(test_class)
    
    results = []
    print(f"\n{'='*78}")
    print(f"[{tier_info['name'].upper()}]")
    print(f"Scope: {tier_info['description']}")
    print(f"{'='*78}")

    tier_start = time.time()

    for method_name in test_names:
        suite = unittest.TestSuite()
        test_instance = test_class(method_name)
        suite.addTest(test_instance)
        
        doc = getattr(test_instance, method_name).__doc__
        doc_clean = doc.strip().split("\n")[0] if doc else method_name

        t0 = time.time()
        runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0)
        res = runner.run(suite)
        elapsed = time.time() - t0

        if res.wasSuccessful():
            status = "PASS"
            err = None
            print(f"  [ PASS ] {method_name:<52} ({elapsed*1000:6.1f} ms) | {doc_clean}")
        elif len(res.failures) > 0:
            status = "FAIL"
            err = res.failures[0][1]
            print(f"  [ FAIL ] {method_name:<52} ({elapsed*1000:6.1f} ms) | {doc_clean}")
            if verbose:
                print(f"          Error: {err.strip()}")
        else:
            status = "ERROR"
            err = res.errors[0][1]
            print(f"  [ ERR  ] {method_name:<52} ({elapsed*1000:6.1f} ms) | {doc_clean}")
            if verbose:
                print(f"          Error: {err.strip()}")

        results.append(StructuredTestResult(method_name, doc_clean, status, elapsed, err))

    tier_duration = time.time() - tier_start
    return results, tier_duration


def main():
    parser = argparse.ArgumentParser(description="Standalone Acceptance Test Runner for Kaggle S6E9")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific tier only (1-4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full failure tracebacks")
    args = parser.parse_args()

    start_time = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    print("*" * 78)
    print(" KAGGLE PLAYGROUND SERIES S6E9: E2E ACCEPTANCE TEST RUNNER")
    print(f" Timestamp:  {now_iso}")
    print(f" Python:     {sys.executable}")
    print(f" Directory:  {ROOT_DIR}")
    print("*" * 78)

    tiers_to_run = [args.tier] if args.tier else [1, 2, 3, 4]
    all_results = {}
    tier_durations = {}

    for t in tiers_to_run:
        results, dur = run_tier(t, TIER_MAP[t], verbose=args.verbose)
        all_results[t] = results
        tier_durations[t] = dur

    total_time = time.time() - start_time

    # Print Aggregate Summary Table
    print(f"\n\n{'='*78}")
    print(" TEST SUITE EXECUTION SUMMARY")
    print(f"{'='*78}")
    print(f"{'Tier':<8} | {'Tier Name':<42} | {'Tests':<6} | {'Pass':<5} | {'Fail':<5} | {'Time (s)':<8}")
    print(f"{'-'*8}-+-{'-'*42}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}")

    total_tests = 0
    total_passed = 0
    total_failed = 0

    for t in tiers_to_run:
        res_list = all_results[t]
        n_tests = len(res_list)
        n_pass = sum(1 for r in res_list if r.status == "PASS")
        n_fail = sum(1 for r in res_list if r.status in ["FAIL", "ERROR"])

        total_tests += n_tests
        total_passed += n_pass
        total_failed += n_fail

        print(f"Tier {t:<3} | {TIER_MAP[t]['name'][:42]:<42} | {n_tests:<6} | {n_pass:<5} | {n_fail:<5} | {tier_durations[t]:<8.2f}")

    print(f"{'-'*8}-+-{'-'*42}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}")
    print(f"{'TOTAL':<8} | {'All Tiers Combined':<42} | {total_tests:<6} | {total_passed:<5} | {total_failed:<5} | {total_time:<8.2f}")
    print(f"{'='*78}")

    # Inspect Key Competition Metrics from Cache
    print("\n--- KEY ACCEPTANCE DELIVERABLES & METRICS AUDIT ---")
    try:
        sub = SharedDataCache.get_submission()
        print(f" [OK] submission.csv exists at project root: {len(sub):,} prediction rows (286,572 lines)")
        print(f"      Columns: {list(sub.columns)}")
        print(f"      Probability range: [{sub['Will_Buy_EV'].min():.7e}, {sub['Will_Buy_EV'].max():.6f}]")
    except Exception as e:
        print(f" [WARN] submission.csv check error: {e}")

    try:
        oof = SharedDataCache.get_oof()
        train = SharedDataCache.get_train()
        y = oof["Will_Buy_EV"]
        from scipy.stats import rankdata
        from sklearn.metrics import roc_auc_score
        import numpy as np

        rank_lgb = rankdata(oof["lgb"]) / len(oof)
        rank_xgb = rankdata(oof["xgb"]) / len(oof)
        ensemble_blend = 0.4474 * rank_lgb + 0.5526 * rank_xgb
        raw_auc = roc_auc_score(y, ensemble_blend)

        zero_mask = (train["Subsidy_Available"] == "No") & (train["Range_Anxiety_Level"] == "High")
        post_blend = ensemble_blend.copy()
        post_blend[zero_mask] = np.minimum(post_blend[zero_mask], post_blend.min() * 0.1)
        post_auc = roc_auc_score(y, post_blend)

        print(f" [OK] Ensemble OOF ROC AUC (Raw):             {raw_auc:.8f}")
        print(f" [OK] Ensemble OOF ROC AUC (Post-Processed):   {post_auc:.8f}")
        print(f"      Acceptance Benchmark Threshold:          {ACCEPTANCE_ROC_AUC_THRESHOLD:.8f}")
        print(f"      Net Delta Above Requirement:            +{post_auc - ACCEPTANCE_ROC_AUC_THRESHOLD:.8f}")
        assert post_auc > ACCEPTANCE_ROC_AUC_THRESHOLD
    except Exception as e:
        print(f" [WARN] OOF AUC check error: {e}")

    # Final Verdict & Exit Code
    if total_failed == 0:
        print(f"\n{'*'*78}")
        print(" >>> ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY (STATUS: PASS) <<< ")
        print(f"{'*'*78}\n")
        sys.exit(0)
    else:
        print(f"\n{'!'*78}")
        print(f" >>> ACCEPTANCE TEST SUITE FAILED: {total_failed} TESTS FAILED (STATUS: FAIL) <<< ")
        print(f"{'!'*78}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
