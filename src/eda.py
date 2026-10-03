import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def run_eda():
    train_path = "playground-series-s6e9/train.csv"
    test_path = "playground-series-s6e9/test.csv"
    sample_sub_path = "playground-series-s6e9/sample_submission.csv"

    print("--- Loading Datasets ---")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")

    # Binary encode target
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    features = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    cat_features = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]

    print(f"\nCategorical features ({len(cat_features)}): {cat_features}")
    print(f"Numerical features ({len(num_features)}): {num_features}")

    print("\n--- Testing 'Secret Buying Recipe' Formula ---")
    med_worry = (train['Range_Anxiety_Level'] == 'Medium').astype(float)
    high_worry = (train['Range_Anxiety_Level'] == 'High').astype(float)
    subsidy = (train['Subsidy_Available'] == 'Yes').astype(float)
    income_scaled = train['Annual_Income_USD'] / 100000.0
    concern = train['Environmental_Concern_Level']

    recipe_score = (
        1.2 * income_scaled +
        0.6 * concern +
        2.0 * subsidy -
        1.0 * med_worry -
        3.0 * high_worry
    )

    auc_recipe = roc_auc_score(y, recipe_score)
    print(f"Raw Recipe Buy Score ROC AUC on train: {auc_recipe:.6f}")

    # Baseline Logistic Regression on all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ]
    )

    X_train_trans = preprocessor.fit_transform(train[features])
    encoded_feature_names = preprocessor.get_feature_names_out()

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_trans, y)
    lr_preds = lr.predict_proba(X_train_trans)[:, 1]
    auc_lr = roc_auc_score(y, lr_preds)
    print(f"\nStandard Logistic Regression (all features) ROC AUC on train: {auc_lr:.6f}")

    print("\n--- Logistic Regression Coefficients ---")
    coef_df = pd.DataFrame({
        'Feature': encoded_feature_names,
        'Coefficient': lr.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    print(coef_df.to_string(index=False))

    # Fast LightGBM baseline
    import lightgbm as lgb
    print("\n--- Fast LightGBM Baseline (5-fold Stratified CV) ---")
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(train))
    
    # Pre-encode categoricals as category dtype for LightGBM
    train_lgb = train[features].copy()
    for col in cat_features:
        train_lgb[col] = train_lgb[col].astype('category')

    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        X_tr, y_tr = train_lgb.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = train_lgb.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42,
            n_jobs=4,
            verbose=-1
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        val_pred = model.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        print(f"Fold {fold + 1} LightGBM AUC: {fold_auc:.6f}")

    total_auc = roc_auc_score(y, oof_lgb)
    print(f"\nLightGBM 5-Fold OOF ROC AUC: {total_auc:.6f}")

    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\n--- LightGBM Feature Importances ---")
    print(importance_df.to_string(index=False))

    # Adversarial Validation
    print("\n--- Adversarial Validation (Train vs Test) ---")
    X_tr_adv = train[features].sample(50000, random_state=42).copy()
    X_tr_adv['is_test'] = 0
    X_te_adv = test[features].sample(50000, random_state=42).copy()
    X_te_adv['is_test'] = 1
    adv_df = pd.concat([X_tr_adv, X_te_adv], ignore_index=True)
    y_adv = adv_df['is_test']
    
    for col in cat_features:
        adv_df[col] = adv_df[col].astype('category')

    adv_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=4, verbose=-1, random_state=42)
    adv_model.fit(adv_df[features], y_adv)
    adv_preds = adv_model.predict_proba(adv_df[features])[:, 1]
    adv_auc = roc_auc_score(y_adv, adv_preds)
    print(f"Adversarial Validation ROC AUC (LightGBM): {adv_auc:.4f} (0.50 means identical train/test distributions)")

    print("\nEDA completed successfully.")

if __name__ == "__main__":
    run_eda()
