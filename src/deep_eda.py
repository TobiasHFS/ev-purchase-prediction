import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def deep_eda():
    train = pd.read_csv("playground-series-s6e9/train.csv")
    test = pd.read_csv("playground-series-s6e9/test.csv")
    
    features = [c for c in train.columns if c not in ['id', 'Will_Buy_EV']]
    cat_features = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]

    print("=== 1. ADVERSARIAL VALIDATION FEATURE IMPORTANCE ===")
    tr_samp = train[features].sample(50000, random_state=42).copy()
    tr_samp['is_test'] = 0
    te_samp = test[features].sample(50000, random_state=42).copy()
    te_samp['is_test'] = 1
    adv_df = pd.concat([tr_samp, te_samp], ignore_index=True)
    
    for c in cat_features:
        adv_df[c] = adv_df[c].astype('category')
        
    adv_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)
    adv_model.fit(adv_df[features], adv_df['is_test'])
    
    adv_imp = pd.DataFrame({
        'Feature': features,
        'Importance': adv_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(adv_imp.to_string(index=False))

    print("\n--- Comparing train vs test distributions for top adversarial features ---")
    for f in adv_imp['Feature'].head(5):
        if f in num_features:
            print(f"\n{f}:")
            print(f"  Train: mean={train[f].mean():.2f}, std={train[f].std():.2f}, min={train[f].min():.2f}, max={train[f].max():.2f}")
            print(f"  Test:  mean={test[f].mean():.2f}, std={test[f].std():.2f}, min={test[f].min():.2f}, max={test[f].max():.2f}")
        else:
            print(f"\n{f} (value counts %):")
            print("  Train:", dict(round(train[f].value_counts(normalize=True), 4)))
            print("  Test: ", dict(round(test[f].value_counts(normalize=True), 4)))

    print("\n=== 2. RECIPE ANALYSIS & LOGISTIC FIT ===")
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)

    # Let's inspect the recipe components and see optimal weights using LogisticRegression
    train_feat = train.copy()
    train_feat['med_worry'] = (train['Range_Anxiety_Level'] == 'Medium').astype(float)
    train_feat['high_worry'] = (train['Range_Anxiety_Level'] == 'High').astype(float)
    train_feat['subsidy'] = (train['Subsidy_Available'] == 'Yes').astype(float)
    train_feat['income_scaled'] = train['Annual_Income_USD'] / 100000.0
    train_feat['concern'] = train['Environmental_Concern_Level']

    recipe_cols = ['income_scaled', 'concern', 'subsidy', 'med_worry', 'high_worry']
    from sklearn.linear_model import LogisticRegression
    lr_recipe = LogisticRegression()
    lr_recipe.fit(train_feat[recipe_cols], y)
    print("Fitted Logistic Regression on the 5 recipe columns:")
    for col, coef in zip(recipe_cols, lr_recipe.coef_[0]):
        print(f"  {col:<15}: {coef:.4f} (ratio to income: {coef / lr_recipe.coef_[0][0]:.4f})")
    print(f"  Intercept: {lr_recipe.intercept_[0]:.4f}")
    
    recipe_fitted_prob = lr_recipe.predict_proba(train_feat[recipe_cols])[:, 1]
    print(f"Fitted Recipe Logistic AUC: {roc_auc_score(y, recipe_fitted_prob):.6f}")

    # Check the original recipe score:
    # 1.2 * income/1e5 + 0.6 * concern + 2.0 * subsidy - 1.0 * med_worry - 3.0 * high_worry
    # Let's see if there are other features in the true DGP (data generating process)!
    print("\n=== 3. CHECKING RESIDUAL CORRELATIONS (WHAT IS MISSING FROM RECIPE?) ===")
    residuals = y - recipe_fitted_prob
    for col in num_features:
        r_corr = np.corrcoef(train[col], residuals)[0, 1]
        print(f"Residual correlation with {col:<30}: {r_corr:.5f}")

    for col in cat_features:
        cat_res = pd.DataFrame({'cat': train[col], 'residual': residuals}).groupby('cat')['residual'].mean()
        print(f"\nMean residual for {col}:")
        print(cat_res)

    print("\n=== 4. SIMPSON'S PARADOX CHECK (Charging Stations vs City Type) ===")
    # Check charging stations correlation by city type
    for city in train['City_Type'].unique():
        sub = train[train['City_Type'] == city]
        sub_y = (sub['Will_Buy_EV'] == 'Yes').astype(int)
        corr_home = np.corrcoef(sub['Charging_Stations_Near_Home'], sub_y)[0, 1]
        corr_work = np.corrcoef(sub['Charging_Stations_Near_Work'], sub_y)[0, 1]
        print(f"City: {city:<10} | Home station corr: {corr_home:.4f} | Work station corr: {corr_work:.4f} | Base rate: {sub_y.mean():.4f}")

    print("\nDeep EDA complete.")

if __name__ == "__main__":
    deep_eda()
