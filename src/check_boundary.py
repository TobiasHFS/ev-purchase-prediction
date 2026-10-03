import pandas as pd
import numpy as np

train = pd.read_csv("playground-series-s6e9/train.csv")
test = pd.read_csv("playground-series-s6e9/test.csv")

tr_mask = (train['Subsidy_Available'] == 'No') & (train['Range_Anxiety_Level'] == 'High')
te_mask = (test['Subsidy_Available'] == 'No') & (test['Range_Anxiety_Level'] == 'High')

print(f"Train matches: {tr_mask.sum()} / {len(train)} (buy count = {(train.loc[tr_mask, 'Will_Buy_EV'] == 'Yes').sum()})")
print(f"Test matches:  {te_mask.sum()} / {len(test)}")

# Also check Subsidy == 'No' overall
tr_nosub = (train['Subsidy_Available'] == 'No')
te_nosub = (test['Subsidy_Available'] == 'No')
print(f"Train Subsidy=No buy rate: {(train.loc[tr_nosub, 'Will_Buy_EV'] == 'Yes').mean():.6%}")

# Check Subsidy == 'No' & Concern <= 2
tr_nosub_low_concern = tr_nosub & (train['Environmental_Concern_Level'] <= 2)
print(f"Train Subsidy=No & Concern<=2 count: {tr_nosub_low_concern.sum()}, buy rate: {(train.loc[tr_nosub_low_concern, 'Will_Buy_EV'] == 'Yes').mean():.6%}")

# Check Recipe score distribution for Subsidy=No
inc_s = train['Annual_Income_USD'] / 100000.0
med_w = (train['Range_Anxiety_Level'] == 'Medium').astype(float)
high_w = (train['Range_Anxiety_Level'] == 'High').astype(float)
recipe = 1.2 * inc_s + 0.6 * train['Environmental_Concern_Level'] - 1.0 * med_w - 3.0 * high_w
print(f"Recipe score for Subsidy=No & Worry=High: max score = {recipe[tr_mask].max():.2f}, threshold = 5.5")
