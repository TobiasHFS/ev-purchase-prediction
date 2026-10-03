import pandas as pd
import numpy as np

df = pd.read_csv("playground-series-s6e9/train.csv")
print(f"Total rows: {len(df)}")

is_30k = (df['Annual_Income_USD'] == 30000)
print(f"Income == 30,000 count: {is_30k.sum()} ({is_30k.mean():.4%})")

# Let's check min and max for other features
for col in ['Age', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
    min_val = df[col].min()
    max_val = df[col].max()
    min_cnt = (df[col] == min_val).sum()
    max_cnt = (df[col] == max_val).sum()
    print(f"Feature {col}: min={min_val} (count={min_cnt}, {min_cnt/len(df):.3%}), max={max_val} (count={max_cnt}, {max_cnt/len(df):.3%})")

print("\n--- Examining Income == 30,000 subgroup ---")
sub = df[is_30k]
print(pd.crosstab([sub['Subsidy_Available'], sub['Range_Anxiety_Level']], sub['Will_Buy_EV'], normalize='index'))

print("\n--- Examining Daily_Commute_km == 5.0 subgroup ---")
is_5km = (df['Daily_Commute_km'] == 5.0)
print(f"Commute == 5.0 count: {is_5km.sum()} ({is_5km.mean():.4%})")

print("\n--- Examining Zero Buy Rate Segments ---")
# Are there segments where Will_Buy_EV is strictly 0 or 1?
sub_nosub = df[(df['Subsidy_Available'] == 'No') & (df['Range_Anxiety_Level'] == 'High')]
print(f"Subsidy=No & Worry=High count: {len(sub_nosub)}, Buy rate: {(sub_nosub['Will_Buy_EV'] == 'Yes').mean():.6%}")

sub_perfect = df[(df['Subsidy_Available'] == 'Yes') & (df['Range_Anxiety_Level'] == 'Low') & (df['Environmental_Concern_Level'] == 5) & (df['Annual_Income_USD'] > 120000)]
print(f"Top tier (Subsidy=Yes, Worry=Low, Concern=5, Income>120k) count: {len(sub_perfect)}, Buy rate: {(sub_perfect['Will_Buy_EV'] == 'Yes').mean():.6%}")
