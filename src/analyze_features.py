import pandas as pd
import numpy as np

# Chicago training data
X_train = pd.read_csv("data/feature_engineering/X_train.csv")
y_train = pd.read_csv("data/feature_engineering/y_train.csv").squeeze()
X_test = pd.read_csv("data/feature_engineering/X_test.csv")
y_test = pd.read_csv("data/feature_engineering/y_test.csv").squeeze()

# NIBRS data
X_nibrs = pd.read_csv("data/nibrs/X_test_nibrs.csv")
y_nibrs = pd.read_csv("data/nibrs/y_test_nibrs.csv").squeeze()

print("="*80)
print("SPATIAL UNIT ANALYSIS")
print("="*80)
print(f"\nChicago:")
print(f"  - Training samples: {len(X_train)} (rolling window 2018-2024)")
print(f"  - Test samples: {len(X_test)} (2025 prediction)")
print(f"  - Spatial unit: community_area")
print(f"  - District columns: {len([c for c in X_train.columns if c.startswith('district_')])}")

print(f"\nNIBRS:")
print(f"  - Test samples: {len(X_nibrs)} agencies")
print(f"  - Spatial unit: agency_id")

# Get ratio features
ratio_features = [col for col in X_train.columns if col.endswith('_ratio')]
print(f"\n11 RATIO FEATURES: {ratio_features}")

print(f"\n"+"="*80)
print("CHICAGO TRAINING (n={})".format(len(X_train)))
print("="*80)
for feat in ratio_features:
    vals = X_train[feat]
    print(f"{feat:30s}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

print(f"\n"+"="*80)
print("CHICAGO TEST (n={})".format(len(X_test)))
print("="*80)
for feat in ratio_features:
    vals = X_test[feat]
    print(f"{feat:30s}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

print(f"\n"+"="*80)
print("NIBRS TEST (n={})".format(len(X_nibrs)))
print("="*80)
X_nibrs_aligned = X_nibrs.drop(columns=['agency_id'])
for feat in ratio_features:
    if feat in X_nibrs_aligned.columns:
        vals = X_nibrs_aligned[feat]
        print(f"{feat:30s}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

print(f"\n"+"="*80)
print("LABEL DISTRIBUTION")
print("="*80)
print(f"Chicago Training: {y_train.sum():3d} hotspots / {len(y_train):3d} samples = {y_train.mean()*100:5.1f}%")
print(f"Chicago Test:    {y_test.sum():3d} hotspots / {len(y_test):3d} samples = {y_test.mean()*100:5.1f}%")
print(f"NIBRS Test:      {y_nibrs.sum():3d} hotspots / {len(y_nibrs):3d} samples = {y_nibrs.mean()*100:5.1f}%")

print(f"\n"+"="*80)
print("MEAN DIFFERENCE BETWEEN CHICAGO & NIBRS")
print("="*80)
chicago_mean = X_train[ratio_features].mean()
nibrs_mean = X_nibrs_aligned[ratio_features].mean()
print(f"\n{'Feature':30s} {'Chicago':>12} {'NIBRS':>12} {'Diff %':>12}")
print("-" * 68)
for feat in ratio_features:
    c_mean = chicago_mean[feat]
    n_mean = nibrs_mean[feat]
    diff = ((n_mean - c_mean) / c_mean * 100) if c_mean != 0 else 0
    print(f"{feat:30s} {c_mean:12.4f} {n_mean:12.4f} {diff:12.1f}%")
