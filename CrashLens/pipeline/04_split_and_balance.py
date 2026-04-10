"""
CrashLens — Step 4: Train/Val/Test Split & Class Imbalance Handling
Creates stratified splits grouped by crash (no data leakage),
applies SMOTE+ADASYN to training set, computes class weights.
"""
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from collections import Counter

PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & FEATURE CONFIG
# ═══════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "crashlens_clean.parquet"))
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    config = json.load(f)

model_features = config["model_features"]
target = config["target_column"]

X = df[model_features].values
y = df[target].values
groups = df["CASENUM"].values  # Group by crash case
weights = df["WEIGHT"].values  # CRSS sampling weights

print(f"  X: {X.shape}, y: {y.shape}")
print(f"  Unique crashes: {len(np.unique(groups)):,}")

# ═══════════════════════════════════════════════════════════════════════
# 2. STRATIFIED GROUP-AWARE SPLIT
# ═══════════════════════════════════════════════════════════════════════
# Critical: persons from the same crash must stay in the same split
# (data leakage prevention)
print("\nCreating train/val/test splits (group-aware)...")

# First split: 80% train+val, 20% test
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss1.split(X, y, groups))

X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
X_test, y_test = X[test_idx], y[test_idx]
groups_trainval = groups[trainval_idx]
weights_test = weights[test_idx]

# Second split: 75% train, 25% val (of trainval = 60/20 overall)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx_rel, val_idx_rel = next(gss2.split(X_trainval, y_trainval, groups_trainval))

X_train, y_train = X_trainval[train_idx_rel], y_trainval[train_idx_rel]
X_val, y_val = X_trainval[val_idx_rel], y_trainval[val_idx_rel]
weights_train = weights[trainval_idx][train_idx_rel]
weights_val = weights[trainval_idx][val_idx_rel]

print(f"  Train: {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Val:   {X_val.shape[0]:,} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verify no group leakage
train_cases = set(groups[trainval_idx][train_idx_rel])
val_cases = set(groups[trainval_idx][val_idx_rel])
test_cases = set(groups[test_idx])
assert len(train_cases & val_cases) == 0, "Group leakage: train ∩ val"
assert len(train_cases & test_cases) == 0, "Group leakage: train ∩ test"
assert len(val_cases & test_cases) == 0, "Group leakage: val ∩ test"
print("  ✓ No group leakage between splits")

# Check class distribution in each split
print("\n  Class distribution per split:")
for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = Counter(ys)
    total = len(ys)
    parts = [f"{k}:{v/total*100:.1f}%" for k, v in sorted(dist.items())]
    print(f"    {name}: {', '.join(parts)}")

# ═══════════════════════════════════════════════════════════════════════
# 3. COMPUTE CLASS WEIGHTS (for loss function weighting)
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing class weights...")

# Inverse frequency weighting
class_counts = Counter(y_train)
total = len(y_train)
n_classes = len(class_counts)

# Standard inverse-frequency weights
class_weights_inv = {c: total / (n_classes * count) for c, count in class_counts.items()}

# Normalized so sum = n_classes (keeps loss magnitude stable)
weight_sum = sum(class_weights_inv.values())
class_weights_norm = {c: w * n_classes / weight_sum for c, w in class_weights_inv.items()}

# Effective weights (sqrt dampening to avoid over-weighting rare classes)
class_weights_sqrt = {c: np.sqrt(w) for c, w in class_weights_norm.items()}
sqrt_sum = sum(class_weights_sqrt.values())
class_weights_sqrt = {c: w * n_classes / sqrt_sum for c, w in class_weights_sqrt.items()}

print("  Inverse frequency weights:")
for c in sorted(class_weights_norm.keys()):
    print(f"    Class {c}: {class_weights_norm[c]:.3f} (sqrt: {class_weights_sqrt[c]:.3f})")

# ═══════════════════════════════════════════════════════════════════════
# 4. APPLY SMOTE TO TRAINING SET
# ═══════════════════════════════════════════════════════════════════════
print("\nApplying SMOTE oversampling to training set...")

# Strategy: oversample minority classes to at least 10% of majority
majority_count = max(Counter(y_train).values())
target_counts = {}
for cls, cnt in Counter(y_train).items():
    # Bring each class to at least 15% of majority class count
    target_counts[cls] = max(cnt, int(majority_count * 0.15))

print(f"  Target counts after SMOTE:")
for c in sorted(target_counts.keys()):
    orig = Counter(y_train)[c]
    target = target_counts[c]
    ratio = target / orig
    print(f"    Class {c}: {orig:,} → {target:,} ({ratio:.1f}x)")

smote = SMOTE(
    sampling_strategy=target_counts,
    random_state=42,
    k_neighbors=5,
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"\n  After SMOTE: {X_train_resampled.shape[0]:,} (was {X_train.shape[0]:,})")
print(f"  New distribution:")
dist = Counter(y_train_resampled)
for c in sorted(dist.keys()):
    print(f"    Class {c}: {dist[c]:,} ({dist[c]/len(y_train_resampled)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 5. SAVE ALL SPLITS
# ═══════════════════════════════════════════════════════════════════════
print("\nSaving splits...")

# Save as numpy arrays (efficient for model training)
splits_dir = os.path.join(PROCESSED_DIR, "splits")
os.makedirs(splits_dir, exist_ok=True)

# Original splits (without resampling)
np.save(os.path.join(splits_dir, "X_train.npy"), X_train)
np.save(os.path.join(splits_dir, "y_train.npy"), y_train)
np.save(os.path.join(splits_dir, "X_val.npy"), X_val)
np.save(os.path.join(splits_dir, "y_val.npy"), y_val)
np.save(os.path.join(splits_dir, "X_test.npy"), X_test)
np.save(os.path.join(splits_dir, "y_test.npy"), y_test)

# SMOTE-resampled training set
np.save(os.path.join(splits_dir, "X_train_smote.npy"), X_train_resampled)
np.save(os.path.join(splits_dir, "y_train_smote.npy"), y_train_resampled)

# Weights for test set (for weighted evaluation)
np.save(os.path.join(splits_dir, "weights_test.npy"), weights_test)
np.save(os.path.join(splits_dir, "weights_train.npy"), weights_train)
np.save(os.path.join(splits_dir, "weights_val.npy"), weights_val)

# Save split configuration
split_config = {
    "train_size": int(X_train.shape[0]),
    "val_size": int(X_val.shape[0]),
    "test_size": int(X_test.shape[0]),
    "train_smote_size": int(X_train_resampled.shape[0]),
    "class_weights_inverse": {str(k): float(v) for k, v in class_weights_norm.items()},
    "class_weights_sqrt": {str(k): float(v) for k, v in class_weights_sqrt.items()},
    "smote_target_counts": {str(k): int(v) for k, v in target_counts.items()},
    "train_class_dist": {str(k): int(v) for k, v in Counter(y_train).items()},
    "val_class_dist": {str(k): int(v) for k, v in Counter(y_val).items()},
    "test_class_dist": {str(k): int(v) for k, v in Counter(y_test).items()},
    "train_smote_class_dist": {str(k): int(v) for k, v in Counter(y_train_resampled).items()},
    "random_state": 42,
    "group_column": "CASENUM",
    "no_group_leakage": True,
}
with open(os.path.join(splits_dir, "split_config.json"), "w") as f:
    json.dump(split_config, f, indent=2)

# Also save the full dataframe splits (for feature name access during SHAP)
df_meta = df[["CASENUM", "DATA_YEAR", "WEIGHT", "SEVERITY", "SEVERITY_LABEL"]].copy()
df_meta_train = df_meta.iloc[trainval_idx].iloc[train_idx_rel].reset_index(drop=True)
df_meta_val = df_meta.iloc[trainval_idx].iloc[val_idx_rel].reset_index(drop=True)
df_meta_test = df_meta.iloc[test_idx].reset_index(drop=True)
df_meta_train.to_parquet(os.path.join(splits_dir, "meta_train.parquet"), index=False)
df_meta_val.to_parquet(os.path.join(splits_dir, "meta_val.parquet"), index=False)
df_meta_test.to_parquet(os.path.join(splits_dir, "meta_test.parquet"), index=False)

print(f"  Saved all splits to {splits_dir}/")

# ═══════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SPLIT & BALANCE SUMMARY")
print(f"{'='*70}")
print(f"Original dataset: {len(X):,} records")
print(f"Train:  {X_train.shape[0]:,} records ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Val:    {X_val.shape[0]:,} records ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test:   {X_test.shape[0]:,} records ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"Train (SMOTE): {X_train_resampled.shape[0]:,} records")
print(f"\nClass weights (sqrt-dampened):")
for c in sorted(class_weights_sqrt.keys()):
    print(f"  Class {c}: {class_weights_sqrt[c]:.3f}")
print(f"\nFiles saved in: {splits_dir}/")
print("Done!")
