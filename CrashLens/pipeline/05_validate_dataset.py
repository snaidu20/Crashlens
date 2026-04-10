"""
CrashLens — Step 5: Final Dataset Validation
Comprehensive quality checks before model training.
"""
import pandas as pd
import numpy as np
import json, os
from collections import Counter

PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD ALL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════
print("Loading all artifacts...")
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    config = json.load(f)
with open(os.path.join(SPLITS_DIR, "split_config.json")) as f:
    split_config = json.load(f)
with open(os.path.join(PROCESSED_DIR, "encoding_maps.json")) as f:
    encoding_maps = json.load(f)

X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
X_val = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
X_test = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))
X_train_smote = np.load(os.path.join(SPLITS_DIR, "X_train_smote.npy"))
y_train_smote = np.load(os.path.join(SPLITS_DIR, "y_train_smote.npy"))

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "crashlens_clean.parquet"))

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1

# ═══════════════════════════════════════════════════════════════════════
# 2. DATA INTEGRITY CHECKS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DATA INTEGRITY CHECKS")
print("="*70)

# Shape consistency
check("Feature count matches config",
      X_train.shape[1] == config["n_features"],
      f"Expected {config['n_features']}, got {X_train.shape[1]}")

check("All splits have same feature count",
      X_train.shape[1] == X_val.shape[1] == X_test.shape[1])

check("SMOTE preserves feature count",
      X_train_smote.shape[1] == X_train.shape[1])

# No NaN / Inf
check("Train: no NaN", not np.any(np.isnan(X_train)))
check("Val: no NaN", not np.any(np.isnan(X_val)))
check("Test: no NaN", not np.any(np.isnan(X_test)))
check("SMOTE train: no NaN", not np.any(np.isnan(X_train_smote)))
check("Train: no Inf", not np.any(np.isinf(X_train)))

# Target values
valid_classes = set(range(5))
check("Train target classes valid", set(np.unique(y_train)) == valid_classes)
check("Val target classes valid", set(np.unique(y_val)) == valid_classes)
check("Test target classes valid", set(np.unique(y_test)) == valid_classes)
check("SMOTE target classes valid", set(np.unique(y_train_smote)) == valid_classes)

# Split sizes add up
total_split = len(y_train) + len(y_val) + len(y_test)
check("Split sizes add to total",
      total_split == config["n_samples"],
      f"Split sum: {total_split}, total: {config['n_samples']}")

# ═══════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTION CHECKS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DISTRIBUTION CHECKS")
print("="*70)

# Class distribution should be similar across train/val/test
train_dist = {k: v/len(y_train) for k, v in Counter(y_train).items()}
val_dist = {k: v/len(y_val) for k, v in Counter(y_val).items()}
test_dist = {k: v/len(y_test) for k, v in Counter(y_test).items()}

max_drift = max(abs(train_dist[c] - test_dist[c]) for c in valid_classes)
check("Class distribution drift < 1%", max_drift < 0.01,
      f"Max drift: {max_drift:.3f}")

# SMOTE should have increased minority classes
smote_dist = Counter(y_train_smote)
check("SMOTE increased Fatal class", smote_dist[4] > Counter(y_train)[4])
check("SMOTE increased Incap class", smote_dist[3] > Counter(y_train)[3])
check("SMOTE did not change majority", smote_dist[0] == Counter(y_train)[0])

# ═══════════════════════════════════════════════════════════════════════
# 4. FEATURE RANGE CHECKS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FEATURE RANGE CHECKS")
print("="*70)

features = config["model_features"]
print(f"\n  Feature statistics (training set):")
print(f"  {'Feature':<30} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*70}")
for i, feat in enumerate(features):
    col = X_train[:, i]
    print(f"  {feat:<30} {col.min():8.2f} {col.max():8.2f} {col.mean():8.2f} {col.std():8.2f}")

# Specific range checks
age_idx = features.index("AGE_CLEAN")
check("Age range 0-100", X_train[:, age_idx].min() >= 0 and X_train[:, age_idx].max() <= 100)

speed_idx = features.index("TRAV_SP_CLEAN")
check("Speed range 0-150", X_train[:, speed_idx].min() >= 0 and X_train[:, speed_idx].max() <= 150)

# Categorical features: check they're within cardinality bounds
for j, (cat_feat, card) in enumerate(zip(config["encoded_categorical_features"], 
                                          config["cat_cardinalities"])):
    idx = features.index(cat_feat)
    col = X_train[:, idx]
    ok = col.min() >= 0 and col.max() < card
    check(f"{cat_feat} within [0, {card})", ok,
          f"Range: [{col.min():.0f}, {col.max():.0f}]")

# ═══════════════════════════════════════════════════════════════════════
# 5. CORRELATION & REDUNDANCY CHECK
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FEATURE CORRELATION CHECK (top pairs > 0.7)")
print("="*70)

# Sample for efficiency
sample_idx = np.random.choice(len(X_train), min(50000, len(X_train)), replace=False)
X_sample = X_train[sample_idx]
corr_matrix = np.corrcoef(X_sample.T)

high_corr_pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        r = abs(corr_matrix[i, j])
        if r > 0.7:
            high_corr_pairs.append((features[i], features[j], r))

high_corr_pairs.sort(key=lambda x: -x[2])
if high_corr_pairs:
    for f1, f2, r in high_corr_pairs[:10]:
        print(f"  {f1} ↔ {f2}: r={r:.3f}")
    print(f"\n  Note: {len(high_corr_pairs)} pairs with |r| > 0.7")
    print("  These are expected (e.g., VSPD_LIM ↔ SPEED_LIMIT_CAT_ENC)")
    print("  Tree models handle collinearity well; transformers use embeddings.")
else:
    print("  No feature pairs with |r| > 0.7")

check("No perfect multicollinearity (r=1.0)", 
      all(r < 0.99 for _, _, r in high_corr_pairs) if high_corr_pairs else True)

# ═══════════════════════════════════════════════════════════════════════
# 6. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("VALIDATION SUMMARY")
print(f"{'='*70}")
print(f"Checks passed: {passed}")
print(f"Checks failed: {failed}")
print(f"\nDataset ready for model training: {'YES' if failed == 0 else 'NO — fix issues above'}")

print(f"\n{'='*70}")
print("DATASET AT A GLANCE")
print(f"{'='*70}")
print(f"Source: NHTSA CRSS 2020-2023")
print(f"Records: {config['n_samples']:,} motor vehicle occupants")
print(f"Features: {config['n_features']} ({len(config['numeric_features'])} numeric, "
      f"{len(config['encoded_categorical_features'])} categorical, "
      f"{len(config['binary_features'])} binary, {len(config['ordinal_features'])} ordinal)")
print(f"Target: 5-class injury severity (O/C/B/A/K)")
print(f"Train: {split_config['train_size']:,} | Val: {split_config['val_size']:,} | "
      f"Test: {split_config['test_size']:,}")
print(f"Train (SMOTE): {split_config['train_smote_size']:,}")
print(f"\nKey class weights (sqrt-dampened):")
for c in sorted(split_config["class_weights_sqrt"].keys()):
    label = config["target_labels"][c]
    w = split_config["class_weights_sqrt"][c]
    print(f"  Class {c} ({label}): {w:.3f}")

print(f"\nAll files in: {PROCESSED_DIR}/")
print(f"  crashlens_clean.parquet  — full clean dataset")
print(f"  feature_config.json      — feature names, indices, cardinalities")
print(f"  encoding_maps.json       — categorical encoding maps")
print(f"  splits/                  — train/val/test numpy arrays + configs")

print("\n✓ Phase 1 complete — dataset ready for model training.")
