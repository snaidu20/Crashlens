"""
CrashLens — Step 3: Clean, Encode & Prepare Final Feature Set
Handles missing values, encodes categoricals, removes redundant columns,
and exports a model-ready dataset.
"""
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD MERGED DATASET
# ═══════════════════════════════════════════════════════════════════════
print("Loading merged dataset...")
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "crashlens_merged.parquet"))
print(f"  Shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════
# 2. SELECT FINAL FEATURES (drop raw codes, keep engineered ones)
# ═══════════════════════════════════════════════════════════════════════
print("\nSelecting final features...")

# --- Numeric features (continuous / ordinal) ---
numeric_features = [
    "AGE_CLEAN",          # Person age (cleaned)
    "TRAV_SP_CLEAN",      # Travel speed (cleaned)
    "SPEED_OVER_LIMIT",   # Speed over posted limit
    "VEHICLE_AGE",        # Vehicle age in years
    "VE_TOTAL",           # Total vehicles in crash
    "NUMOCCS",            # Number of occupants in vehicle
    "VSPD_LIM",           # Posted speed limit (numeric)
    "VNUM_LAN",           # Number of lanes
    "NUM_CRASH_FACTORS",  # Count of crash-related factors
    "NUM_DRIVER_RF",       # Count of driver risk factor entries
    "NUM_VIOLATIONS",      # Count of traffic violations charged
]

# --- Categorical features (engineered categories) ---
categorical_features = [
    "BODY_TYPE_CAT",      # Vehicle body type
    "LIGHT_CAT",          # Lighting condition
    "WEATHER_CAT",        # Weather condition
    "COLLISION_TYPE",     # Manner of collision
    "AGE_GROUP",          # Age group
    "SEX_CLEAN",          # Sex
    "RESTRAINT_CAT",      # Restraint use
    "AIRBAG_CAT",         # Airbag deployment
    "EJECTION_CAT",       # Ejection status
    "DEFORMATION_CAT",    # Vehicle deformation
    "SURFACE_CAT",        # Road surface condition
    "SPEED_LIMIT_CAT",    # Speed limit category
    "TIME_PERIOD",        # Time of day
    "VTRAFCON_CAT",        # Traffic control device at scene
]

# --- Binary features ---
binary_features = [
    "IS_WEEKEND",         # Weekend flag
    "IS_DRIVER",          # Driver vs passenger
    "MULTI_VEHICLE",      # Multi-vehicle crash
    "ROLLOVER_FLAG",      # Rollover occurred
    "SPEED_RELATED",      # Speed-related crash
    "IN_WORK_ZONE",       # Work zone
    "AT_JUNCTION",        # At junction/intersection
    "DISTRACTED",         # Driver distracted
    "DRIVER_IMPAIRED",    # Driver impaired
    "DRINKING_FLAG",      # Drinking involved (-1=Unknown)
    "HAS_PRE_CRASH_MANEUVER",  # Active pre-crash maneuver (turn/lane change/backup)
    "HAS_DRIVER_RF",      # Driver had coded risk factor
    "HAS_VIOLATION",      # Driver had a cited traffic violation
    "HIT_RUN_FLAG",       # Driver fled scene
]

# --- Ordinal features (raw CRSS codes with meaningful ordering) ---
ordinal_features = [
    "URBANICITY",         # 1=Urban, 2=Rural
    "REGION",             # 1-4 (NE, MW, S, W)
    "DAY_WEEK",           # 1-7 (Sun-Sat)
    "HOUR",               # 0-23
    "MONTH",              # 1-12
]

# --- Meta columns (not features, but needed for splitting/analysis) ---
meta_cols = ["CASENUM", "DATA_YEAR", "WEIGHT", "SEVERITY", "SEVERITY_LABEL"]

all_features = numeric_features + categorical_features + binary_features + ordinal_features
all_cols = meta_cols + all_features

# Filter to selected columns
df_sel = df[all_cols].copy()
print(f"  Selected {len(all_features)} features + {len(meta_cols)} meta columns")
print(f"  Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}, "
      f"Binary: {len(binary_features)}, Ordinal: {len(ordinal_features)}")

# ═══════════════════════════════════════════════════════════════════════
# 3. HANDLE MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════
print("\nHandling missing values...")

# -- Numeric features: replace CRSS coded unknowns with NaN, then impute --
# VSPD_LIM: 98, 99 = unknown
df_sel.loc[df_sel["VSPD_LIM"].isin([98, 99]), "VSPD_LIM"] = np.nan
# VNUM_LAN: 8, 9 = unknown
df_sel.loc[df_sel["VNUM_LAN"].isin([8, 9]), "VNUM_LAN"] = np.nan
# NUMOCCS: 99, 98 = unknown  
df_sel.loc[df_sel["NUMOCCS"].isin([98, 99]), "NUMOCCS"] = np.nan
# HOUR: 99, 98 = unknown
df_sel.loc[df_sel["HOUR"].isin([98, 99]), "HOUR"] = np.nan
# MONTH: 99 = unknown (unlikely but just in case)

# Print missing value summary before imputation
print("\n  Missing value summary (before imputation):")
for col in all_features:
    na_count = df_sel[col].isna().sum()
    if na_count > 0:
        pct = na_count / len(df_sel) * 100
        print(f"    {col}: {na_count:,} ({pct:.1f}%)")

# -- Strategy: --
# For numeric: median imputation (robust to outliers)
# For categorical: "Unknown" category (already handled in engineering)
# For binary: mode imputation (most common value)

# Numeric imputation with median
for col in numeric_features:
    median_val = df_sel[col].median()
    na_before = df_sel[col].isna().sum()
    df_sel[col] = df_sel[col].fillna(median_val)
    if na_before > 0:
        print(f"    Imputed {col}: {na_before:,} NaN → median={median_val:.1f}")

# Ordinal: impute HOUR with median, rest should be clean
for col in ordinal_features:
    if df_sel[col].isna().any():
        median_val = df_sel[col].median()
        na_before = df_sel[col].isna().sum()
        df_sel[col] = df_sel[col].fillna(median_val)
        print(f"    Imputed {col}: {na_before:,} NaN → median={median_val:.1f}")

# Categorical: already have "Unknown" category from feature engineering
# Binary: DRINKING_FLAG has -1 for unknown — keep as separate value
# (the model can learn from the pattern of missingness)

# ═══════════════════════════════════════════════════════════════════════
# 4. ENCODE CATEGORICALS
# ═══════════════════════════════════════════════════════════════════════
print("\nEncoding categoricals...")

# For tree-based models (RF, XGBoost, LightGBM): ordinal/label encoding works fine
# For FT-Transformer: needs integer indices for embedding layers
# → Use label encoding for all categoricals (each unique value gets an integer)

encoding_maps = {}
for col in categorical_features:
    unique_vals = sorted(df_sel[col].unique())
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    encoding_maps[col] = val_to_idx
    df_sel[col + "_ENC"] = df_sel[col].map(val_to_idx)
    print(f"  {col}: {len(unique_vals)} categories → encoded")

# Save encoding maps for later use (dashboard interpretation, etc.)
encoding_maps_serializable = {k: {str(kk): vv for kk, vv in v.items()} 
                               for k, v in encoding_maps.items()}
with open(os.path.join(PROCESSED_DIR, "encoding_maps.json"), "w") as f:
    json.dump(encoding_maps_serializable, f, indent=2)
print(f"  Saved encoding maps to encoding_maps.json")

# ═══════════════════════════════════════════════════════════════════════
# 5. REMOVE OUTLIERS & CLEAN EDGE CASES
# ═══════════════════════════════════════════════════════════════════════
print("\nCleaning outliers...")

# Speed: cap at reasonable values
df_sel.loc[df_sel["TRAV_SP_CLEAN"] > 150, "TRAV_SP_CLEAN"] = 150
df_sel.loc[df_sel["TRAV_SP_CLEAN"] < 0, "TRAV_SP_CLEAN"] = 0

# Age: cap at reasonable values
df_sel.loc[df_sel["AGE_CLEAN"] > 100, "AGE_CLEAN"] = 100
df_sel.loc[df_sel["AGE_CLEAN"] < 0, "AGE_CLEAN"] = 0

# Vehicle age: already capped at 50 in engineering step

# Speed over limit: cap at ±50
df_sel["SPEED_OVER_LIMIT"] = df_sel["SPEED_OVER_LIMIT"].clip(-50, 80)

# Number of lanes: reasonable range
df_sel.loc[df_sel["VNUM_LAN"] > 8, "VNUM_LAN"] = 8

# NUMOCCS: cap at 10
df_sel.loc[df_sel["NUMOCCS"] > 10, "NUMOCCS"] = 10

print(f"  Outlier cleaning complete")

# ═══════════════════════════════════════════════════════════════════════
# 6. BUILD FINAL FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding final feature matrix...")

# For the models, we'll use:
# - Numeric features as-is
# - Encoded categorical features (integer indices)
# - Binary features as-is
# - Ordinal features as-is

encoded_cat_features = [c + "_ENC" for c in categorical_features]
model_features = numeric_features + encoded_cat_features + binary_features + ordinal_features

# Define which features are numerical vs categorical for the FT-Transformer
numerical_feature_indices = list(range(len(numeric_features)))
cat_start = len(numeric_features)
categorical_feature_indices = list(range(cat_start, cat_start + len(encoded_cat_features)))
binary_start = cat_start + len(encoded_cat_features)
binary_feature_indices = list(range(binary_start, binary_start + len(binary_features)))
ordinal_start = binary_start + len(binary_features)
ordinal_feature_indices = list(range(ordinal_start, ordinal_start + len(ordinal_features)))

# Category cardinalities (for embedding layers)
cat_cardinalities = [len(encoding_maps[c]) for c in categorical_features]

# Final check: no NaN in model features
print("\n  Final NaN check in model features:")
total_nan = 0
for col in model_features:
    na = df_sel[col].isna().sum()
    if na > 0:
        print(f"    WARNING: {col} still has {na:,} NaN")
        total_nan += na
if total_nan == 0:
    print("    ✓ No NaN values in any model feature")

# ═══════════════════════════════════════════════════════════════════════
# 7. SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════
print("\nSaving final dataset...")

# Save full dataset with all columns
out_path = os.path.join(PROCESSED_DIR, "crashlens_clean.parquet")
df_sel.to_parquet(out_path, index=False)
print(f"  Full dataset: {out_path} — {df_sel.shape}")

# Save feature configuration
feature_config = {
    "model_features": model_features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "encoded_categorical_features": encoded_cat_features,
    "binary_features": binary_features,
    "ordinal_features": ordinal_features,
    "numerical_feature_indices": numerical_feature_indices,
    "categorical_feature_indices": categorical_feature_indices,
    "binary_feature_indices": binary_feature_indices,
    "ordinal_feature_indices": ordinal_feature_indices,
    "cat_cardinalities": cat_cardinalities,
    "target_column": "SEVERITY",
    "target_labels": {str(k): v for k, v in {0: "O_NoInjury", 1: "C_Possible", 
                      2: "B_NonIncap", 3: "A_Incap", 4: "K_Fatal"}.items()},
    "n_classes": 5,
    "n_features": len(model_features),
    "n_samples": len(df_sel),
}
config_path = os.path.join(PROCESSED_DIR, "feature_config.json")
with open(config_path, "w") as f:
    json.dump(feature_config, f, indent=2)
print(f"  Feature config: {config_path}")

# ═══════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FINAL CLEAN DATASET SUMMARY")
print(f"{'='*70}")
print(f"Records: {len(df_sel):,}")
print(f"Model features: {len(model_features)}")
print(f"  Numeric: {len(numeric_features)}")
print(f"  Categorical (encoded): {len(encoded_cat_features)}")
print(f"  Binary: {len(binary_features)}")
print(f"  Ordinal: {len(ordinal_features)}")
print(f"\nTarget distribution:")
for sev in sorted(df_sel["SEVERITY"].unique()):
    cnt = (df_sel["SEVERITY"] == sev).sum()
    pct = cnt / len(df_sel) * 100
    label = feature_config["target_labels"][str(sev)]
    print(f"  {sev} ({label}): {cnt:,} ({pct:.1f}%)")
print(f"\nCategory cardinalities: {dict(zip(categorical_features, cat_cardinalities))}")
print("\nDone!")
