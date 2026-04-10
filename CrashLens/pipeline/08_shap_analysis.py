"""
CrashLens — Step 8: SHAP Explainability Analysis
Computes SHAP values for LightGBM (best balanced model) to identify
which crash conditions most strongly predict fatal/severe outcomes.
"""
import numpy as np
import json, os, pickle, warnings, gc
warnings.filterwarnings("ignore")
import shap

SPLITS_DIR = "/home/user/workspace/crashlens/data/processed/splits"
PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"
RESULTS_DIR = "/home/user/workspace/crashlens/results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL & DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading model and data...")
with open(os.path.join(MODELS_DIR, "lightgbm.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    config = json.load(f)
with open(os.path.join(PROCESSED_DIR, "encoding_maps.json")) as f:
    encoding_maps = json.load(f)

feature_names = config["model_features"]
target_labels = ["O_NoInjury", "C_Possible", "B_NonIncap", "A_Incap", "K_Fatal"]

X_test = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))
print(f"  Test set: {X_test.shape}")

# Use a sample for SHAP (full test set too large)
rng = np.random.RandomState(42)
sample_size = 3000
sample_idx = rng.choice(len(X_test), sample_size, replace=False)
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]
print(f"  SHAP sample: {sample_size}")

# ═══════════════════════════════════════════════════════════════════════
# 2. COMPUTE SHAP VALUES (TreeExplainer — fast for LightGBM)
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)  # Returns list of arrays, one per class

# shap_values is a list of 5 arrays, each (3000, 37) — one per severity class
print(f"  SHAP values shape: {len(shap_values)} classes × {shap_values[0].shape}")

# ═══════════════════════════════════════════════════════════════════════
# 3. GLOBAL FEATURE IMPORTANCE (mean |SHAP| per class)
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing global feature importance...")

# Overall importance (averaged across all classes)
shap_abs_all = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (3000, 37)
global_importance = np.mean(shap_abs_all, axis=0)  # (37,)
importance_ranking = sorted(zip(feature_names, global_importance), key=lambda x: -x[1])

print("\n  Global Feature Importance (mean |SHAP|):")
for i, (feat, imp) in enumerate(importance_ranking[:15]):
    print(f"    {i+1:2d}. {feat:<30s} {imp:.4f}")

# Per-class importance (which features matter most for Fatal vs No Injury)
class_importance = {}
for cls_idx, cls_name in enumerate(target_labels):
    cls_shap = np.abs(shap_values[cls_idx])
    cls_imp = np.mean(cls_shap, axis=0)
    ranking = sorted(zip(feature_names, cls_imp.tolist()), key=lambda x: -x[1])
    class_importance[cls_name] = ranking
    
    print(f"\n  Top 10 for {cls_name}:")
    for i, (feat, imp) in enumerate(ranking[:10]):
        print(f"    {i+1:2d}. {feat:<30s} {imp:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. DIRECTIONAL ANALYSIS — Fatal (K) class
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FATAL CRASH RISK FACTOR ANALYSIS")
print("="*60)

fatal_shap = shap_values[4]  # Class 4 = Fatal
fatal_mean_shap = np.mean(fatal_shap, axis=0)

# Positive SHAP = increases fatal probability
# Negative SHAP = decreases fatal probability
print("\n  Features that INCREASE fatal crash risk:")
fatal_risk_increase = sorted(zip(feature_names, fatal_mean_shap), key=lambda x: -x[1])
for feat, val in fatal_risk_increase[:10]:
    print(f"    {feat:<30s} SHAP={val:+.4f}")

print("\n  Features that DECREASE fatal crash risk:")
for feat, val in fatal_risk_increase[-10:]:
    print(f"    {feat:<30s} SHAP={val:+.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. FEATURE INTERACTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\nAnalyzing feature interactions...")

# For each top feature, compute correlation with other features' SHAP values
top_features_idx = [feature_names.index(f) for f, _ in importance_ranking[:10]]

# ═══════════════════════════════════════════════════════════════════════
# 6. SAVE SHAP RESULTS
# ═══════════════════════════════════════════════════════════════════════
print("\nSaving SHAP results...")

# Create human-readable feature descriptions
feature_descriptions = {
    "AGE_CLEAN": "Person Age",
    "TRAV_SP_CLEAN": "Travel Speed (MPH)",
    "SPEED_OVER_LIMIT": "Speed Over Limit (MPH)",
    "VEHICLE_AGE": "Vehicle Age (Years)",
    "VE_TOTAL": "Total Vehicles in Crash",
    "NUMOCCS": "Occupants in Vehicle",
    "VSPD_LIM": "Posted Speed Limit",
    "VNUM_LAN": "Number of Lanes",
    "NUM_CRASH_FACTORS": "Crash-Related Factors",
    "BODY_TYPE_CAT_ENC": "Vehicle Body Type",
    "LIGHT_CAT_ENC": "Lighting Condition",
    "WEATHER_CAT_ENC": "Weather Condition",
    "COLLISION_TYPE_ENC": "Collision Type",
    "AGE_GROUP_ENC": "Age Group",
    "SEX_CLEAN_ENC": "Sex",
    "RESTRAINT_CAT_ENC": "Restraint Use",
    "AIRBAG_CAT_ENC": "Airbag Deployment",
    "EJECTION_CAT_ENC": "Ejection Status",
    "DEFORMATION_CAT_ENC": "Vehicle Deformation",
    "SURFACE_CAT_ENC": "Road Surface",
    "SPEED_LIMIT_CAT_ENC": "Speed Limit Category",
    "TIME_PERIOD_ENC": "Time of Day",
    "IS_WEEKEND": "Weekend",
    "IS_DRIVER": "Is Driver",
    "MULTI_VEHICLE": "Multi-Vehicle Crash",
    "ROLLOVER_FLAG": "Rollover",
    "SPEED_RELATED": "Speed-Related",
    "IN_WORK_ZONE": "Work Zone",
    "AT_JUNCTION": "At Junction",
    "DISTRACTED": "Driver Distracted",
    "DRIVER_IMPAIRED": "Driver Impaired",
    "DRINKING_FLAG": "Alcohol Involved",
    "URBANICITY": "Urban/Rural",
    "REGION": "U.S. Region",
    "DAY_WEEK": "Day of Week",
    "HOUR": "Hour of Day",
    "MONTH": "Month",
}

shap_results = {
    "global_importance": [
        {"feature": feat, "display_name": feature_descriptions.get(feat, feat),
         "importance": float(imp)}
        for feat, imp in importance_ranking
    ],
    "class_importance": {
        cls_name: [
            {"feature": feat, "display_name": feature_descriptions.get(feat, feat),
             "importance": float(imp)}
            for feat, imp in ranking[:15]
        ]
        for cls_name, ranking in class_importance.items()
    },
    "fatal_risk_factors": {
        "increase": [
            {"feature": feat, "display_name": feature_descriptions.get(feat, feat),
             "mean_shap": float(val)}
            for feat, val in fatal_risk_increase[:15]
        ],
        "decrease": [
            {"feature": feat, "display_name": feature_descriptions.get(feat, feat),
             "mean_shap": float(val)}
            for feat, val in fatal_risk_increase[-15:]
        ],
    },
    "feature_descriptions": feature_descriptions,
    "sample_size": sample_size,
    "model": "LightGBM",
}

with open(os.path.join(RESULTS_DIR, "shap_results.json"), "w") as f:
    json.dump(shap_results, f, indent=2)

# Save raw SHAP values for dashboard visualizations
np.save(os.path.join(RESULTS_DIR, "shap_values_fatal.npy"), fatal_shap)
np.save(os.path.join(RESULTS_DIR, "shap_sample_X.npy"), X_sample)
np.save(os.path.join(RESULTS_DIR, "shap_sample_y.npy"), y_sample)

print(f"\nSaved to: {RESULTS_DIR}/shap_results.json")
print("Done!")
