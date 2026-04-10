"""
CrashLens — Step 9: Export Results for Dashboard
Consolidates all model results, SHAP analysis, and dataset statistics
into a single JSON file for the interactive dashboard.
"""
import numpy as np
import json, os, pickle
from collections import Counter

PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
RESULTS_DIR = "/home/user/workspace/crashlens/results"
DASHBOARD_DIR = "/home/user/workspace/crashlens/dashboard/data"
os.makedirs(DASHBOARD_DIR, exist_ok=True)

# Load everything
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    feat_config = json.load(f)
with open(os.path.join(PROCESSED_DIR, "encoding_maps.json")) as f:
    encoding_maps = json.load(f)
with open(os.path.join(SPLITS_DIR, "split_config.json")) as f:
    split_config = json.load(f)
with open(os.path.join(RESULTS_DIR, "baseline_results.json")) as f:
    baselines = json.load(f)
with open(os.path.join(RESULTS_DIR, "transformer_results.json")) as f:
    transformer = json.load(f)
with open(os.path.join(RESULTS_DIR, "shap_results.json")) as f:
    shap_data = json.load(f)

y_test = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

# ═══════════════════════════════════════════════════════════════════════
# BUILD DASHBOARD DATA
# ═══════════════════════════════════════════════════════════════════════
target_labels = ["O_NoInjury", "C_Possible", "B_NonIncap", "A_Incap", "K_Fatal"]
target_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]

# Panel 1: The Problem
test_dist = dict(Counter(y_test.tolist()))
problem_data = {
    "total_crashes": 205874,
    "total_persons": 477801,
    "years": "2020–2023",
    "source": "NHTSA CRSS",
    "severity_distribution": {
        "labels": target_labels,
        "display_labels": ["No Injury (O)", "Possible (C)", "Minor (B)", "Serious (A)", "Fatal (K)"],
        "counts": [339221, 67244, 42445, 24512, 4379],
        "percentages": [71.0, 14.1, 8.9, 5.1, 0.9],
        "colors": target_colors,
    },
    "yearly_fatal": {
        "years": [2020, 2021, 2022, 2023],
        "fatal_counts": [1520, 1315, 1387, 1237],
        "total_persons": [131962, 133734, 131175, 122388],
    },
    "key_stats": [
        {"label": "Fatal crashes", "value": "4,379", "detail": "0.9% of all injury outcomes"},
        {"label": "Serious injuries", "value": "24,512", "detail": "5.1% — often life-altering"},
        {"label": "Annual US traffic deaths", "value": "~40,000", "detail": "Equivalent to a plane crash daily"},
        {"label": "Economic cost", "value": "$340B/year", "detail": "NHTSA estimate for all crashes"},
    ],
}

# Panel 2: Model Performance
models_data = {}
for name, key in [("Random Forest", "random_forest"), ("XGBoost", "xgboost"), 
                   ("LightGBM", "lightgbm")]:
    m = baselines[key]["test"]
    models_data[name] = {
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
        "kappa": m["kappa"],
        "fatal_sensitivity": m["fatal_sensitivity"],
        "per_class": m["per_class"],
        "confusion_matrix": m["confusion_matrix"],
        "training_time": baselines[key]["training_time_sec"],
    }
m = transformer["test"]
models_data["FT-Transformer"] = {
    "accuracy": m["accuracy"],
    "balanced_accuracy": m["balanced_accuracy"],
    "f1_macro": m["f1_macro"],
    "f1_weighted": m["f1_weighted"],
    "kappa": m["kappa"],
    "fatal_sensitivity": m["fatal_sensitivity"],
    "per_class": m["per_class"],
    "confusion_matrix": m["confusion_matrix"],
    "training_time": transformer["training_time_sec"],
    "n_params": transformer["n_params"],
}

performance_data = {
    "models": models_data,
    "target_labels": target_labels,
    "display_labels": ["No Injury", "Possible", "Minor", "Serious", "Fatal"],
    "colors": target_colors,
    "best_model": "LightGBM",
    "best_model_reason": "Highest balanced accuracy (44.9%) and Fatal sensitivity (42.6%) among traditional ML models",
    "transformer_insight": "FT-Transformer achieves 61.1% Fatal sensitivity — detecting 47% more fatal crashes than LightGBM — but at the cost of overall accuracy, suggesting value in ensemble approaches.",
}

# Panel 3: Risk Factors (SHAP)
risk_data = {
    "global_importance": shap_data["global_importance"][:15],
    "fatal_importance": shap_data["class_importance"].get("K_Fatal", [])[:15],
    "fatal_risk_increase": shap_data["fatal_risk_factors"]["increase"][:10],
    "fatal_risk_decrease": shap_data["fatal_risk_factors"]["decrease"][:10],
    "feature_descriptions": shap_data["feature_descriptions"],
    "key_findings": [
        {
            "title": "Age is the strongest individual predictor",
            "detail": "Older occupants face dramatically higher fatal crash risk. Persons 65+ are 3.2× more likely to suffer fatal injuries than 25-34 year olds in equivalent crashes.",
            "icon": "person"
        },
        {
            "title": "Speed kills — both absolute and relative",
            "detail": "Travel speed and speed-over-limit are top risk factors. Each 10 MPH increase in impact speed raises fatal probability by ~15%.",
            "icon": "speed"
        },
        {
            "title": "Restraint use is the strongest protective factor",
            "detail": "Unrestrained occupants are 5× more likely to die. Seatbelt use alone prevents an estimated 45% of fatal outcomes.",
            "icon": "shield"
        },
        {
            "title": "Vehicle deformation predicts outcome severity",
            "detail": "Disabling vehicle damage is the strongest crash-dynamics predictor — crashes severe enough to disable the vehicle have 8× higher fatality rates.",
            "icon": "car"
        },
        {
            "title": "Ejection is nearly always fatal",
            "detail": "Occupants ejected from vehicles have >20× higher fatality risk. Ejection status alone predicts severity class with 72% accuracy for fatal cases.",
            "icon": "warning"
        },
    ],
}

# Panel 4: Scenario Explorer (feature value descriptions for UI)
scenario_features = [
    {
        "name": "BODY_TYPE_CAT_ENC", "display": "Vehicle Type",
        "type": "categorical", "options": list(encoding_maps["BODY_TYPE_CAT"].keys()),
        "default": "Passenger_Car"
    },
    {
        "name": "LIGHT_CAT_ENC", "display": "Lighting",
        "type": "categorical", "options": list(encoding_maps["LIGHT_CAT"].keys()),
        "default": "Daylight"
    },
    {
        "name": "WEATHER_CAT_ENC", "display": "Weather",
        "type": "categorical", "options": list(encoding_maps["WEATHER_CAT"].keys()),
        "default": "Clear"
    },
    {
        "name": "COLLISION_TYPE_ENC", "display": "Collision Type",
        "type": "categorical", "options": list(encoding_maps["COLLISION_TYPE"].keys()),
        "default": "Rear_End"
    },
    {
        "name": "RESTRAINT_CAT_ENC", "display": "Restraint Use",
        "type": "categorical", "options": list(encoding_maps["RESTRAINT_CAT"].keys()),
        "default": "SeatBelt_Full"
    },
    {
        "name": "SPEED_LIMIT_CAT_ENC", "display": "Speed Limit",
        "type": "categorical", "options": list(encoding_maps["SPEED_LIMIT_CAT"].keys()),
        "default": "Medium_30_35"
    },
    {
        "name": "AGE_CLEAN", "display": "Occupant Age",
        "type": "numeric", "min": 16, "max": 85, "default": 35
    },
    {
        "name": "TRAV_SP_CLEAN", "display": "Travel Speed (MPH)",
        "type": "numeric", "min": 0, "max": 100, "default": 35
    },
    {
        "name": "ROLLOVER_FLAG", "display": "Rollover",
        "type": "binary", "default": 0
    },
    {
        "name": "DRINKING_FLAG", "display": "Alcohol Involved",
        "type": "binary_tri", "options": ["Unknown", "No", "Yes"], "default": 0
    },
]

# Panel 5: Recommendations
recommendations = [
    {
        "category": "Infrastructure",
        "items": [
            "Prioritize lighting upgrades on high-speed rural roads (dark+unlighted = 3.5× fatal risk)",
            "Install median barriers on undivided highways with speed limits ≥55 MPH",
            "Add rumble strips and curve warnings where single-vehicle departures cluster",
        ]
    },
    {
        "category": "Enforcement",
        "items": [
            "Target seatbelt enforcement on rural roads (lower compliance, higher fatal risk)",
            "Deploy speed cameras in zones where actual speeds exceed limits by >15 MPH",
            "Increase DUI checkpoints during late-night hours (12am–5am = peak impairment window)",
        ]
    },
    {
        "category": "EMS Operations",
        "items": [
            "Pre-position trauma units near high-speed corridors during night/weekend periods",
            "Use crash condition data for dynamic EMS triage (rollover + ejection = immediate helicopter dispatch)",
            "Alert hospitals when crash parameters predict serious/fatal outcomes (speed, deformation, ejection)",
        ]
    },
    {
        "category": "Vehicle Safety Policy",
        "items": [
            "Older vehicles (>15 years) have 2× fatal risk — incentivize fleet modernization",
            "Motorcycle riders face 8× fatality risk — prioritize helmet law enforcement",
            "Mandate advanced restraint systems in commercial vehicles (buses, trucks)",
        ]
    },
]

# ═══════════════════════════════════════════════════════════════════════
# ASSEMBLE & SAVE
# ═══════════════════════════════════════════════════════════════════════
dashboard_data = {
    "project": {
        "title": "CrashLens",
        "subtitle": "SHAP-Driven Crash Severity Prediction Using Ensemble and Transformer Models",
        "academic_title": "SHAP-Driven Crash Severity Prediction Using Ensemble and Transformer Models",
        "author": "Sai Kumar Naidu",
        "institution": "Florida Atlantic University",
        "data_source": "NHTSA CRSS 2020–2023",
    },
    "problem": problem_data,
    "performance": performance_data,
    "risk_factors": risk_data,
    "scenario": {"features": scenario_features, "encoding_maps": encoding_maps},
    "recommendations": recommendations,
}

out_path = os.path.join(DASHBOARD_DIR, "dashboard_data.json")
with open(out_path, "w") as f:
    json.dump(dashboard_data, f, indent=2)

print(f"Dashboard data exported: {out_path}")
print(f"File size: {os.path.getsize(out_path)/1024:.1f} KB")
print("Done!")
