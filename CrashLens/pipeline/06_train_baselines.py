"""
CrashLens — Step 6: Train Baseline Models
Trains Random Forest, XGBoost, and LightGBM with class-weighted loss.
Evaluates on validation set and saves results.
"""
import numpy as np
import json, os, time, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, balanced_accuracy_score, cohen_kappa_score
)
import xgboost as xgb
import lightgbm as lgb

SPLITS_DIR = "/home/user/workspace/crashlens/data/processed/splits"
PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"
RESULTS_DIR = "/home/user/workspace/crashlens/results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading data...")
X_train = np.load(os.path.join(SPLITS_DIR, "X_train_smote.npy"))
y_train = np.load(os.path.join(SPLITS_DIR, "y_train_smote.npy"))
X_val = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
X_test = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

with open(os.path.join(SPLITS_DIR, "split_config.json")) as f:
    split_config = json.load(f)
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    feat_config = json.load(f)

class_weights_dict = {int(k): v for k, v in split_config["class_weights_sqrt"].items()}
feature_names = feat_config["model_features"]
target_labels = ["O_NoInjury", "C_Possible", "B_NonIncap", "A_Incap", "K_Fatal"]

print(f"  Train (SMOTE): {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════════════
# 2. EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════
def evaluate_model(name, model, X_eval, y_eval, dataset_label="Val"):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_eval)
    
    acc = accuracy_score(y_eval, y_pred)
    bal_acc = balanced_accuracy_score(y_eval, y_pred)
    f1_macro = f1_score(y_eval, y_pred, average="macro")
    f1_weighted = f1_score(y_eval, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_eval, y_pred)
    
    # Per-class metrics
    report = classification_report(y_eval, y_pred, target_names=target_labels, 
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{name} — {dataset_label} Set Results")
    print(f"{'='*60}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f"\n  Per-class F1:")
    for label in target_labels:
        if label in report:
            r = report[label]
            print(f"    {label:20s} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  N={r['support']}")
    
    print(f"\n  Confusion Matrix:")
    print(f"  {'':15s} " + " ".join(f"{l[:6]:>8s}" for l in target_labels))
    for i, row in enumerate(cm):
        print(f"  {target_labels[i]:15s} " + " ".join(f"{v:8d}" for v in row))
    
    # Fatal class sensitivity (critical metric)
    fatal_recall = report.get("K_Fatal", {}).get("recall", 0)
    print(f"\n  ★ Fatal (K) Sensitivity: {fatal_recall:.3f}")
    
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "kappa": float(kappa),
        "per_class": {label: {k: float(v) for k, v in report[label].items()} 
                      for label in target_labels if label in report},
        "confusion_matrix": cm.tolist(),
        "fatal_sensitivity": float(fatal_recall),
    }
    return metrics

# ═══════════════════════════════════════════════════════════════════════
# 3. RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: Random Forest")
print("="*60)
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=class_weights_dict,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
rf.fit(X_train, y_train)
rf_time = time.time() - t0
print(f"  Training time: {rf_time:.1f}s")

rf_val_metrics = evaluate_model("Random Forest", rf, X_val, y_val, "Val")
rf_test_metrics = evaluate_model("Random Forest", rf, X_test, y_test, "Test")

# Feature importance
rf_importance = dict(zip(feature_names, rf.feature_importances_.tolist()))
rf_importance_sorted = dict(sorted(rf_importance.items(), key=lambda x: -x[1]))
print(f"\n  Top 10 features (RF importance):")
for i, (feat, imp) in enumerate(list(rf_importance_sorted.items())[:10]):
    print(f"    {i+1}. {feat}: {imp:.4f}")

# Save model
with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
    pickle.dump(rf, f)

# ═══════════════════════════════════════════════════════════════════════
# 4. XGBOOST
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: XGBoost")
print("="*60)
t0 = time.time()

# Convert class weights to sample weights for XGBoost
sample_weights = np.array([class_weights_dict[int(y)] for y in y_train])

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=30,
)
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
xgb_time = time.time() - t0
print(f"  Training time: {xgb_time:.1f}s")
print(f"  Best iteration: {xgb_model.best_iteration}")

xgb_val_metrics = evaluate_model("XGBoost", xgb_model, X_val, y_val, "Val")
xgb_test_metrics = evaluate_model("XGBoost", xgb_model, X_test, y_test, "Test")

# Feature importance
xgb_importance = dict(zip(feature_names, xgb_model.feature_importances_.tolist()))
xgb_importance_sorted = dict(sorted(xgb_importance.items(), key=lambda x: -x[1]))
print(f"\n  Top 10 features (XGBoost importance):")
for i, (feat, imp) in enumerate(list(xgb_importance_sorted.items())[:10]):
    print(f"    {i+1}. {feat}: {imp:.4f}")

# Save model
with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)

# ═══════════════════════════════════════════════════════════════════════
# 5. LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: LightGBM")
print("="*60)
t0 = time.time()

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight=class_weights_dict,
    objective="multiclass",
    num_class=5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
)
lgb_time = time.time() - t0
print(f"  Training time: {lgb_time:.1f}s")
print(f"  Best iteration: {lgb_model.best_iteration_}")

lgb_val_metrics = evaluate_model("LightGBM", lgb_model, X_val, y_val, "Val")
lgb_test_metrics = evaluate_model("LightGBM", lgb_model, X_test, y_test, "Test")

# Feature importance
lgb_importance = dict(zip(feature_names, lgb_model.feature_importances_.tolist()))
lgb_importance_sorted = dict(sorted(lgb_importance.items(), key=lambda x: -x[1]))
print(f"\n  Top 10 features (LightGBM importance):")
for i, (feat, imp) in enumerate(list(lgb_importance_sorted.items())[:10]):
    print(f"    {i+1}. {feat}: {imp:.4f}")

# Save model
with open(os.path.join(MODELS_DIR, "lightgbm.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)

# ═══════════════════════════════════════════════════════════════════════
# 6. SAVE COMPARATIVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPARATIVE SUMMARY")
print("="*60)

all_results = {
    "random_forest": {
        "val": rf_val_metrics, "test": rf_test_metrics,
        "training_time_sec": rf_time,
        "feature_importance": rf_importance,
    },
    "xgboost": {
        "val": xgb_val_metrics, "test": xgb_test_metrics,
        "training_time_sec": xgb_time,
        "feature_importance": xgb_importance,
        "best_iteration": int(xgb_model.best_iteration),
    },
    "lightgbm": {
        "val": lgb_val_metrics, "test": lgb_test_metrics,
        "training_time_sec": lgb_time,
        "feature_importance": lgb_importance,
        "best_iteration": int(lgb_model.best_iteration_),
    },
}

with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'Model':<15s} {'Acc':>7s} {'Bal Acc':>8s} {'F1 Mac':>8s} {'K_Recall':>9s} {'Time':>7s}")
print("-" * 60)
for name, key in [("Random Forest", "random_forest"), ("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
    m = all_results[key]["test"]
    t = all_results[key]["training_time_sec"]
    print(f"{name:<15s} {m['accuracy']:7.4f} {m['balanced_accuracy']:8.4f} "
          f"{m['f1_macro']:8.4f} {m['fatal_sensitivity']:9.3f} {t:6.1f}s")

print(f"\nResults saved to: {RESULTS_DIR}/baseline_results.json")
print("Models saved to: results/models/")
print("\nDone!")
