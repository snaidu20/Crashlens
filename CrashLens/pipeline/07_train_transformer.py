"""
CrashLens — Step 7: Train FT-Transformer (CPU-optimized)
Lightweight Feature Tokenizer + Transformer for crash severity prediction.
"""
import numpy as np
import json, os, time, pickle, warnings, gc
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler
from collections import Counter

SPLITS_DIR = "/home/user/workspace/crashlens/data/processed/splits"
PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"
RESULTS_DIR = "/home/user/workspace/crashlens/results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
device = torch.device("cpu")

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD CONFIG & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading config...")
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    config = json.load(f)
with open(os.path.join(SPLITS_DIR, "split_config.json")) as f:
    split_config = json.load(f)

feature_names = config["model_features"]
num_idx = config["numerical_feature_indices"]
cat_idx = config["categorical_feature_indices"]
bin_idx = config["binary_feature_indices"]
ord_idx = config["ordinal_feature_indices"]
cat_cards = config["cat_cardinalities"]
n_classes = config["n_classes"]
target_labels = ["O_NoInjury", "C_Possible", "B_NonIncap", "A_Incap", "K_Fatal"]
cont_idx = num_idx + bin_idx + ord_idx
n_cont = len(cont_idx)
n_cat = len(cat_idx)

# Load and subsample training data
print("Loading training data...")
X_train_full = np.load(os.path.join(SPLITS_DIR, "X_train_smote.npy"))
y_train_full = np.load(os.path.join(SPLITS_DIR, "y_train_smote.npy"))

# Subsample: 10K per class max
rng = np.random.RandomState(42)
train_indices = []
for cls in range(n_classes):
    cls_idx = np.where(y_train_full == cls)[0]
    if len(cls_idx) > 10000:
        train_indices.extend(rng.choice(cls_idx, 10000, replace=False))
    else:
        train_indices.extend(cls_idx)
train_indices = np.array(train_indices)
rng.shuffle(train_indices)
X_train = X_train_full[train_indices]
y_train = y_train_full[train_indices]
del X_train_full, y_train_full
gc.collect()
print(f"  Train (subsampled): {X_train.shape[0]:,} — {dict(Counter(y_train))}")

# Scale continuous features
scaler = StandardScaler()
X_train_cont = scaler.fit_transform(X_train[:, cont_idx]).astype(np.float32)
X_train_cat = X_train[:, cat_idx].astype(np.int64)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Create training tensors
train_ds = TensorDataset(
    torch.from_numpy(X_train_cont),
    torch.from_numpy(X_train_cat),
    torch.from_numpy(y_train.astype(np.int64))
)
del X_train, X_train_cont, X_train_cat
gc.collect()
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=True)

# Load val & test
print("Loading val/test data...")
X_val = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
X_test = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

X_val_cont = scaler.transform(X_val[:, cont_idx]).astype(np.float32)
X_val_cat = X_val[:, cat_idx].astype(np.int64)
X_test_cont = scaler.transform(X_test[:, cont_idx]).astype(np.float32)
X_test_cat = X_test[:, cat_idx].astype(np.int64)
del X_val, X_test
gc.collect()

val_ds = TensorDataset(
    torch.from_numpy(X_val_cont), torch.from_numpy(X_val_cat),
    torch.from_numpy(y_val.astype(np.int64))
)
test_ds = TensorDataset(
    torch.from_numpy(X_test_cont), torch.from_numpy(X_test_cat),
    torch.from_numpy(y_test.astype(np.int64))
)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

# ═══════════════════════════════════════════════════════════════════════
# 2. FT-TRANSFORMER MODEL (Compact)
# ═══════════════════════════════════════════════════════════════════════
class FTTransformer(nn.Module):
    def __init__(self, n_cont, cat_cardinalities, d_model=32, n_heads=4,
                 n_layers=2, d_ff=64, dropout=0.2, n_classes=5):
        super().__init__()
        self.d_model = d_model
        self.n_cont = n_cont
        self.n_cat = len(cat_cardinalities)
        n_tokens = n_cont + self.n_cat + 1

        # Continuous: single linear layer for all at once, then reshape
        self.cont_proj = nn.Linear(n_cont, n_cont * d_model)
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model) for card in cat_cardinalities
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes)
        )

    def forward(self, x_cont, x_cat):
        B = x_cont.size(0)
        # Continuous tokens
        cont_all = self.cont_proj(x_cont).view(B, self.n_cont, self.d_model)
        # Categorical tokens
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        cat_all = torch.cat(cat_tokens, dim=1)
        # CLS + all tokens
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, cont_all, cat_all], dim=1) + self.pos_encoding
        tokens = self.ln(self.transformer(tokens))
        return self.head(tokens[:, 0, :])

# ═══════════════════════════════════════════════════════════════════════
# 3. TRAIN
# ═══════════════════════════════════════════════════════════════════════
print("\nInitializing FT-Transformer...")
model = FTTransformer(n_cont, cat_cards, d_model=32, n_heads=4, n_layers=2, 
                       d_ff=64, dropout=0.2, n_classes=n_classes)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")

class_weights = torch.FloatTensor([split_config["class_weights_sqrt"][str(i)] for i in range(n_classes)])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)

EPOCHS = 6
PATIENCE = 3
best_val_f1 = 0
patience_counter = 0
train_history = []

print(f"\nTraining (up to {EPOCHS} epochs, patience={PATIENCE})...")
print(f"{'Ep':>3s} {'TrLoss':>8s} {'VLoss':>8s} {'VF1':>7s} {'K_Rec':>7s} {'Time':>6s}")
print("-" * 45)
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    
    model.train()
    train_loss = 0
    n_b = 0
    for xc, xcat, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xc, xcat), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        n_b += 1
    train_loss /= n_b

    model.eval()
    preds_all, labels_all = [], []
    v_loss = 0
    v_n = 0
    with torch.no_grad():
        for xc, xcat, yb in val_loader:
            logits = model(xc, xcat)
            v_loss += criterion(logits, yb).item() * yb.size(0)
            v_n += yb.size(0)
            preds_all.extend(logits.argmax(1).numpy())
            labels_all.extend(yb.numpy())
    v_loss /= v_n
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    vf1 = f1_score(labels_all, preds_all, average="macro")
    k_mask = labels_all == 4
    k_rec = (preds_all[k_mask] == 4).mean() if k_mask.sum() > 0 else 0

    ep_time = time.time() - t_ep
    print(f"{epoch:3d} {train_loss:8.4f} {v_loss:8.4f} {vf1:7.4f} {k_rec:7.3f} {ep_time:5.1f}s")
    train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": v_loss,
                          "val_f1_macro": vf1, "val_fatal_recall": k_rec})

    if vf1 > best_val_f1:
        best_val_f1 = vf1
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "ft_transformer_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best F1: {best_val_f1:.4f})")
            break
    scheduler.step()

total_time = time.time() - t_start
print(f"\nTraining time: {total_time:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# 4. EVALUATE BEST MODEL
# ═══════════════════════════════════════════════════════════════════════
print("\nEvaluating best model...")
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "ft_transformer_best.pt"), weights_only=True))
model.eval()

def evaluate(loader, name):
    preds, probs_all, labels = [], [], []
    with torch.no_grad():
        for xc, xcat, yb in loader:
            logits = model(xc, xcat)
            probs_all.extend(torch.softmax(logits, 1).numpy())
            preds.extend(logits.argmax(1).numpy())
            labels.extend(yb.numpy())
    preds, labels = np.array(preds), np.array(labels)
    probs_all = np.array(probs_all)
    
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    f1w = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds)
    report = classification_report(labels, preds, target_names=target_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    k_rec = report.get("K_Fatal", {}).get("recall", 0)
    
    print(f"\n{'='*60}")
    print(f"FT-Transformer — {name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}  Balanced: {bal_acc:.4f}  F1-macro: {f1m:.4f}")
    print(f"  F1-weighted: {f1w:.4f}  Kappa: {kappa:.4f}")
    for l in target_labels:
        r = report[l]
        print(f"    {l:20s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
    print(f"  ★ Fatal Sensitivity: {k_rec:.3f}")
    print(f"  Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"    {target_labels[i]:15s} {row}")
    
    return {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1m), "f1_weighted": float(f1w), "kappa": float(kappa),
        "per_class": {l: {k: float(v) for k, v in report[l].items()} for l in target_labels},
        "confusion_matrix": cm.tolist(), "fatal_sensitivity": float(k_rec),
    }, probs_all

val_metrics, _ = evaluate(val_loader, "Validation")
test_metrics, test_probs = evaluate(test_loader, "Test")
np.save(os.path.join(SPLITS_DIR, "test_probs_transformer.npy"), test_probs)

# ═══════════════════════════════════════════════════════════════════════
# 5. SAVE RESULTS & COMPARE ALL MODELS
# ═══════════════════════════════════════════════════════════════════════
results = {
    "val": val_metrics, "test": test_metrics,
    "training_time_sec": total_time,
    "best_epoch": int(np.argmax([h["val_f1_macro"] for h in train_history]) + 1),
    "total_epochs": len(train_history), "n_params": n_params,
    "architecture": {"d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64, 
                     "dropout": 0.2, "n_cont": n_cont, "n_cat": n_cat},
    "training_history": train_history,
}
with open(os.path.join(RESULTS_DIR, "transformer_results.json"), "w") as f:
    json.dump(results, f, indent=2)

model_cfg = {"n_cont": n_cont, "cat_cardinalities": cat_cards, "d_model": 32,
             "n_heads": 4, "n_layers": 2, "d_ff": 64, "dropout": 0.2,
             "n_classes": n_classes, "cont_idx": cont_idx, "cat_idx": cat_idx}
with open(os.path.join(MODELS_DIR, "ft_transformer_config.json"), "w") as f:
    json.dump(model_cfg, f, indent=2)

# Compare all models
print(f"\n{'='*70}")
print("ALL MODELS — TEST SET COMPARISON")
print(f"{'='*70}")
with open(os.path.join(RESULTS_DIR, "baseline_results.json")) as f:
    baselines = json.load(f)

print(f"\n{'Model':<18s} {'Acc':>7s} {'BalAcc':>7s} {'F1Mac':>7s} {'F1Wgt':>7s} {'K_Rec':>7s} {'Kappa':>7s}")
print("-" * 65)
for name, key in [("Random Forest", "random_forest"), ("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
    m = baselines[key]["test"]
    print(f"{name:<18s} {m['accuracy']:7.4f} {m['balanced_accuracy']:7.4f} "
          f"{m['f1_macro']:7.4f} {m['f1_weighted']:7.4f} {m['fatal_sensitivity']:7.3f} {m['kappa']:7.3f}")
m = test_metrics
print(f"{'FT-Transformer':<18s} {m['accuracy']:7.4f} {m['balanced_accuracy']:7.4f} "
      f"{m['f1_macro']:7.4f} {m['f1_weighted']:7.4f} {m['fatal_sensitivity']:7.3f} {m['kappa']:7.3f}")
print("\nDone!")
