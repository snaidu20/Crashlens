"""
CrashLens — Step 7b: Evaluate saved FT-Transformer checkpoint
Loads the best checkpoint from the training run and evaluates on val/test.
Saves updated transformer_results.json.
"""
import numpy as np, json, os, pickle, warnings, time
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    accuracy_score, balanced_accuracy_score, cohen_kappa_score)

SPLITS_DIR = "/home/user/workspace/crashlens/data/processed/splits"
PROCESSED_DIR = "/home/user/workspace/crashlens/data/processed"
RESULTS_DIR = "/home/user/workspace/crashlens/results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# ── Config ──────────────────────────────────────────────────────────────
with open(os.path.join(PROCESSED_DIR, "feature_config.json")) as f:
    config = json.load(f)
with open(os.path.join(SPLITS_DIR, "split_config.json")) as f:
    split_config = json.load(f)

num_idx  = config["numerical_feature_indices"]
cat_idx  = config["categorical_feature_indices"]
bin_idx  = config["binary_feature_indices"]
ord_idx  = config["ordinal_feature_indices"]
cat_cards = config["cat_cardinalities"]
n_classes = config["n_classes"]
cont_idx  = num_idx + bin_idx + ord_idx
n_cont, n_cat = len(cont_idx), len(cat_idx)
target_labels = ["O_NoInjury", "C_Possible", "B_NonIncap", "A_Incap", "K_Fatal"]
print(f"n_cont={n_cont}, n_cat={n_cat}, cat_cards={cat_cards}")

# ── Model class (must match 07_train_transformer.py) ────────────────────
class FTTransformer(nn.Module):
    def __init__(self, n_cont, cat_cardinalities, d_model=32, n_heads=4,
                 n_layers=2, d_ff=64, dropout=0.2, n_classes=5):
        super().__init__()
        self.d_model = d_model
        self.n_cont = n_cont
        self.n_cat  = len(cat_cardinalities)
        n_tokens = n_cont + self.n_cat + 1
        self.cont_proj = nn.Linear(n_cont, n_cont * d_model)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model) for card in cat_cardinalities])
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes))

    def forward(self, x_cont, x_cat):
        B = x_cont.size(0)
        cont_all   = self.cont_proj(x_cont).view(B, self.n_cont, self.d_model)
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        cat_all    = torch.cat(cat_tokens, dim=1)
        cls        = self.cls_token.expand(B, -1, -1)
        tokens     = torch.cat([cls, cont_all, cat_all], dim=1) + self.pos_encoding
        tokens     = self.ln(self.transformer(tokens))
        return self.head(tokens[:, 0, :])

# ── Load checkpoint ──────────────────────────────────────────────────────
ckpt_path = os.path.join(MODELS_DIR, "ft_transformer_best.pt")
print(f"\nLoading checkpoint: {ckpt_path}")
model = FTTransformer(n_cont, cat_cards, d_model=32, n_heads=4, n_layers=2,
                      d_ff=64, dropout=0.2, n_classes=n_classes)
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")

# ── Load scaler and val/test data ────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

print("Loading val/test data...")
X_val   = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

X_val_cont  = scaler.transform(X_val[:, cont_idx]).astype(np.float32)
X_val_cat   = X_val[:, cat_idx].astype(np.int64)
X_test_cont = scaler.transform(X_test[:, cont_idx]).astype(np.float32)
X_test_cat  = X_test[:, cat_idx].astype(np.int64)

val_ds  = TensorDataset(torch.from_numpy(X_val_cont),  torch.from_numpy(X_val_cat),  torch.from_numpy(y_val.astype(np.int64)))
test_ds = TensorDataset(torch.from_numpy(X_test_cont), torch.from_numpy(X_test_cat), torch.from_numpy(y_test.astype(np.int64)))
val_loader  = DataLoader(val_ds,  batch_size=4096, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

# ── Evaluate ─────────────────────────────────────────────────────────────
class_weights = torch.FloatTensor([split_config["class_weights_sqrt"][str(i)] for i in range(n_classes)])
criterion = nn.CrossEntropyLoss(weight=class_weights)

def evaluate(loader, name):
    preds, probs_all, labels, v_loss, v_n = [], [], [], 0, 0
    with torch.no_grad():
        for xc, xcat, yb in loader:
            logits = model(xc, xcat)
            v_loss += criterion(logits, yb).item() * yb.size(0)
            v_n    += yb.size(0)
            probs_all.extend(torch.softmax(logits, 1).numpy())
            preds.extend(logits.argmax(1).numpy())
            labels.extend(yb.numpy())
    preds, labels = np.array(preds), np.array(labels)
    probs_all = np.array(probs_all)

    acc     = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1m     = f1_score(labels, preds, average="macro")
    f1w     = f1_score(labels, preds, average="weighted")
    kappa   = cohen_kappa_score(labels, preds)
    report  = classification_report(labels, preds, target_names=target_labels,
                                    output_dict=True, zero_division=0)
    cm      = confusion_matrix(labels, preds)
    k_rec   = report.get("K_Fatal", {}).get("recall", 0)

    print(f"\n{'='*60}")
    print(f"FT-Transformer — {name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}  Balanced: {bal_acc:.4f}  F1-macro: {f1m:.4f}")
    print(f"  F1-weighted: {f1w:.4f}  Kappa: {kappa:.4f}")
    for l in target_labels:
        r = report[l]
        print(f"    {l:20s} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")
    print(f"  ★ Fatal Sensitivity: {k_rec:.3f}")
    print("  Confusion Matrix:")
    hdr = "".join(f"{t[:6]:>9s}" for t in target_labels)
    print(f"  {'':18s}{hdr}")
    for i, row in enumerate(cm):
        print(f"  {target_labels[i]:18s} " + " ".join(f"{v:8d}" for v in row))

    return {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1m), "f1_weighted": float(f1w), "kappa": float(kappa),
        "per_class": {l: {k: float(v) for k, v in report[l].items()} for l in target_labels},
        "confusion_matrix": cm.tolist(), "fatal_sensitivity": float(k_rec),
    }, probs_all

t0 = time.time()
val_metrics,  _           = evaluate(val_loader,  "Validation")
test_metrics, test_probs  = evaluate(test_loader, "Test")
np.save(os.path.join(SPLITS_DIR, "test_probs_transformer.npy"), test_probs)

results = {
    "val": val_metrics, "test": test_metrics,
    "training_time_sec": None,   # partial training (checkpoint eval)
    "best_epoch": None,
    "total_epochs": None,
    "n_params": n_params,
    "architecture": {"d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64,
                     "dropout": 0.2, "n_cont": n_cont, "n_cat": n_cat},
    "note": "Evaluated from best checkpoint saved during training run",
}
with open(os.path.join(RESULTS_DIR, "transformer_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → transformer_results.json")
print(f"Eval time: {time.time()-t0:.1f}s")
print("\nDone!")
