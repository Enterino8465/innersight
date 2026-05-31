import sys
import os
import copy

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from innersight.backend.config import FEATURE_COLS

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading data")
print("=" * 60)

try:
    from innersight.backend.data.pipeline import load_data
    from innersight.backend.features.features import build_features_for_split
    data    = load_data()
    splits  = build_features_for_split(data)
    train_df, val_df, test_df = splits['train'], splits['val'], splits['test']
    print("Loaded real CERT r4.2 dataset.")
except FileNotFoundError as e:
    print(f"CERT dataset not found — using synthetic data.")
    print(f"(Set INNERSIGHT_DATA_DIR to use real data.)\n")
    import pandas as pd
    rng = np.random.default_rng(0)

    def _synth(n, pos_frac):
        n_pos = max(2, int(n * pos_frac))
        X = rng.exponential(scale=3.0, size=(n, len(FEATURE_COLS))).astype(np.float32)
        # Make positives visibly different so the model can learn something
        pos_idx = rng.choice(n, n_pos, replace=False)
        X[pos_idx] *= 3.0
        y = np.zeros(n, dtype=int)
        y[pos_idx] = 1
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df['is_malicious'] = y
        return df

    train_df = _synth(5000, 0.004)
    val_df   = _synth(1500, 0.004)
    test_df  = _synth(1500, 0.004)

for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    n_pos = int(df['is_malicious'].sum())
    print(f"  {name:>6}: {len(df):>6} rows  pos={n_pos} ({n_pos/len(df)*100:.3f}%)")

# ── 2. Standardize (train stats only) ────────────────────────────────────────
X_tr = torch.tensor(train_df[FEATURE_COLS].values, dtype=torch.float32)
y_tr = torch.tensor(train_df['is_malicious'].values, dtype=torch.float32).unsqueeze(1)

X_va = torch.tensor(val_df[FEATURE_COLS].values,   dtype=torch.float32)
y_va = torch.tensor(val_df['is_malicious'].values,  dtype=torch.float32).unsqueeze(1)

X_te = torch.tensor(test_df[FEATURE_COLS].values,  dtype=torch.float32)
y_te = torch.tensor(test_df['is_malicious'].values, dtype=torch.float32).unsqueeze(1)

mean = X_tr.mean(dim=0)
std  = X_tr.std(dim=0).clamp(min=1e-8)

X_tr = (X_tr - mean) / std
X_va = (X_va - mean) / std
X_te = (X_te - mean) / std

# ── 3. DataLoaders ────────────────────────────────────────────────────────────
train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64,  shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

# ── 4. Model / loss / optimizer ───────────────────────────────────────────────
class InsiderThreatMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model     = InsiderThreatMLP([18, 64, 32, 1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Helper: compute P / R / F1 from a loader ─────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu()
            all_preds.append(preds)
            all_labels.append(y_b)
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    p  = (tp / (tp + fp + 1e-8)).item()
    r  = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * p * r / (p + r + 1e-8))
    return p, r, f1

# ── 5. Training loop with early stopping on val F1 ───────────────────────────
print("\n" + "=" * 60)
print("Training")
print("=" * 60)

MAX_EPOCHS = 50
PATIENCE   = 5

best_f1    = -1.0
best_state = None
no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    p, r, f1   = evaluate(val_loader)

    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val P: {p:.2f} R: {r:.2f} F1: {f1:.2f}"
          + (" *" if f1 > best_f1 else ""))

    if f1 > best_f1:
        best_f1    = f1
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val F1 improvement for {PATIENCE} epochs).")
            break

# ── 6. Test evaluation with best model ───────────────────────────────────────
model.load_state_dict(best_state)
tp_p, tp_r, tp_f1 = evaluate(test_loader)

print("\n" + "=" * 60)
print(f"Best val F1: {best_f1:.4f}")
print(f"Test  — Precision: {tp_p:.4f}  Recall: {tp_r:.4f}  F1: {tp_f1:.4f}")
print("=" * 60)

# ── 7. Comparison ─────────────────────────────────────────────────────────────
print()
print("NumPy pipeline  : trainer.py (150) + network.py (49) + loss.py (14)")
print("                  + backprop.py (76) + optimizer.py (32) = 321 lines")
print(f"PyTorch version : ~{sum(1 for _ in open(__file__))} lines total")
print("Same architecture, same data, same metrics.")
