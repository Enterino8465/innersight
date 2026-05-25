import sys
import os

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import torch.nn as nn


# ── Model: last layer outputs raw logits (no sigmoid) ─────────────────────────
# BCEWithLogitsLoss fuses sigmoid + BCE in a single numerically stable op.
# Separate sigmoid → log can hit log(0) for extreme activations; fused avoids it.
class InsiderThreatMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
            # No activation on the final layer — raw logits for BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = InsiderThreatMLP([18, 64, 32, 1])

# ── Loss ──────────────────────────────────────────────────────────────────────
# pos_weight=50: each positive sample contributes as much as 50 negatives.
# With ~0.4% anomaly rate (2 positives in 500 samples), this rebalances
# the gradient so the model doesn't just predict "benign" for everything.
pos_weight = torch.tensor([50.0])
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("=" * 60)
print("Loss function")
print("=" * 60)
print(criterion)
print(f"pos_weight: {criterion.pos_weight.item()}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n" + "=" * 60)
print("Optimizer param groups")
print("=" * 60)
for i, pg in enumerate(optimizer.param_groups):
    param_shapes = [tuple(p.shape) for p in pg['params']]
    print(f"Group {i}: lr={pg['lr']}  betas={pg['betas']}  params={param_shapes}")

# ── Imbalanced fake data: 100 samples, 2 positives ───────────────────────────
torch.manual_seed(0)
np.random.seed(0)

n_samples, n_pos = 100, 2
X = torch.randn(n_samples, 18)
y = torch.zeros(n_samples, 1)
pos_idx = torch.randperm(n_samples)[:n_pos]
y[pos_idx] = 1.0

print("\n" + "=" * 60)
print(f"Data: {n_samples} samples, {n_pos} positives ({n_pos/n_samples*100:.1f}%)")
print("=" * 60)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training (50 steps)")
print("=" * 60)

model.train()
for step in range(1, 51):
    optimizer.zero_grad()
    logits = model(X)
    loss   = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step:>3}  loss={loss.item():.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits = model(X)
    probs  = torch.sigmoid(logits)
    preds  = (probs > 0.5).float()

tp = ((preds == 1) & (y == 1)).sum().item()
fp = ((preds == 1) & (y == 0)).sum().item()
fn = ((preds == 0) & (y == 1)).sum().item()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

print("\n" + "=" * 60)
print("Evaluation on training data")
print("=" * 60)
print(f"TP={int(tp)}  FP={int(fp)}  FN={int(fn)}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"Positives flagged: {int(preds.sum())} of {n_samples} samples")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("loss.py (14 lines) + optimizer.py (32 lines) +")
print("backprop.py (72 lines) = 118 lines replaced by 4 lines of PyTorch")
print("=" * 60)
