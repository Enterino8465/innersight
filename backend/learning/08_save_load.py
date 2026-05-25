import sys
import os
import copy

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

_here = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH   = os.path.join(_here, 'model_checkpoint.pt')
PREPROCESSOR_PATH = os.path.join(_here, 'preprocessor.pt')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from innersight.backend.config import FEATURE_COLS

# ── Device ────────────────────────────────────────────────────────────────────
device = (torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))
print(f"Training device: {device}")


# ── Model (same architecture as 07) ──────────────────────────────────────────
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


# ── 1. Quick training on synthetic data ──────────────────────────────────────
print("\n" + "=" * 60)
print("Training (10 epochs on synthetic data)")
print("=" * 60)

rng = np.random.default_rng(0)
n   = 2000
X_np = rng.exponential(scale=3.0, size=(n, len(FEATURE_COLS))).astype(np.float32)
y_np = np.zeros(n, dtype=np.float32)
pos  = rng.choice(n, int(n * 0.004), replace=False)
X_np[pos] *= 3.0
y_np[pos]  = 1.0

X_t = torch.tensor(X_np)
y_t = torch.tensor(y_np).unsqueeze(1)
mean = X_t.mean(0);  std = X_t.std(0).clamp(min=1e-8)
X_t  = (X_t - mean) / std

loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

model     = InsiderThreatMLP([18, 64, 32, 1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 11):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b.to(device)), y_b.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"  Epoch {epoch:02d}/10  loss={total/len(loader):.4f}")

# ── 2. Save model and preprocessor ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving")
print("=" * 60)

torch.save(model.state_dict(), CHECKPOINT_PATH)
torch.save({'mean': mean, 'std': std}, PREPROCESSOR_PATH)

model_kb = os.path.getsize(CHECKPOINT_PATH)   / 1024
prep_kb  = os.path.getsize(PREPROCESSOR_PATH) / 1024
print(f"  model_checkpoint.pt  : {model_kb:.1f} KB")
print(f"  preprocessor.pt      : {prep_kb:.1f} KB")

# ── 3. Load into a fresh model ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading into fresh model")
print("=" * 60)

fresh_model = InsiderThreatMLP([18, 64, 32, 1])
fresh_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))
fresh_model.eval()
print("  fresh_model loaded and set to eval mode")

# ── 4. Prove identity ─────────────────────────────────────────────────────────
# Load fresh model onto the same device so arithmetic is identical.
print("\n" + "=" * 60)
print("Identity check: original vs reloaded model (same device)")
print("=" * 60)

model.eval()
sample = X_t[:32].cpu()

fresh_model.to(device)
fresh_model.eval()
with torch.no_grad():
    out_original = torch.sigmoid(model(sample.to(device))).cpu()
    out_fresh    = torch.sigmoid(fresh_model(sample.to(device))).cpu()

max_diff = (out_original - out_fresh).abs().max().item()
print(f"  Max |diff| between outputs: {max_diff:.1e}")
assert max_diff == 0.0, f"Expected 0.0, got {max_diff}"
print("  PASS — outputs are bit-identical on the same device")

# ── 5. Device portability ─────────────────────────────────────────────────────
# Two CPU-loaded models must agree exactly; vs MPS they may differ by ~1e-8
# due to hardware-level float32 rounding — that's normal and expected.
print("\n" + "=" * 60)
print("Device portability: load with map_location='cpu'")
print("=" * 60)

cpu_model_a = InsiderThreatMLP([18, 64, 32, 1])
cpu_model_a.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))
cpu_model_a.eval()

cpu_model_b = InsiderThreatMLP([18, 64, 32, 1])
cpu_model_b.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))
cpu_model_b.eval()

with torch.no_grad():
    out_cpu_a = torch.sigmoid(cpu_model_a(sample))
    out_cpu_b = torch.sigmoid(cpu_model_b(sample))

max_diff_cpu = (out_cpu_a - out_cpu_b).abs().max().item()
vs_mps_diff  = (out_original - out_cpu_a).abs().max().item()
print(f"  Trained on      : {device}")
print(f"  Loaded onto     : cpu  (map_location='cpu')")
print(f"  CPU vs CPU diff : {max_diff_cpu:.1e}  (bit-identical)")
print(f"  {device} vs CPU diff : {vs_mps_diff:.1e}  (hardware rounding, normal)")
assert max_diff_cpu == 0.0
print("  PASS — map_location='cpu' works regardless of training device")

# ── 6. State dict contents ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("State dict contents")
print("=" * 60)

state = model.state_dict()
total_params = 0
for key, tensor in state.items():
    n_params = tensor.numel()
    total_params += n_params
    print(f"  {key:<30} shape={str(tuple(tensor.shape)):<18} ({n_params:>5} params)")

print(f"\n  Total parameters serialised: {total_params:,}")
print(f"  File size: {model_kb:.1f} KB  "
      f"({model_kb*1024/total_params*8:.1f} bits/param — float32 = 32 bits)")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 1 complete. You now know: tensors, autograd, nn.Module,")
print("loss functions, optimizers, DataLoader, training loops, and")
print("model persistence.")
print()
print("Phase 2: rewrite the production code using everything you")
print("just learned.")
print("=" * 60)
