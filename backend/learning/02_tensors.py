import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
from b3_network.network import Network

# ── 1. NumPy forward pass ─────────────────────────────────────────────────────
np.random.seed(42)
X_np = np.random.randn(5, 18).astype(np.float32)

network = Network([18, 64, 32, 1])
out_np = network.forward(X_np)

print("=" * 60)
print("NumPy forward pass")
print("=" * 60)
print(f"Input shape : {X_np.shape}")
print(f"Output shape: {out_np.shape}")
print(f"Output values:\n{out_np}")

# ── 2. PyTorch forward pass (same weights) ────────────────────────────────────
X_t = torch.from_numpy(X_np)

# Mirror every layer weight/bias as a torch tensor
weights_t = [torch.from_numpy(W.astype(np.float32)) for W in network.weights]
biases_t  = [torch.from_numpy(b.astype(np.float32)) for b in network.biases]

# Hidden layers: ReLU
current = X_t
for W, b in zip(weights_t[:-1], biases_t[:-1]):
    current = torch.relu(torch.mm(current, W) + b)

# Output layer: sigmoid
z_out = torch.mm(current, weights_t[-1]) + biases_t[-1]
out_t = torch.sigmoid(z_out)

print("\n" + "=" * 60)
print("PyTorch forward pass (identical weights)")
print("=" * 60)
print(f"Input shape : {X_t.shape}")
print(f"Output shape: {out_t.shape}")
print(f"Output values:\n{out_t}")

max_diff = float((out_t - torch.from_numpy(out_np.astype(np.float32))).abs().max())
print(f"\nMax abs difference (numpy vs torch): {max_diff:.2e}")
assert max_diff < 1e-6, f"Outputs diverged: {max_diff}"
print("PASS — outputs match to < 1e-6")

# ── 3. Tensor properties ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Tensor vs NumPy properties")
print("=" * 60)

arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t   = torch.from_numpy(arr)

print(f"np.shape  : {arr.shape}  | tensor.shape  : {t.shape}")
print(f"np.dtype  : {arr.dtype}  | tensor.dtype  : {t.dtype}")
print(f"tensor.device: {t.device}")

# ── 4. Round-trip: numpy → tensor → numpy ─────────────────────────────────────
print("\n" + "=" * 60)
print("Round-trip: numpy → tensor → numpy")
print("=" * 60)
back_to_np = t.numpy()
print(f"Original numpy  :\n{arr}")
print(f"→ tensor        :\n{t}")
print(f"→ back to numpy :\n{back_to_np}")
print(f"Arrays equal: {np.array_equal(arr, back_to_np)}")
print("Note: torch.from_numpy shares memory — no copy made")

# ── 5. Broadcasting ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Broadcasting")
print("=" * 60)
t2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"Original tensor:\n{t2}")
print(f"+ scalar 10    :\n{t2 + 10}")
print(f"+ row [1,2,3]  :\n{t2 + torch.tensor([1.0, 2.0, 3.0])}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Key takeaway: same math, same shapes, but tensors track")
print("computation for autograd")
print("=" * 60)
