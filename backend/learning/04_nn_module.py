import sys
import os

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import torch.nn as nn
# NOTE: b3_network was deleted in Phase 0 cleanup — this script is archived/non-runnable
# from innersight.backend.b3_network.network import Network


class InsiderThreatMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Build both models ─────────────────────────────────────────────────────────
np.random.seed(7)
layer_sizes = [18, 64, 32, 1]

old_net = Network(layer_sizes)
new_net = InsiderThreatMLP(layer_sizes)

# ── Copy weights: our layout is (in, out); nn.Linear expects (out, in) ────────
linear_layers = [m for m in new_net.net if isinstance(m, nn.Linear)]
with torch.no_grad():
    for torch_layer, W_np, b_np in zip(linear_layers, old_net.weights, old_net.biases):
        torch_layer.weight.copy_(torch.from_numpy(W_np.T.astype(np.float32)))
        torch_layer.bias.copy_(torch.from_numpy(b_np.squeeze().astype(np.float32)))

# ── Forward pass comparison ───────────────────────────────────────────────────
np.random.seed(42)
X_np = np.random.randn(5, 18).astype(np.float32)
X_t  = torch.from_numpy(X_np)

out_np = old_net.forward(X_np)
out_t  = new_net(X_t).detach().numpy()

print("=" * 60)
print("Forward pass comparison")
print("=" * 60)
print(f"{'Sample':<8} {'NumPy':>12} {'PyTorch':>12} {'|diff|':>12}")
print("-" * 48)
for i, (a, b) in enumerate(zip(out_np.flatten(), out_t.flatten())):
    print(f"{i:<8} {a:>12.8f} {b:>12.8f} {abs(a-b):>12.2e}")

max_diff = float(np.abs(out_np - out_t).max())
status = "PASS" if max_diff < 1e-5 else "FAIL"
print(f"\nMax |diff|: {max_diff:.2e}  [{status}]")

# ── Model summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Model architecture")
print("=" * 60)
print(new_net)

total = sum(p.numel() for p in new_net.parameters())
print(f"\nTotal parameters: {total:,}")

print("\nNamed parameters:")
for name, param in new_net.named_parameters():
    print(f"  {name:<30} shape={tuple(param.shape)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Our Network class is now an nn.Module. We get parameter")
print("tracking, device movement, and save/load for free.")
print("=" * 60)
