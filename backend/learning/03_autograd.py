import sys
import os

# Add the Developer root so innersight.backend.* imports resolve inside backprop.py
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
from innersight.backend.b3_network.network import Network
from innersight.backend.b5_backprop.backprop import backward
from innersight.backend.b4_loss.loss import compute_loss

# ── Setup ─────────────────────────────────────────────────────────────────────
np.random.seed(0)
net = Network([18, 8, 1])

X = np.random.randn(4, 18).astype(np.float32)
y = np.array([[1], [0], [1], [0]], dtype=np.float32)

# ── Manual backprop ───────────────────────────────────────────────────────────
pred_np = net.forward(X)
loss_np, grad_out = compute_loss(pred_np, y, pos_weight=1.0)
manual_grads = backward(net, grad_out)

print("=" * 60)
print("MANUAL backprop")
print("=" * 60)
print(f"Loss : {loss_np:.6f}")
print(f"dW[0] first row : {manual_grads[0]['dW'][0, :4]}")

# ── PyTorch autograd ──────────────────────────────────────────────────────────
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

W_t = [torch.tensor(W.astype(np.float32), requires_grad=True) for W in net.weights]
b_t = [torch.tensor(b.astype(np.float32), requires_grad=True) for b in net.biases]

cur = X_t
for W, b in zip(W_t[:-1], b_t[:-1]):
    cur = torch.relu(torch.mm(cur, W) + b)

z_out = torch.mm(cur, W_t[-1]) + b_t[-1]
pred_t = torch.sigmoid(z_out)

eps = 1e-7
pred_c = pred_t.clamp(eps, 1 - eps)
loss_t = -(y_t * torch.log(pred_c) + (1 - y_t) * torch.log(1 - pred_c)).mean()
loss_t.backward()

print("\n" + "=" * 60)
print("PyTorch autograd")
print("=" * 60)
print(f"Loss : {loss_t.item():.6f}")
print(f"dW[0] first row : {W_t[0].grad[0, :4].numpy()}")

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Gradient comparison (manual vs autograd)")
print("=" * 60)
all_close = True
for i in range(len(net.weights)):
    dW_manual = manual_grads[i]['dW']
    dW_auto   = W_t[i].grad.numpy()
    db_manual = manual_grads[i]['db']
    db_auto   = b_t[i].grad.numpy()

    diff_W = float(np.abs(dW_manual - dW_auto).max())
    diff_b = float(np.abs(db_manual - db_auto).max())
    status = "PASS" if diff_W < 1e-5 and diff_b < 1e-5 else "FAIL"
    if status == "FAIL":
        all_close = False
    print(f"Layer {i}  dW max|diff|={diff_W:.2e}  db max|diff|={diff_b:.2e}  [{status}]")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_close:
    print("All gradients match.")
else:
    print("WARNING: some gradients did not match.")
print("Our 70 lines of backprop.py are replaced by one line: loss.backward()")
print("=" * 60)
