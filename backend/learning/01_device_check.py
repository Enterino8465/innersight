import time
import torch

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

small = torch.rand(3, 3, device=device)
print(f"\n3x3 random tensor on {device}:\n{small}")

a = torch.rand(1000, 1000, device=device)
b = torch.rand(1000, 1000, device=device)

# Warm-up (first call can include JIT / kernel-launch overhead)
_ = a @ b
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()

start = time.time()
c = a @ b
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()
elapsed = time.time() - start

print(f"\n1000x1000 matmul on {device}: {elapsed*1000:.2f} ms")
