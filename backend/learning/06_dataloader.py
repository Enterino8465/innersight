import sys
import os

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from innersight.backend.config import FEATURE_COLS

# ── 1. Load real CERT data if available, else fall back to synthetic ───────────
print("=" * 60)
print("Loading data")
print("=" * 60)

try:
    from innersight.backend.b2_data.pipeline import load_data
    from innersight.backend.b2_features.features import build_features_for_split

    data    = load_data()
    splits  = build_features_for_split(data)
    train_df = splits['train']
    val_df   = splits['val']
    test_df  = splits['test']
    print("Loaded real CERT r4.2 dataset.")
    using_real = True

except FileNotFoundError as e:
    print(f"CERT dataset not found ({e}).")
    print("Generating synthetic data with the same schema and class ratio.")
    using_real = False

    import pandas as pd

    rng = np.random.default_rng(42)

    def _synthetic_split(n, pos_frac, name):
        n_pos = max(1, int(n * pos_frac))
        X = rng.exponential(scale=3.0, size=(n, len(FEATURE_COLS))).astype(np.float32)
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, n_pos, replace=False)] = 1
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df['is_malicious'] = y
        print(f"  {name:>6}: {n:>6} rows  {n_pos} positives ({n_pos/n*100:.2f}%)")
        return df

    print()
    train_df = _synthetic_split(5000, 0.004, 'train')
    val_df   = _synthetic_split(1500, 0.004, 'val')
    test_df  = _synthetic_split(1500, 0.004, 'test')

# ── 2. Shape and class distribution ──────────────────────────────────────────
print(f"\nTrain split shape : {train_df.shape}  "
      f"(expected ~N rows × {len(FEATURE_COLS)+1} cols)")

for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    n_pos  = int(df['is_malicious'].sum())
    n_neg  = len(df) - n_pos
    ratio  = n_pos / len(df) * 100
    print(f"  {split_name:>6}: {len(df):>6} rows | "
          f"neg={n_neg}  pos={n_pos}  ({ratio:.3f}% malicious)")

# ── 3. Split into X / y ────────────────────────────────────────────────────────
X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df['is_malicious'].values.astype(np.float32)

X_val   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val   = val_df['is_malicious'].values.astype(np.float32)

X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df['is_malicious'].values.astype(np.float32)

# ── 4. Convert to tensors ─────────────────────────────────────────────────────
X_tensor = torch.tensor(X_train)
y_tensor = torch.tensor(y_train).unsqueeze(1)   # (N,) → (N, 1)

print(f"\nX_tensor shape : {X_tensor.shape}   dtype={X_tensor.dtype}")
print(f"y_tensor shape : {y_tensor.shape}   dtype={y_tensor.dtype}")

# ── 5. TensorDataset + DataLoader ─────────────────────────────────────────────
dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

print(f"\nDataLoader: {len(dataset)} samples  "
      f"batch_size=64  → {len(loader)} batches")

# ── 6. First 3 batches ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("First 3 batches")
print("=" * 60)
for batch_idx, (X_b, y_b) in enumerate(loader):
    if batch_idx >= 3:
        break
    n_pos = int(y_b.sum().item())
    print(f"Batch {batch_idx}  X={tuple(X_b.shape)}  y={tuple(y_b.shape)}  "
          f"positives={n_pos}")

# ── 7. All three splits as loaders ────────────────────────────────────────────
def make_loader(X, y, batch_size=64, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

print("\n" + "=" * 60)
print("All three DataLoaders")
print("=" * 60)
print(f"  train_loader : {len(train_loader)} batches  (shuffle=True)")
print(f"  val_loader   : {len(val_loader)} batches")
print(f"  test_loader  : {len(test_loader)} batches")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("trainer.py's manual batching loop is replaced by:")
print("  for X_batch, y_batch in loader:")
print("=" * 60)
