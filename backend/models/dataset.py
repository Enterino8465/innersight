"""Dataset utilities: feature tensor construction, standardization, DataLoader assembly."""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow running as __main__ from repo root or backend/
# _backend adds "from config import ..." and "from features.features import ..."
# _pkg_root adds "from innersight.backend.* import ..." used inside pipeline.py
_backend  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_pkg_root = os.path.abspath(os.path.join(_backend, '..', '..'))
for _p in (_backend, _pkg_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Standardizer:
    """Per-feature zero-mean / unit-variance scaler backed by torch tensors.

    Must be fitted on training data only; use :meth:`transform` on val/test.

    Attributes:
        mean: Feature means of shape ``(num_features,)``, set after :meth:`fit`.
        std:  Feature standard deviations (+ eps) of the same shape.
    """

    _EPS: float = 1e-8

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def fit(self, X: torch.Tensor) -> None:
        """Compute and store per-feature mean and std from *X*.

        Args:
            X: 2-D float tensor of shape ``(N, num_features)``.
        """
        self.mean = X.mean(dim=0)
        self.std  = X.std(dim=0) + self._EPS

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Standardize *X* using the stored statistics.

        Args:
            X: 2-D float tensor of shape ``(N, num_features)``.

        Returns:
            Standardized tensor of the same shape.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted. Call fit() first.")
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit on *X* and return the standardized result.

        Args:
            X: 2-D float tensor of shape ``(N, num_features)``.

        Returns:
            Standardized tensor of the same shape.
        """
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist mean and std to *path* using :func:`torch.save`.

        Args:
            path: Destination file path (e.g. ``model_dir/standardizer.pt``).
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted. Nothing to save.")
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str) -> "Standardizer":
        """Load a previously saved :class:`Standardizer` from *path*.

        Args:
            path: Path previously written by :meth:`save`.

        Returns:
            A fitted :class:`Standardizer` instance.
        """
        checkpoint = torch.load(path, weights_only=True)
        instance = cls()
        instance.mean = checkpoint["mean"]
        instance.std  = checkpoint["std"]
        return instance


# ---------------------------------------------------------------------------

def build_features_tensor(
    split_logs: dict,
    labels: set,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Convert one split's raw logs into feature and label tensors.

    Calls :func:`features.features.build_user_day_features` to perform
    feature engineering, then converts the resulting DataFrame to tensors.

    Args:
        split_logs: Logs dict for a single split, keyed by source name
            (``'logon'``, ``'device'``, ``'file'``, ``'email'``, ``'http'``).
        labels: Set of ``(user, date)`` malicious tuples from the full dataset.

    Returns:
        ``(X, y, user_ids)`` where *X* is float32 of shape ``(N, num_features)``,
        *y* is float32 of shape ``(N, 1)``, and *user_ids* is a list of N
        user ID strings in the same row order as *X* — required by
        :class:`~innersight.backend.models.embeddings.EmbeddingManager` for
        alignment with Node2Vec embeddings.
    """
    from features.features import build_user_day_features
    from config import FEATURE_COLS

    df = build_user_day_features(split_logs, labels)

    # Zero-fill any feature columns absent from df (e.g. when only some log
    # types are present for a given split or date range).
    n_rows = len(df)
    X = torch.zeros(n_rows, len(FEATURE_COLS), dtype=torch.float32)
    for j, col in enumerate(FEATURE_COLS):
        if col in df.columns:
            X[:, j] = torch.tensor(df[col].values.astype("float32"))

    y = torch.tensor(df["is_malicious"].values, dtype=torch.float32).unsqueeze(1)

    # Preserve per-row user IDs for embedding alignment downstream.
    user_ids: list = df["user"].tolist() if "user" in df.columns else []

    return X, y, user_ids


def build_dataloaders(
    data: dict,
    batch_size: int = 64,
    embedding_manager=None,
) -> dict:
    """Build train/val/test :class:`~torch.utils.data.DataLoader` objects.

    Fits a :class:`Standardizer` on the training split and applies it to all
    splits. When an active :class:`EmbeddingManager` is supplied, Node2Vec
    embeddings are concatenated after standardisation so the combined tensor
    lands in the :class:`TensorDataset` and flows into every training batch.

    Args:
        data: Output of :func:`data.pipeline.load_data`, containing
            ``data['splits']`` and ``data['labels']``.
        batch_size: Mini-batch size for the training loader (val/test use 256).
        embedding_manager: Optional
            :class:`~innersight.backend.models.embeddings.EmbeddingManager`.
            When provided and ``embedding_manager.available`` is True, each
            split's feature matrix is extended with aligned user embeddings
            (concatenated after standardisation, so embeddings are NOT scaled).

    Returns:
        Dict with keys ``"train_loader"``, ``"val_loader"``, ``"test_loader"``,
        and ``"standardizer"``.
    """
    labels = data["labels"]
    splits = data["splits"]

    # Build feature tensors for every split (X, y, user_ids).
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor, list]] = {}
    for split_name, split_logs in splits.items():
        tensors[split_name] = build_features_tensor(split_logs, labels)

    X_train, y_train, train_user_ids = tensors["train"]
    standardizer = Standardizer()
    X_train_std = standardizer.fit_transform(X_train)

    def _loader(X: torch.Tensor, y: torch.Tensor, bs: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(X, y),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )

    loaders: dict = {"standardizer": standardizer}
    for split_name, (X, y, user_ids) in tensors.items():
        X_std = X_train_std if split_name == "train" else standardizer.transform(X)

        # Optionally append Node2Vec embeddings (not standardised — they're
        # already meaningful representations from the random-walk training).
        if embedding_manager is not None and getattr(embedding_manager, 'available', False):
            X_final = embedding_manager.get_combined_features(X_std, user_ids)
        else:
            X_final = X_std

        if split_name == "train":
            loaders["train_loader"] = _loader(X_final, y, batch_size, shuffle=True)
        elif split_name == "val":
            loaders["val_loader"] = _loader(X_final, y, 256, shuffle=False)
        elif split_name == "test":
            loaders["test_loader"] = _loader(X_final, y, 256, shuffle=False)

    return loaders


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("dataset.py smoke test")
    print("=" * 60)

    # ── Try real data first ────────────────────────────────────────────
    try:
        from data.pipeline import load_data

        print("Loading real CERT data …")
        data = load_data()
        loaders = build_dataloaders(data)

        for name in ("train_loader", "val_loader", "test_loader"):
            loader = loaders[name]
            # Peek at one batch to get shapes
            X_b, y_b = next(iter(loader))
            n_total = len(loader.dataset)  # type: ignore[arg-type]
            n_pos   = int(sum(y_item for _, y_item in loader.dataset))  # type: ignore[union-attr]
            print(
                f"  {name:<14}: {n_total:>6} samples | "
                f"X={tuple(X_b.shape)}  y={tuple(y_b.shape)} | "
                f"pos={n_pos} ({n_pos / n_total * 100:.3f}%)"
            )

        std = loaders["standardizer"]
        print(f"\nStandardizer mean[:4]: {std.mean[:4].tolist()}")  # type: ignore[index]
        print(f"Standardizer std[:4]:  {std.std[:4].tolist()}")    # type: ignore[index]

    except FileNotFoundError as exc:
        print(f"Real data not available ({exc}).")
        print("Running Standardizer-only test with synthetic tensors.\n")

        import numpy as np

        rng = np.random.default_rng(42)
        NUM_FEATURES = 18

        def _synth_tensors(n: int, pos_frac: float = 0.004):
            X_np = rng.exponential(scale=3.0, size=(n, NUM_FEATURES)).astype("float32")
            y_np = np.zeros(n, dtype="float32")
            n_pos = max(1, int(n * pos_frac))
            y_np[rng.choice(n, n_pos, replace=False)] = 1.0
            return torch.from_numpy(X_np), torch.from_numpy(y_np).unsqueeze(1)

        X_tr, y_tr = _synth_tensors(5000)
        X_va, y_va = _synth_tensors(1500)
        X_te, y_te = _synth_tensors(1500)

        scaler = Standardizer()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        for name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            n_pos = int(y.sum().item())
            print(
                f"  {name:<6}: X={tuple(X.shape)}  y={tuple(y.shape)}  "
                f"pos={n_pos} ({n_pos / len(y) * 100:.3f}%)"
            )

        print(f"\nStandardizer mean[:4]: {scaler.mean[:4].tolist()}")  # type: ignore[index]
        print(f"Standardizer std[:4]:  {scaler.std[:4].tolist()}")    # type: ignore[index]

        # Round-trip save / load
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "standardizer.pt")
            scaler.save(path)
            loaded = Standardizer.load(path)
            diff = float((loaded.mean - scaler.mean).abs().max())  # type: ignore[operator]
            print(f"\nSave/load round-trip max diff: {diff:.2e}  "
                  f"{'PASS' if diff == 0.0 else 'FAIL'}")

    # ── Verify unfitted guard ──────────────────────────────────────────
    try:
        Standardizer().transform(torch.randn(4, 18))
    except RuntimeError as exc:
        print(f"\nUnfitted guard (expected): {exc}")
