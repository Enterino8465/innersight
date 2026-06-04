"""Dataset utilities: feature tensor construction, standardization, DataLoader assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from innersight.config import DEFAULT_WINDOW_CONFIG
from innersight.schema import FEATURE_NAMES

if TYPE_CHECKING:
    # Imported under TYPE_CHECKING only (annotation use); avoids any import
    # cycle between models and the data package at runtime.
    from innersight.data.answers import InsiderRecord


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
        :class:`~innersight.models.embeddings.EmbeddingManager` for
        alignment with Node2Vec embeddings.
    """
    from innersight.features.features import build_user_day_features
    from innersight.config import FEATURE_COLS

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
            :class:`~innersight.models.embeddings.EmbeddingManager`.
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


# ===========================================================================
# Phase 2 — windowed deviation dataset for Conv1d temporal models
# ===========================================================================

class DeviationWindowDataset(torch.utils.data.Dataset):
    """Sliding windows of per-user z-scored deviations, labelled by attack overlap.

    Each user's daily deviation history is sliced into fixed-length windows. A
    window is labelled by how much it overlaps that user's known attack period:

        * overlap ratio ``>= overlap_threshold`` → positive (1)
        * overlap ratio ``== 0``                 → negative (0)
        * anything in between                    → EXCLUDED (ambiguous, dropped)

    Windows are stored channels-first, shape ``(num_features, window_size)``, so
    they feed straight into a ``Conv1d`` (features = channels, days = length).

    Args:
        deviations_df: Z-scored deviations with columns ``user``, ``date`` and
            the feature columns (raw values are the per-day deviations).
        attack_windows: Mapping ``user_id`` → ``InsiderRecord`` (insiders only).
        window_size: Number of days per window.
        stride: Days to advance between consecutive windows.
        overlap_threshold: Minimum attack-overlap fraction for a positive label.
        feature_columns: Feature columns to use; defaults to the 18 FEATURE_NAMES.
    """

    def __init__(
        self,
        deviations_df: pd.DataFrame,
        attack_windows: dict[str, InsiderRecord],
        window_size: int = 28,
        stride: int = 7,
        overlap_threshold: float = 0.5,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.overlap_threshold = float(overlap_threshold)
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(FEATURE_NAMES)

        # Each entry: (array shape (num_features, window_size), label, metadata).
        self.windows: list[tuple[np.ndarray, int, dict]] = []

        for user_id, group in deviations_df.groupby("user", sort=False):
            ordered = group.sort_values("date")
            dates = ordered["date"].to_numpy()                                   # datetime64, (n_days,)
            feats = ordered[self.feature_columns].to_numpy(dtype=np.float32)     # shape: (n_days, num_features)
            n_days = feats.shape[0]

            record = attack_windows.get(user_id)
            attack_start = record.attack_start if record is not None else None
            attack_end = record.attack_end if record is not None else None
            scenario = record.scenario if record is not None else 0

            start = 0
            while start + self.window_size <= n_days:
                end = start + self.window_size
                window_dates = dates[start:end]
                overlap = self._compute_overlap(window_dates, attack_start, attack_end)

                if overlap >= self.overlap_threshold:
                    label = 1
                elif overlap == 0.0:
                    label = 0
                else:
                    start += self.stride  # ambiguous partial overlap → exclude
                    continue

                # Channels-first for Conv1d: shape: (num_features, window_size).
                arr = feats[start:end].T.copy()
                meta = {
                    "user_id": user_id,
                    "window_start": pd.Timestamp(window_dates[0]),
                    "window_end": pd.Timestamp(window_dates[-1]),
                    "overlap_ratio": float(overlap),
                    "scenario": int(scenario) if label == 1 else 0,
                }
                self.windows.append((arr, label, meta))
                start += self.stride

    @staticmethod
    def _compute_overlap(window_dates, attack_start, attack_end) -> float:
        """Fraction of ``window_dates`` falling within ``[attack_start, attack_end]``."""
        if attack_start is None or attack_end is None:
            return 0.0
        start = pd.Timestamp(attack_start).normalize()
        end = pd.Timestamp(attack_end).normalize()
        days = pd.to_datetime(window_dates).normalize()
        in_attack = (days >= start) & (days <= end)
        return float(np.count_nonzero(in_attack)) / float(len(window_dates))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        arr, label, meta = self.windows[idx]
        x = torch.from_numpy(arr)                        # shape: (num_features, window_size)
        y = torch.tensor([label], dtype=torch.float32)   # shape: (1,)
        return x, y, meta


def _window_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Collate windowed samples, keeping metadata dicts as a plain list.

    The default collate would try to batch the metadata dicts (which hold
    timestamps and strings) and fail, so windows/labels are stacked while metas
    pass through untouched.
    """
    windows = torch.stack([item[0] for item in batch])   # shape: (B, num_features, window_size)
    labels = torch.stack([item[1] for item in batch])    # shape: (B, 1)
    metas = [item[2] for item in batch]
    return windows, labels, metas


def build_window_dataloaders(
    deviations_df: pd.DataFrame,
    attack_windows: dict[str, InsiderRecord],
    window_size: int | None = None,
    stride: int | None = None,
    overlap_threshold: float | None = None,
    feature_columns: list[str] | None = None,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Build train/val window DataLoaders with user-level (not window-level) split.

    Users — not individual windows — are partitioned into train/val so that no
    user's windows leak across the split. Insiders and benign users are split
    separately with the same ``val_fraction``, keeping the insider proportion of
    each split roughly equal (stratified by insider status).

    Window hyper-parameters default to ``DEFAULT_WINDOW_CONFIG`` when not given.

    Args:
        deviations_df: Z-scored deviations (see :class:`DeviationWindowDataset`).
        attack_windows: Mapping ``user_id`` → ``InsiderRecord``.
        window_size: Days per window (default: config ``window_size``).
        stride: Days between windows (default: config ``window_stride``).
        overlap_threshold: Positive-label overlap fraction
            (default: config ``overlap_positive_threshold``).
        feature_columns: Feature columns to use (default: FEATURE_NAMES).
        batch_size: Mini-batch size for both loaders.
        val_fraction: Fraction of users (per stratum) held out for validation.
        seed: Seed for the user shuffle.

    Returns:
        Dict with ``train_loader``, ``val_loader``, ``train_dataset``,
        ``val_dataset``.
    """
    window_size = DEFAULT_WINDOW_CONFIG["window_size"] if window_size is None else window_size
    stride = DEFAULT_WINDOW_CONFIG["window_stride"] if stride is None else stride
    if overlap_threshold is None:
        overlap_threshold = DEFAULT_WINDOW_CONFIG["overlap_positive_threshold"]

    # Unique users, order-preserving.
    users = list(dict.fromkeys(deviations_df["user"].tolist()))
    insider_ids = set(attack_windows.keys())
    insiders = [u for u in users if u in insider_ids]
    benign = [u for u in users if u not in insider_ids]

    rng = np.random.default_rng(seed)

    def _split(group: list[str]) -> tuple[list[str], list[str]]:
        shuffled = [group[i] for i in rng.permutation(len(group))]
        n_val = int(round(len(shuffled) * val_fraction))
        return shuffled[n_val:], shuffled[:n_val]  # (train, val)

    train_ins, val_ins = _split(insiders)
    train_ben, val_ben = _split(benign)
    train_users = set(train_ins) | set(train_ben)
    val_users = set(val_ins) | set(val_ben)

    train_df = deviations_df[deviations_df["user"].isin(train_users)]
    val_df = deviations_df[deviations_df["user"].isin(val_users)]

    common = dict(
        window_size=window_size,
        stride=stride,
        overlap_threshold=overlap_threshold,
        feature_columns=feature_columns,
    )
    train_dataset = DeviationWindowDataset(train_df, attack_windows, **common)
    val_dataset = DeviationWindowDataset(val_df, attack_windows, **common)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_window_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_window_collate, drop_last=False,
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("dataset.py smoke test")
    print("=" * 60)

    # ── Try real data first ────────────────────────────────────────────
    try:
        from innersight.data.pipeline import load_data

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
        import pathlib
        import tempfile
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
