"""EmbeddingManager: aligns Node2Vec user embeddings with the MLP feature pipeline.

The graph and the MLP feature pipeline share the same user ID strings but
index users independently. EmbeddingManager bridges the two by looking up each
user_id in the embedding table and returning the right row — or a zero vector
when a user has no graph embedding (e.g. first-day new hire with no history).
"""

from __future__ import annotations

import logging
import os
import sys

import torch

_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND  = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PKG_ROOT = os.path.abspath(os.path.join(_BACKEND, '..', '..'))
for _p in (_PKG_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Loads Node2Vec embeddings and aligns them with the MLP feature matrix.

    The alignment step is critical: each row of the feature matrix corresponds
    to one (user, day) pair. Multiple rows can share the same user, and the
    same user embedding is replicated for every such row. Users absent from
    the graph (no HTTP/logon activity in the training window) get zero vectors.

    Parameters
    ----------
    embeddings_path:
        Path to a ``.pt`` file saved by ``node2vec_trainer.save_embeddings()``.
        Pass ``None`` (or a path that does not exist) to create an inactive
        manager that passes feature tensors through unchanged.
    """

    def __init__(self, embeddings_path: str | None = None) -> None:
        self.available: bool = False
        self.embeddings: torch.Tensor | None = None
        self.user_to_idx: dict | None = None
        self.embedding_dim: int | None = None

        if embeddings_path and os.path.exists(embeddings_path):
            checkpoint = torch.load(embeddings_path, weights_only=False)
            self.embeddings  = checkpoint['embeddings'].float()   # (num_users, emb_dim)
            self.user_to_idx = checkpoint['user_to_idx']
            self.embedding_dim = int(self.embeddings.shape[1])
            self.available = True
            logger.info(
                'EmbeddingManager: loaded %d users × %d dims from %s',
                self.embeddings.shape[0], self.embedding_dim, embeddings_path,
            )
        elif embeddings_path:
            logger.warning('EmbeddingManager: path not found (%s) — disabled', embeddings_path)

    # ── Core methods ──────────────────────────────────────────────────────────

    def align_embeddings(self, user_ids: list) -> torch.Tensor:
        """Return an embedding row for every user_id in *user_ids*.

        Users not present in the training graph receive a zero vector.
        The implementation is fully vectorised: the Python list comprehension
        builds an index array once, then PyTorch handles the gather.

        Parameters
        ----------
        user_ids:
            Ordered list of user ID strings, one per row of the feature matrix.
            Length N; duplicates are expected (multiple days per user).

        Returns
        -------
        torch.Tensor
            Shape ``(N, embedding_dim)``, float32.
        """
        if not self.available:
            raise RuntimeError('align_embeddings called on an inactive EmbeddingManager.')

        n_emb = int(self.embeddings.shape[0])  # type: ignore[union-attr]

        # Map each user_id to its row index; unknown users → sentinel n_emb
        raw_idx = torch.tensor(
            [self.user_to_idx.get(uid, n_emb) for uid in user_ids],  # type: ignore[union-attr]
            dtype=torch.long,
        )

        # Append a zero row so that out-of-vocab users get zeros via indexing
        padded = torch.cat(
            [self.embeddings, torch.zeros(1, self.embedding_dim, dtype=torch.float32)],  # type: ignore[arg-type]
            dim=0,
        )
        return padded[raw_idx]  # (N, embedding_dim)

    def get_combined_features(
        self,
        X_flat: torch.Tensor,
        user_ids: list,
    ) -> torch.Tensor:
        """Concatenate flat MLP features with aligned Node2Vec embeddings.

        Parameters
        ----------
        X_flat:
            Feature matrix, shape ``(N, num_flat_features)``.
        user_ids:
            List of N user ID strings in the same row order as *X_flat*.

        Returns
        -------
        torch.Tensor
            ``(N, num_flat_features + embedding_dim)`` if embeddings are
            available, otherwise *X_flat* unchanged.
        """
        if not self.available:
            return X_flat
        aligned = self.align_embeddings(user_ids)
        return torch.cat([X_flat, aligned], dim=1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist embeddings, index mapping, and metadata to *path*."""
        if not self.available:
            raise RuntimeError('Nothing to save: EmbeddingManager has no embeddings.')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                'embeddings':    self.embeddings,
                'user_to_idx':   self.user_to_idx,
                'embedding_dim': self.embedding_dim,
            },
            path,
        )
        logger.info('EmbeddingManager saved → %s', path)

    @classmethod
    def load(cls, path: str) -> 'EmbeddingManager':
        """Load and return an EmbeddingManager from *path*."""
        return cls(embeddings_path=path)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    ap = argparse.ArgumentParser(description='Test EmbeddingManager on real CERT features.')
    ap.add_argument('--data-dir',  default=os.environ.get('INNERSIGHT_DATA_DIR', ''))
    ap.add_argument('--model-dir', default=os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'))
    args = ap.parse_args()
    if not args.data_dir:
        ap.error('--data-dir or INNERSIGHT_DATA_DIR required')

    os.environ['INNERSIGHT_DATA_DIR']  = args.data_dir
    os.environ['INNERSIGHT_MODEL_DIR'] = args.model_dir

    W = 70
    print('=' * W)
    print('EmbeddingManager smoke test')
    print('=' * W)

    # ── 1. Load embeddings ────────────────────────────────────────────────────
    emb_path = os.path.join(args.model_dir, 'node2vec_embeddings.pt')
    mgr = EmbeddingManager(emb_path)
    print(f'\nEmbeddingManager')
    print(f'  available     : {mgr.available}')
    print(f'  embedding_dim : {mgr.embedding_dim}')
    if mgr.available:
        print(f'  users in graph: {len(mgr.user_to_idx):,}')  # type: ignore[arg-type]

    # ── 2. Build real training features ──────────────────────────────────────
    from innersight.backend.data.pipeline import load_data
    from innersight.backend.models.dataset import build_features_tensor

    print(f'\nLoading data from: {args.data_dir}')
    data = load_data(args.data_dir)
    train_logs = data['splits']['train']
    labels     = data['labels']

    X_flat, y, user_ids = build_features_tensor(train_logs, labels)

    print(f'\nFlat features')
    print(f'  shape    : {tuple(X_flat.shape)}')
    print(f'  rows     : {len(user_ids):,}  ({len(set(user_ids)):,} unique users)')

    # ── 3. Combined features ──────────────────────────────────────────────────
    X_combined = mgr.get_combined_features(X_flat, user_ids)
    print(f'\nCombined features')
    print(f'  shape    : {tuple(X_combined.shape)}')

    # ── 4. Verify correctness ─────────────────────────────────────────────────
    print('\nChecks')

    nan_count = int(torch.isnan(X_combined).sum().item())
    print(f'  NaN count   : {nan_count}  →  {"PASS" if nan_count == 0 else "FAIL"}')

    flat_cols  = X_flat.shape[1]
    max_diff   = float((X_combined[:, :flat_cols] - X_flat).abs().max().item())
    print(f'  First {flat_cols} cols match original: max_diff={max_diff:.2e}  →  {"PASS" if max_diff == 0.0 else "FAIL"}')

    if mgr.available:
        found = sum(1 for uid in set(user_ids) if uid in mgr.user_to_idx)  # type: ignore[operator]
        print(f'  Embedding coverage: {found}/{len(set(user_ids))} unique users have embeddings')

        # Expected combined width
        expected_width = flat_cols + mgr.embedding_dim
        actual_width   = X_combined.shape[1]
        print(f'  Width check: {flat_cols} + {mgr.embedding_dim} = {expected_width}  '
              f'(got {actual_width})  →  {"PASS" if expected_width == actual_width else "FAIL"}')

    print('=' * W)
