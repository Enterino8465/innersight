#!/usr/bin/env python
"""Visualize Node2Vec (or MetaPath2Vec) user embeddings in 2-D.

Produces four PNG files in the output directory:
  embeddings_tsne_labels.png      — t-SNE, malicious vs benign
  embeddings_tsne_departments.png — t-SNE, colored by department proxy
  embeddings_umap_labels.png      — UMAP, malicious vs benign
  embeddings_umap_departments.png — UMAP, colored by department proxy

Department proxy: when no LDAP.csv is available (partial CERT dataset) we
derive a pseudo-department from the first letter of the user ID.  The real
CERT LDAP.csv maps user → department; if it exists it is used automatically.

Usage
-----
    python scripts/visualize_embeddings.py \\
        --embeddings checkpoints/node2vec_embeddings.pt \\
        --data-dir   innersight/data/cert_r4.2 \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_embeddings(path: str):
    """Return (embeddings np.ndarray, user_ids list, user_to_idx dict)."""
    import torch
    data = torch.load(path, weights_only=False)
    emb = data['embeddings'].numpy().astype(np.float32)
    u2i = data['user_to_idx']
    # user_ids[i] is the user whose embedding is row i
    user_ids = [''] * len(u2i)
    for uid, idx in u2i.items():
        user_ids[idx] = uid
    return emb, user_ids, u2i


def _load_labels(data_dir: str) -> set:
    """Return set of malicious user IDs (empty when answers dir is absent)."""
    os.environ.setdefault('INNERSIGHT_DATA_DIR', data_dir)
    try:
        from innersight.backend.b2_data.pipeline import load_data
        data = load_data(data_dir)
        labels_raw = data['labels']   # set of (user, date) tuples
        return {uid for uid, _ in labels_raw}
    except Exception as exc:
        logger.warning('Could not load labels: %s', exc)
        return set()


def _load_ldap_departments(data_dir: str) -> dict[str, str]:
    """Map user_id → department.  Parses LDAP.csv if present; else returns {}."""
    ldap_path = os.path.join(data_dir, 'LDAP', 'month-1.csv')
    # CERT r4.2 has LDAP/ subdir with one CSV per month; month-1 covers all employees
    alt_paths = [
        ldap_path,
        os.path.join(data_dir, 'ldap.csv'),
        os.path.join(data_dir, 'LDAP.csv'),
    ]
    import pandas as pd
    for p in alt_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, usecols=lambda c: c.lower() in ('user_id', 'department'))
                df.columns = [c.lower() for c in df.columns]
                return dict(zip(df['user_id'], df['department']))
            except Exception as exc:
                logger.warning('LDAP parse failed (%s): %s', p, exc)
    return {}


def _dept_from_uid(user_ids: list[str], ldap_map: dict[str, str]) -> list[str]:
    """Return a department label per user, falling back to first-letter proxy."""
    if ldap_map:
        fallback_count = sum(1 for u in user_ids if u not in ldap_map)
        if fallback_count == 0:
            return [ldap_map[u] for u in user_ids]
        logger.info('LDAP missing %d/%d users — using first-letter proxy for those',
                    fallback_count, len(user_ids))
    return [ldap_map.get(u, u[0].upper() if u else '?') for u in user_ids]


def _dept_colors(dept_labels: list[str]):
    """Return (color_array, legend_patches) for up to 26 departments."""
    unique_depts = sorted(set(dept_labels))
    n = len(unique_depts)
    # Build a palette with enough distinct hues
    cmap = plt.colormaps['tab20'] if n <= 20 else plt.colormaps['nipy_spectral']
    palette = [cmap(i / max(n - 1, 1)) for i in range(n)]
    d2c = {d: palette[i] for i, d in enumerate(unique_depts)}
    colors = [d2c[d] for d in dept_labels]
    patches = [
        mpatches.Patch(color=d2c[d], label=d) for d in unique_depts
    ]
    return colors, patches


# ── Dimensionality reduction ──────────────────────────────────────────────────

def _run_tsne(emb: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, (emb.shape[0] - 1) / 3)
    t0 = time.perf_counter()
    proj = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, n_jobs=-1).fit_transform(emb)
    print(f'  t-SNE done in {time.perf_counter() - t0:.1f}s')
    return proj


def _run_umap(emb: np.ndarray, n_neighbors: int = 15,
              min_dist: float = 0.1, seed: int = 42) -> np.ndarray | None:
    try:
        import umap as umap_pkg
        n_neighbors = min(n_neighbors, emb.shape[0] - 1)
        t0 = time.perf_counter()
        proj = umap_pkg.UMAP(n_components=2, n_neighbors=n_neighbors,
                             min_dist=min_dist, random_state=seed).fit_transform(emb)
        print(f'  UMAP done in {time.perf_counter() - t0:.1f}s')
        return proj
    except ImportError:
        print('  umap-learn not installed — skipping UMAP plots.')
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_labels(
    proj: np.ndarray,
    user_ids: list[str],
    malicious_set: set,
    title: str,
    out_path: str,
) -> None:
    """Scatter plot: benign grey, malicious red with labels."""
    is_mal = np.array([u in malicious_set for u in user_ids])
    n_mal  = int(is_mal.sum())

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(proj[~is_mal, 0], proj[~is_mal, 1],
               c='#c0c0c0', s=12, alpha=0.6, linewidths=0, label='Benign')

    if n_mal > 0:
        ax.scatter(proj[is_mal, 0], proj[is_mal, 1],
                   c='#e32626', s=60, alpha=0.9,
                   edgecolors='black', linewidths=0.6, label='Malicious', zorder=5)
        for idx in np.where(is_mal)[0]:
            ax.annotate(user_ids[idx],
                        (proj[idx, 0], proj[idx, 1]),
                        fontsize=6, ha='left', va='bottom',
                        color='#880000', zorder=6)
    else:
        # No labels available: highlight top-5 outliers by embedding norm
        norms = np.linalg.norm(proj, axis=1)
        top_idx = np.argsort(norms)[-5:]
        ax.scatter(proj[top_idx, 0], proj[top_idx, 1],
                   c='#ff9900', s=60, alpha=0.9,
                   edgecolors='black', linewidths=0.6,
                   label='High-norm (no labels)', zorder=5)
        for idx in top_idx:
            ax.annotate(user_ids[idx],
                        (proj[idx, 0], proj[idx, 1]),
                        fontsize=6, ha='left', va='bottom', color='#884400', zorder=6)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {out_path}')


def _plot_departments(
    proj: np.ndarray,
    dept_labels: list[str],
    title: str,
    out_path: str,
) -> None:
    """Scatter plot colored by department."""
    colors, patches = _dept_colors(dept_labels)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(proj[:, 0], proj[:, 1],
               c=colors, s=14, alpha=0.7, linewidths=0)

    n_depts = len(patches)
    ncol = max(1, n_depts // 22 + 1)
    ax.legend(handles=patches, loc='upper right', fontsize=6,
              ncol=ncol, title='Dept / prefix', title_fontsize=7,
              markerscale=1.2, framealpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── Observations ──────────────────────────────────────────────────────────────

def _print_observations(
    tsne_proj: np.ndarray,
    umap_proj: np.ndarray | None,
    user_ids: list[str],
    malicious_set: set,
    dept_labels: list[str],
) -> None:
    from sklearn.metrics import silhouette_score
    from collections import Counter

    n_users = len(user_ids)
    n_mal   = sum(1 for u in user_ids if u in malicious_set)

    print()
    print('─' * 56)
    print('  Observations')
    print('─' * 56)
    print(f'  Users        : {n_users:,}')
    print(f'  Malicious    : {n_mal}  ({n_mal / n_users:.2%})')

    dept_counts = Counter(dept_labels)
    print(f'  Departments  : {len(dept_counts)}  '
          f'(sizes: min={min(dept_counts.values())}  '
          f'max={max(dept_counts.values())}  '
          f'median={int(np.median(list(dept_counts.values())))})')

    # Department clustering quality (t-SNE projection)
    dept_int = np.array([sorted(set(dept_labels)).index(d) for d in dept_labels])
    if len(set(dept_labels)) > 1:
        sil = silhouette_score(tsne_proj, dept_int, sample_size=min(n_users, 500))
        print(f'\n  Dept silhouette score (t-SNE) : {sil:.4f}')
        if sil > 0.2:
            verdict = 'Departments cluster well'
        elif sil > 0.05:
            verdict = 'Departments show some structure'
        else:
            verdict = 'Departments are heavily mixed'
        print(f'    → {verdict}')

    # Malicious outlier test
    if n_mal > 0:
        norms = np.linalg.norm(tsne_proj, axis=1)
        mal_idx   = [i for i, u in enumerate(user_ids) if u in malicious_set]
        ben_idx   = [i for i, u in enumerate(user_ids) if u not in malicious_set]
        mal_norms = norms[mal_idx]
        ben_norms = norms[ben_idx]
        print(f'\n  Malicious user t-SNE norms:')
        print(f'    mean={mal_norms.mean():.2f}  vs  benign mean={ben_norms.mean():.2f}')
        outlier_thresh = np.percentile(ben_norms, 90)
        n_outlier = int((mal_norms > outlier_thresh).sum())
        print(f'    {n_outlier}/{n_mal} malicious users are in top-10% norm (potential outliers)')
    else:
        norms = np.linalg.norm(tsne_proj, axis=1)
        top5 = np.argsort(norms)[-5:]
        print(f'\n  No labels available — top-5 norm outliers (t-SNE):')
        for i in top5:
            print(f'    {user_ids[i]}  norm={norms[i]:.2f}')

    print('─' * 56)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize user graph embeddings.')
    p.add_argument('--embeddings',  default='checkpoints/node2vec_embeddings.pt',
                   help='Path to .pt embeddings file')
    p.add_argument('--data-dir',    default=os.environ.get('INNERSIGHT_DATA_DIR',
                                                            'innersight/data/cert_r4.2'),
                   help='CERT data directory (for labels + LDAP)')
    p.add_argument('--output-dir',  default='outputs',
                   help='Directory to save PNG files')
    p.add_argument('--tsne-perplexity', type=float, default=30.0)
    p.add_argument('--umap-neighbors',  type=int,   default=15)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load embeddings ────────────────────────────────────────────────────
    print(f'Loading embeddings : {args.embeddings}')
    emb, user_ids, _ = _load_embeddings(args.embeddings)
    print(f'  shape : {emb.shape}')

    # ── 2. Load labels ────────────────────────────────────────────────────────
    print(f'Loading labels     : {args.data_dir}')
    malicious_set = _load_labels(args.data_dir)
    print(f'  malicious users  : {len(malicious_set)}')

    # ── 3. Department labels ──────────────────────────────────────────────────
    ldap_map  = _load_ldap_departments(args.data_dir)
    source    = 'LDAP.csv' if ldap_map else 'user-ID first-letter proxy'
    dept_labels = _dept_from_uid(user_ids, ldap_map)
    n_depts   = len(set(dept_labels))
    print(f'  departments      : {n_depts}  (source: {source})')

    # ── 4. t-SNE ──────────────────────────────────────────────────────────────
    print(f'\nComputing t-SNE (perplexity={args.tsne_perplexity}) ...')
    tsne_proj = _run_tsne(emb, perplexity=args.tsne_perplexity)

    emb_name = os.path.splitext(os.path.basename(args.embeddings))[0]

    print('Saving t-SNE plots ...')
    _plot_labels(
        tsne_proj, user_ids, malicious_set,
        title=f't-SNE — {emb_name}  |  Malicious Users in Red',
        out_path=os.path.join(args.output_dir, 'embeddings_tsne_labels.png'),
    )
    _plot_departments(
        tsne_proj, dept_labels,
        title=f't-SNE — {emb_name}  |  Colored by Department ({source})',
        out_path=os.path.join(args.output_dir, 'embeddings_tsne_departments.png'),
    )

    # ── 5. UMAP ───────────────────────────────────────────────────────────────
    print(f'\nComputing UMAP (n_neighbors={args.umap_neighbors}) ...')
    umap_proj = _run_umap(emb, n_neighbors=args.umap_neighbors)

    if umap_proj is not None:
        print('Saving UMAP plots ...')
        _plot_labels(
            umap_proj, user_ids, malicious_set,
            title=f'UMAP — {emb_name}  |  Malicious Users in Red',
            out_path=os.path.join(args.output_dir, 'embeddings_umap_labels.png'),
        )
        _plot_departments(
            umap_proj, dept_labels,
            title=f'UMAP — {emb_name}  |  Colored by Department ({source})',
            out_path=os.path.join(args.output_dir, 'embeddings_umap_departments.png'),
        )

    # ── 6. Observations ───────────────────────────────────────────────────────
    _print_observations(tsne_proj, umap_proj, user_ids, malicious_set, dept_labels)

    return 0


if __name__ == '__main__':
    sys.exit(main())
