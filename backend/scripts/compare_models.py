#!/usr/bin/env python
"""Three-model comparison: MLP (flat) vs MLP+Node2Vec vs GraphSAGE.
Phase 5 report card for InnerSight UEBA.

Usage
-----
    # GNN-only (no CERT data needed):
    python scripts/compare_models.py

    # Full three-model (needs CERT CSV data for the MLP models):
    python scripts/compare_models.py --data-dir /path/to/cert_r4.2

    # Override checkpoint paths:
    python scripts/compare_models.py \\
        --mlp-flat   checkpoints/flat/best_model.pt \\
        --mlp-n2v    checkpoints/n2v/best_model.pt  \\
        --graphsage  checkpoints/graphsage/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

logger = logging.getLogger(__name__)

_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


# ── Metric helpers ────────────────────────────────────────────────────────────

def _binary_metrics(probs: torch.Tensor, labels: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).long()
    y     = labels.long()
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return dict(precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _auc_pr(probs: torch.Tensor, labels: torch.Tensor) -> float:
    from sklearn.metrics import average_precision_score
    y_true  = labels.numpy().astype(int)
    y_score = probs.numpy()
    if y_true.sum() == 0:
        return float('nan')
    return float(average_precision_score(y_true, y_score))


def _fmt(v, kind: str = 'float') -> str:
    if v is None or (isinstance(v, float) and v != v):
        return 'N/A'
    if kind == 'int':
        return str(int(v))
    return f'{float(v):.4f}'


# ── MLP evaluation (requires raw CERT log data) ───────────────────────────────

def evaluate_mlp(
    model_path: str,
    std_path: str,
    embeddings_path: str | None = None,
    split_logs: dict | None = None,
    labels: set | None = None,
) -> dict:
    """Evaluate an MLP (flat or Node2Vec-enriched) on the given split."""
    if not os.path.exists(model_path):
        return {'error': f'checkpoint not found: {model_path}'}
    if not os.path.exists(std_path):
        return {'error': f'standardizer not found: {std_path}'}
    if split_logs is None:
        return {'error': 'no raw log data — set --data-dir to evaluate MLP models'}

    from innersight.backend.models.mlp import InsiderThreatMLP
    from innersight.backend.models.dataset import Standardizer, build_features_tensor

    ck = torch.load(model_path, weights_only=False)
    layer_sizes = ck.get('layer_sizes', [18, 64, 32, 1])
    model = InsiderThreatMLP(layer_sizes)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    std = Standardizer.load(std_path)
    X, y, user_ids = build_features_tensor(split_logs, labels or set())
    X_std = std.transform(X)

    emb_dim = None
    if embeddings_path and os.path.exists(embeddings_path):
        from innersight.backend.models.embeddings import EmbeddingManager
        mgr = EmbeddingManager(embeddings_path)
        if mgr.available:
            X_std   = mgr.get_combined_features(X_std, user_ids)
            emb_dim = mgr.embedding_dim

    if X_std.shape[0] == 0:
        return {'error': 'evaluation split is empty'}

    with torch.no_grad():
        logits = model(X_std)
        probs  = torch.sigmoid(logits).squeeze(1)

    y_flat = y.squeeze(1)
    m = _binary_metrics(probs, y_flat)
    m['auc_pr']        = _auc_pr(probs, y_flat)
    m['n_samples']     = int(X_std.shape[0])
    m['n_pos']         = int(y_flat.sum().item())
    m['model_path']    = model_path
    m['layer_sizes']   = layer_sizes
    m['embedding_dim'] = emb_dim
    m['score_mean']    = float(probs.mean().item())
    m['score_max']     = float(probs.max().item())
    m['score_std']     = float(probs.std().item())
    return m


# ── GNN evaluation (uses cached test_graph.pt — no raw data needed) ───────────

def evaluate_gnn(model_path: str) -> dict:
    """Evaluate a GraphSAGE checkpoint on the cached test_graph.pt."""
    if not os.path.exists(model_path):
        return {'error': f'checkpoint not found: {model_path}'}

    ck = torch.load(model_path, weights_only=False)
    if ck.get('model_type') != 'graphsage':
        return {'error': f'expected model_type=graphsage, got {ck.get("model_type")!r}'}

    graphs_dir      = ck.get('graphs_dir', 'checkpoints')
    test_graph_path = os.path.join(graphs_dir, 'graphs', 'test_graph.pt')
    if not os.path.exists(test_graph_path):
        return {'error': f'test graph not found: {test_graph_path}'}

    graph = torch.load(test_graph_path, weights_only=False)

    from innersight.backend.models.graphsage import InsiderThreatGNN
    from innersight.backend.models.mlp import get_device

    model_cfg = ck.get('config', {})
    model = InsiderThreatGNN(
        metadata=ck['metadata'],
        hidden_dim=model_cfg.get('hidden_dim', 128),
        num_layers=model_cfg.get('num_layers', 2),
        dropout=model_cfg.get('dropout', 0.3),
        head_layers=model_cfg.get('head_layers', [128, 64]),
    )
    model.load_state_dict(ck['state_dict'])

    device = get_device()
    model.to(device).eval()

    x_dict  = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    with torch.no_grad():
        logits = model(x_dict, ei_dict)          # (N_user, 1)
        probs  = torch.sigmoid(logits).squeeze(1).cpu()

    y = graph['user'].y.float().cpu()

    m = _binary_metrics(probs, y)
    m['auc_pr']        = _auc_pr(probs, y)
    m['n_samples']     = int(y.shape[0])
    m['n_pos']         = int(y.sum().item())
    m['model_path']    = model_path
    m['layer_sizes']   = (f"GNN h={model_cfg.get('hidden_dim',128)}"
                          f" head={model_cfg.get('head_layers',[128,64])}")
    m['embedding_dim'] = model_cfg.get('hidden_dim', 128)
    m['score_mean']    = float(probs.mean().item())
    m['score_max']     = float(probs.max().item())
    m['score_std']     = float(probs.std().item())
    return m


# ── 3-column table renderer ───────────────────────────────────────────────────

def _render_table(flat: dict, n2v: dict, gnn: dict, split_name: str) -> str:
    W   = 65
    COL = 13

    def _v(d: dict, key: str, kind: str = 'float') -> str:
        if 'error' in d:
            return '(skip)'
        return _fmt(d.get(key), kind)

    lines = []
    lines.append('=' * W)
    lines.append('  Model Comparison — InnerSight UEBA (Phase 5)')
    lines.append(f'  Evaluation split : {split_name}')
    lines.append('=' * W)
    lines.append(f'  {"Metric":<22} {"MLP (flat)":>{COL}} {"MLP+Node2Vec":>{COL}} {"GraphSAGE":>{COL}}')
    lines.append(f'  {"-"*22} {"-"*COL} {"-"*COL} {"-"*COL}')

    rows = [
        ('Precision',       'precision',  'float'),
        ('Recall',          'recall',     'float'),
        ('F1 Score',        'f1',         'float'),
        ('AUC-PR',          'auc_pr',     'float'),
        ('True Positives',  'tp',         'int'),
        ('False Positives', 'fp',         'int'),
        ('False Negatives', 'fn',         'int'),
    ]
    for label, key, kind in rows:
        lines.append(f'  {label:<22} {_v(flat,key,kind):>{COL}} '
                     f'{_v(n2v,key,kind):>{COL}} {_v(gnn,key,kind):>{COL}}')

    lines.append(f'  {"-"*22} {"-"*COL} {"-"*COL} {"-"*COL}')
    lines.append(f'  {"Samples":<22} {_v(flat,"n_samples","int"):>{COL}} '
                 f'{_v(n2v,"n_samples","int"):>{COL}} {_v(gnn,"n_samples","int"):>{COL}}')
    lines.append(f'  {"Positives in split":<22} {_v(flat,"n_pos","int"):>{COL}} '
                 f'{_v(n2v,"n_pos","int"):>{COL}} {_v(gnn,"n_pos","int"):>{COL}}')
    lines.append(f'  {"Score mean":<22} {_v(flat,"score_mean"):>{COL}} '
                 f'{_v(n2v,"score_mean"):>{COL}} {_v(gnn,"score_mean"):>{COL}}')
    lines.append(f'  {"Score max":<22} {_v(flat,"score_max"):>{COL}} '
                 f'{_v(n2v,"score_max"):>{COL}} {_v(gnn,"score_max"):>{COL}}')
    lines.append('=' * W)

    # Verdict
    f1s = {name: d.get('f1', 0.0) or 0.0
           for name, d in [('MLP (flat)', flat), ('MLP+Node2Vec', n2v), ('GraphSAGE', gnn)]
           if 'error' not in d}

    if not f1s:
        lines.append('  No models evaluated successfully.')
    elif all(v == 0 for v in f1s.values()):
        has_pos = any(d.get('n_pos', 0) > 0 for d in [flat, n2v, gnn] if 'error' not in d)
        if not has_pos:
            lines.append('  Note: evaluation split has 0 positive labels.')
            lines.append('  All F1/Precision/Recall = 0 by definition — not a model failure.')
            lines.append('  Score statistics above show what each model predicts.')
            lines.append('  Train with the full CERT dataset to see real F1/AUC-PR numbers.')
        else:
            lines.append('  All models produce F1 = 0 despite positive labels.')
            lines.append('  Check threshold, class imbalance, or model convergence.')
    else:
        best_name = max(f1s, key=f1s.get)
        best_f1   = f1s[best_name]
        baseline  = f1s.get('MLP (flat)', 0.0)
        delta     = best_f1 - baseline
        lines.append(f'  Best F1: {best_name} ({best_f1:.4f})')
        if baseline > 0 and best_name != 'MLP (flat)':
            pct = delta / baseline * 100
            lines.append(f'  Improvement over baseline: +{delta:.4f} (+{pct:.1f}%)')

    for name, d in [('MLP (flat)', flat), ('MLP+Node2Vec', n2v), ('GraphSAGE', gnn)]:
        if 'error' in d:
            lines.append(f'  {name} skipped: {d["error"]}')

    lines.append('=' * W)
    return '\n'.join(lines)


# ── t-SNE visualization ───────────────────────────────────────────────────────

def _run_tsne(emb: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, (emb.shape[0] - 1) / 3)
    t0  = time.perf_counter()
    proj = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_jobs=-1).fit_transform(emb)
    print(f'    t-SNE done in {time.perf_counter() - t0:.1f}s')
    return proj


def plot_gnn_tsne(
    gnn_emb_path: str,
    n2v_emb_path: str | None,
    output_path: str,
    malicious_set: set | None = None,
) -> None:
    """Side-by-side t-SNE: Node2Vec (left) vs GNN embeddings (right)."""
    if not os.path.exists(gnn_emb_path):
        print(f'  GNN embeddings not found: {gnn_emb_path} — skipping t-SNE')
        return

    def _load(path: str):
        ck  = torch.load(path, weights_only=False)
        emb = ck['embeddings'].numpy().astype(np.float32)
        u2i = ck['user_to_idx']
        ids = [''] * len(u2i)
        for uid, idx in u2i.items():
            ids[idx] = uid
        return emb, ids

    print('  Loading GNN embeddings …')
    gnn_emb, gnn_ids = _load(gnn_emb_path)
    print(f'    shape: {gnn_emb.shape}')

    n2v_emb, n2v_ids = None, None
    if n2v_emb_path and os.path.exists(n2v_emb_path):
        print('  Loading Node2Vec embeddings …')
        n2v_emb, n2v_ids = _load(n2v_emb_path)
        print(f'    shape: {n2v_emb.shape}')

    mal = malicious_set or set()

    print('  Running t-SNE on GNN embeddings …')
    gnn_proj = _run_tsne(gnn_emb)

    n2v_proj = None
    if n2v_emb is not None:
        print('  Running t-SNE on Node2Vec embeddings …')
        n2v_proj = _run_tsne(n2v_emb)

    ncols = 2 if n2v_proj is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    def _scatter(ax, proj, ids, title):
        is_mal = np.array([u in mal for u in ids])
        ax.scatter(proj[~is_mal, 0], proj[~is_mal, 1],
                   c='#c0c0c0', s=10, alpha=0.6, linewidths=0, label='Benign')
        if is_mal.sum() > 0:
            ax.scatter(proj[is_mal, 0], proj[is_mal, 1],
                       c='#e32626', s=50, alpha=0.9,
                       edgecolors='black', linewidths=0.5,
                       label='Malicious', zorder=5)
            for i in np.where(is_mal)[0]:
                ax.annotate(ids[i], (proj[i, 0], proj[i, 1]),
                            fontsize=6, color='#880000', zorder=6)
        else:
            norms = np.linalg.norm(proj, axis=1)
            top   = np.argsort(norms)[-5:]
            ax.scatter(proj[top, 0], proj[top, 1],
                       c='#ff9900', s=50, alpha=0.9,
                       edgecolors='black', linewidths=0.5,
                       label='Top-5 norm outliers (no labels)', zorder=5)
            for i in top:
                ax.annotate(ids[i], (proj[i, 0], proj[i, 1]),
                            fontsize=6, color='#884400', zorder=6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')

    if n2v_proj is not None:
        _scatter(axes[0], n2v_proj, n2v_ids,
                 f'Node2Vec t-SNE  ({n2v_emb.shape[1]}-dim)')
    _scatter(axes[-1], gnn_proj, gnn_ids,
             f'GNN (GraphSAGE) t-SNE  ({gnn_emb.shape[1]}-dim)')

    fig.suptitle('Embedding Comparison — InnerSight Phase 5', fontsize=13, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {output_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Phase 5 three-model comparison: MLP / MLP+N2V / GraphSAGE'
    )
    p.add_argument('--data-dir',    default=os.environ.get('INNERSIGHT_DATA_DIR'),
                   help='CERT data directory (needed for MLP evaluation)')
    p.add_argument('--split',       default=None, choices=['train', 'val', 'test'])
    p.add_argument('--mlp-flat',    default='checkpoints/flat/best_model.pt')
    p.add_argument('--mlp-flat-std', default='checkpoints/flat/standardizer.pt')
    p.add_argument('--mlp-n2v',     default='checkpoints/n2v/best_model.pt')
    p.add_argument('--mlp-n2v-std', default='checkpoints/n2v/standardizer.pt')
    p.add_argument('--n2v-emb',     default='checkpoints/node2vec_embeddings.pt')
    p.add_argument('--graphsage',   default='checkpoints/graphsage/best_model.pt')
    p.add_argument('--gnn-emb',     default='checkpoints/gnn_embeddings.pt')
    p.add_argument('--output-dir',  default=_OUTPUTS_DIR)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    args   = _parse_args()
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    os.environ.setdefault('INNERSIGHT_MODEL_DIR', 'checkpoints')

    # ── 1. Optionally load raw data (MLP models only) ─────────────────────────
    split_logs, labels, split_name = None, None, 'test (cached graphs only)'
    if args.data_dir and os.path.exists(args.data_dir):
        try:
            os.environ['INNERSIGHT_DATA_DIR'] = args.data_dir
            from innersight.backend.data.pipeline import load_data
            print(f'Loading raw data from: {args.data_dir}')
            data   = load_data(args.data_dir)
            splits = data['splits']
            labels = data['labels']

            candidate = args.split
            if not candidate:
                for c in ('test', 'val', 'train'):
                    if any(len(df) > 0 for df in splits[c].values()):
                        candidate = c
                        break
            split_logs = splits[candidate]
            split_name = candidate
            n_rows = sum(len(df) for df in split_logs.values())
            print(f'  split={split_name}  rows={n_rows:,}  labels={len(labels)}')
        except Exception as exc:
            print(f'Could not load raw data ({exc})')
            print('MLP models will be skipped; GNN proceeds on cached graphs.')
    else:
        print('No --data-dir set — MLP models skipped '
              '(GNN evaluates on cached test_graph.pt)')

    print()

    # ── 2. Evaluate ───────────────────────────────────────────────────────────
    print(f'Evaluating  [1/3]  MLP flat        ({args.mlp_flat}) …')
    flat = evaluate_mlp(args.mlp_flat, args.mlp_flat_std,
                        split_logs=split_logs, labels=labels)
    if 'error' in flat:
        print(f'  skipped: {flat["error"]}')
    else:
        print(f'  F1={flat["f1"]:.4f}  P={flat["precision"]:.4f}  '
              f'R={flat["recall"]:.4f}  AUC-PR={_fmt(flat["auc_pr"])}  '
              f'n={flat["n_samples"]:,}  pos={flat["n_pos"]}')

    print(f'Evaluating  [2/3]  MLP + Node2Vec  ({args.mlp_n2v}) …')
    n2v = evaluate_mlp(args.mlp_n2v, args.mlp_n2v_std,
                       embeddings_path=args.n2v_emb,
                       split_logs=split_logs, labels=labels)
    if 'error' in n2v:
        print(f'  skipped: {n2v["error"]}')
    else:
        print(f'  F1={n2v["f1"]:.4f}  P={n2v["precision"]:.4f}  '
              f'R={n2v["recall"]:.4f}  AUC-PR={_fmt(n2v["auc_pr"])}  '
              f'n={n2v["n_samples"]:,}  pos={n2v["n_pos"]}  '
              f'emb_dim={n2v["embedding_dim"]}')

    print(f'Evaluating  [3/3]  GraphSAGE       ({args.graphsage}) …')
    gnn = evaluate_gnn(args.graphsage)
    if 'error' in gnn:
        print(f'  skipped: {gnn["error"]}')
    else:
        print(f'  F1={gnn["f1"]:.4f}  P={gnn["precision"]:.4f}  '
              f'R={gnn["recall"]:.4f}  AUC-PR={_fmt(gnn["auc_pr"])}  '
              f'n={gnn["n_samples"]:,}  pos={gnn["n_pos"]}  '
              f'score_max={gnn["score_max"]:.4f}  score_mean={gnn["score_mean"]:.4f}')

    # ── 3. Print table ────────────────────────────────────────────────────────
    table = _render_table(flat, n2v, gnn, split_name)
    print()
    print(table)

    # ── 4. Save report ────────────────────────────────────────────────────────
    benchmark_path = os.path.join(outdir, 'phase5_benchmark.txt')
    import datetime
    with open(benchmark_path, 'w') as fh:
        fh.write('InnerSight UEBA — Phase 5 Benchmark\n')
        fh.write(f'Generated: {datetime.datetime.now().isoformat()}\n\n')
        fh.write(table)
        fh.write('\n\nModel paths evaluated:\n')
        for name, path in [('MLP flat',  args.mlp_flat),
                            ('MLP+N2V',  args.mlp_n2v),
                            ('GraphSAGE', args.graphsage)]:
            status = 'OK' if os.path.exists(path) else 'MISSING'
            fh.write(f'  {name:<12}: {path}  [{status}]\n')
    print(f'\nReport saved → {benchmark_path}')

    # ── 5. GNN t-SNE visualization ────────────────────────────────────────────
    print('\nGenerating GNN embedding t-SNE …')
    tsne_path = os.path.join(outdir, 'gnn_embeddings_tsne.png')
    plot_gnn_tsne(
        gnn_emb_path=args.gnn_emb,
        n2v_emb_path=args.n2v_emb,
        output_path=tsne_path,
    )

    if os.path.exists(tsne_path):
        n2v_tsne = os.path.join(outdir, 'embeddings_tsne_labels.png')
        print()
        print('Visual comparison:')
        print(f'  GNN t-SNE (new)    → {tsne_path}')
        print(f'  N2V t-SNE (Phase4) → {n2v_tsne}')
        print('  Open both images side-by-side to compare cluster structure.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
