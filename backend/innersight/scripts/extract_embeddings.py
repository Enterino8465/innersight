"""Extract and save learned GNN user embeddings.

The saved file is format-compatible with EmbeddingManager so GNN embeddings
can be dropped in wherever Node2Vec embeddings are used (e.g. as input
features to the MLP head in downstream experiments).

Saved format (identical to node2vec_trainer.save_embeddings):
  torch.save({'embeddings': FloatTensor(N, hidden_dim),
               'user_to_idx': {user_id: row_index}}, output_path)

Usage
-----
  # From backend/
  python scripts/extract_embeddings.py \\
      --model  checkpoints/graphsage/best_model.pt \\
      --output checkpoints/gnn_embeddings.pt

  # Optionally override the graph (default: train_graph.pt from checkpoint's graphs_dir)
  python scripts/extract_embeddings.py \\
      --model  checkpoints/graphsage/best_model.pt \\
      --graph  checkpoints/graphs/train_graph.pt \\
      --output checkpoints/gnn_embeddings.pt
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.nn.functional as F

from innersight.models.graphsage import InsiderThreatGNN
from innersight.models.mlp import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_gnn_embeddings(
    model_path: str,
    graph_path: str | None = None,
    output_path: str = 'checkpoints/gnn_embeddings.pt',
) -> torch.Tensor:
    """Extract per-user embeddings from a trained GNN checkpoint.

    Parameters
    ----------
    model_path:
        Path to a checkpoint saved by ``gnn_trainer.train_gnn()``.
        Must contain ``model_type == 'graphsage'``.
    graph_path:
        Optional path to the HeteroData ``.pt`` file.  If omitted, the
        function reads ``graphs_dir`` from the checkpoint and loads
        ``{graphs_dir}/graphs/train_graph.pt``.
    output_path:
        Where to write ``{'embeddings', 'user_to_idx'}`` so that
        ``EmbeddingManager`` can load the file unchanged.

    Returns
    -------
    torch.Tensor
        Shape ``(num_users, hidden_dim)``.
    """
    device = get_device()

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    logger.info('Loading checkpoint from %s', model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model_type = ckpt.get('model_type', 'unknown')
    if model_type != 'graphsage':
        raise ValueError(
            f"extract_gnn_embeddings requires model_type='graphsage', "
            f"got {model_type!r} in {model_path}"
        )

    # ── 2. Resolve graph path ─────────────────────────────────────────────────
    if graph_path is None:
        graphs_dir = ckpt.get('graphs_dir', 'checkpoints')
        graph_path = os.path.join(graphs_dir, 'graphs', 'train_graph.pt')
        logger.info('Graph path not specified — using %s', graph_path)

    logger.info('Loading graph from %s', graph_path)
    graph = torch.load(graph_path, weights_only=False)

    # ── 3. Build model and load weights ──────────────────────────────────────
    model_cfg = ckpt.get('config', {})
    model = InsiderThreatGNN(
        metadata=ckpt['metadata'],
        hidden_dim=model_cfg.get('hidden_dim', 128),
        num_layers=model_cfg.get('num_layers', 2),
        dropout=model_cfg.get('dropout', 0.3),
        head_layers=model_cfg.get('head_layers', [128, 64]),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    logger.info(
        'Model loaded: hidden_dim=%d  num_layers=%d  head_layers=%s',
        model_cfg.get('hidden_dim', 128),
        model_cfg.get('num_layers', 2),
        model_cfg.get('head_layers', [128, 64]),
    )

    # ── 4. Full-graph embedding extraction ────────────────────────────────────
    x_dict  = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    with torch.no_grad():
        embeddings = model.get_embeddings(x_dict, ei_dict)   # (N, hidden_dim)

    embeddings = embeddings.cpu().float()
    logger.info('Embeddings shape: %s', tuple(embeddings.shape))

    # ── 5. Save in EmbeddingManager-compatible format ─────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({'embeddings': embeddings, 'user_to_idx': graph.user_to_idx}, output_path)
    logger.info('Saved GNN embeddings → %s', output_path)

    return embeddings


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_embeddings(
    gnn_path: str,
    n2v_path: str,
) -> None:
    """Print cosine-similarity stats between GNN and Node2Vec embeddings.

    Both files must have the same ``user_to_idx`` key set (same users).
    If embedding dimensions differ the larger matrix is truncated to match
    the smaller so the comparison is still meaningful.
    """
    gnn_ckpt = torch.load(gnn_path, weights_only=False)
    n2v_ckpt = torch.load(n2v_path, weights_only=False)

    gnn_emb: torch.Tensor = gnn_ckpt['embeddings'].float()   # (N, d_gnn)
    n2v_emb: torch.Tensor = n2v_ckpt['embeddings'].float()   # (N, d_n2v)

    gnn_u2i: dict = gnn_ckpt['user_to_idx']
    n2v_u2i: dict = n2v_ckpt['user_to_idx']

    # ── Align by shared user IDs ──────────────────────────────────────────────
    shared = sorted(set(gnn_u2i) & set(n2v_u2i))
    if not shared:
        print('  No shared users between GNN and Node2Vec embeddings.')
        return

    gnn_rows = torch.stack([gnn_emb[gnn_u2i[u]] for u in shared])   # (M, d_gnn)
    n2v_rows = torch.stack([n2v_emb[n2v_u2i[u]] for u in shared])   # (M, d_n2v)

    d_gnn = gnn_rows.shape[1]
    d_n2v = n2v_rows.shape[1]
    d_min = min(d_gnn, d_n2v)

    if d_gnn != d_n2v:
        print(f'  Dimension mismatch: GNN={d_gnn}  Node2Vec={d_n2v}')
        print(f'  Truncating to min dim={d_min} for cosine comparison.')
        gnn_rows = gnn_rows[:, :d_min]
        n2v_rows = n2v_rows[:, :d_min]

    # Row-wise cosine similarity: cos(gnn_u, n2v_u) for each shared user
    cos_sim = F.cosine_similarity(gnn_rows, n2v_rows, dim=1)   # (M,)

    print(f'  Shared users     : {len(shared):,}')
    print(f'  GNN dim          : {d_gnn}')
    print(f'  Node2Vec dim     : {d_n2v}')
    print(f'  Cosine similarity (per-user GNN vs N2V):')
    print(f'    mean : {cos_sim.mean().item():.4f}')
    print(f'    std  : {cos_sim.std().item():.4f}')
    print(f'    min  : {cos_sim.min().item():.4f}')
    print(f'    max  : {cos_sim.max().item():.4f}')
    print()
    print('  Interpretation:')
    print('    Low mean similarity = methods learned different representations')
    print('    (expected — GNN uses graph structure, N2V uses random walks)')
    print('    Both can be useful: GNN captures relational context,')
    print('    N2V captures co-occurrence in the user-resource graph.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Extract GNN user embeddings and save in EmbeddingManager format.'
    )
    p.add_argument(
        '--model',
        default=os.path.join(
            os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
            'graphsage', 'best_model.pt',
        ),
        metavar='PATH',
        help='GNN checkpoint (must have model_type=graphsage)',
    )
    p.add_argument(
        '--graph',
        default=None,
        metavar='PATH',
        help='HeteroData graph .pt file (default: from checkpoint graphs_dir)',
    )
    p.add_argument(
        '--output',
        default=os.path.join(
            os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
            'gnn_embeddings.pt',
        ),
        metavar='PATH',
        help='Output path for embeddings (default: checkpoints/gnn_embeddings.pt)',
    )
    p.add_argument(
        '--compare-n2v',
        default=None,
        metavar='PATH',
        help='If given, compare GNN embeddings against Node2Vec embeddings at this path.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    args = _parse_args()

    DIVIDER = '=' * 60

    print(f'\n{DIVIDER}')
    print('GNN Embedding Extraction')
    print(DIVIDER)
    print(f'  model  : {args.model}')
    print(f'  graph  : {args.graph or "(from checkpoint)"}')
    print(f'  output : {args.output}')
    print()

    embeddings = extract_gnn_embeddings(
        model_path=args.model,
        graph_path=args.graph,
        output_path=args.output,
    )

    print(f'\n{DIVIDER}')
    print('Extraction results')
    print(DIVIDER)
    print(f'  Embedding shape : {tuple(embeddings.shape)}')
    print(f'  dtype           : {embeddings.dtype}')
    print(f'  norm (mean)     : {embeddings.norm(dim=1).mean().item():.4f}')
    print(f'  norm (std)      : {embeddings.norm(dim=1).std().item():.4f}')
    print(f'  value range     : [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]')

    # ── Verify EmbeddingManager can load it ───────────────────────────────────
    print(f'\n{DIVIDER}')
    print('EmbeddingManager compatibility check')
    print(DIVIDER)
    from innersight.models.embeddings import EmbeddingManager

    mgr = EmbeddingManager(args.output)
    assert mgr.available, 'EmbeddingManager failed to load!'
    assert mgr.embedding_dim == embeddings.shape[1], 'Dimension mismatch!'

    ckpt_loaded = torch.load(args.output, weights_only=False)
    user_to_idx = ckpt_loaded['user_to_idx']
    sample_users = list(user_to_idx.keys())[:5]

    print(f'  available      : {mgr.available}')
    print(f'  embedding_dim  : {mgr.embedding_dim}')
    print(f'  users in graph : {len(user_to_idx):,}')
    print(f'  sample users   : {sample_users}')

    aligned = mgr.align_embeddings(sample_users)
    print(f'  align_embeddings({len(sample_users)} users) shape: {tuple(aligned.shape)}')
    print('  EmbeddingManager: OK')

    # ── Compare with Node2Vec (if requested / auto-detect) ────────────────────
    model_dir = os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints')
    n2v_path  = args.compare_n2v or os.path.join(model_dir, 'node2vec_embeddings.pt')

    if os.path.exists(n2v_path):
        print(f'\n{DIVIDER}')
        print(f'Comparison: GNN vs Node2Vec')
        print(f'  Node2Vec file: {n2v_path}')
        print(DIVIDER)
        _compare_embeddings(args.output, n2v_path)
    else:
        print(f'\n  (Node2Vec embeddings not found at {n2v_path} — skipping comparison)')
