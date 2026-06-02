"""Graph DataLoaders for heterogeneous GNN training on InnerSight UEBA graphs.

NeighborLoader is used for mini-batch training: given a set of seed user nodes
it samples a fixed-size neighbourhood (fanout) per hop and returns a subgraph.
A full-graph fallback is provided for small graphs / debugging.

Typical usage
-------------
    from innersight.models.graph_loader import build_graph_dataloaders
    from innersight.models.graph_builder import load_temporal_graphs

    graphs  = load_temporal_graphs()
    loaders = build_graph_dataloaders(graphs, batch_size=128)

    for batch in loaders['train_loader']:
        y_seed = batch['user'].y[:batch['user'].batch_size]
        ...
"""

from __future__ import annotations

import logging
import os

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

logger = logging.getLogger(__name__)

# Default neighbour fanout: sample ≤10 first-hop, ≤5 second-hop per edge type.
DEFAULT_NUM_NEIGHBORS: list[int] = [10, 5]


# ── Public API ────────────────────────────────────────────────────────────────

def build_graph_dataloaders(
    graphs: dict,
    batch_size: int = 128,
    num_neighbors: list[int] | None = None,
) -> dict:
    """Build NeighborLoader DataLoaders for train / val / test splits.

    Each loader samples subgraphs rooted at seed user nodes.  Training uses
    ``shuffle=True``; validation and test use ``shuffle=False`` and a larger
    batch size so the full eval set is covered in fewer iterations.

    Parameters
    ----------
    graphs:
        ``{'train': HeteroData, 'val': HeteroData, 'test': HeteroData}`` as
        returned by ``load_temporal_graphs()``.
    batch_size:
        Number of *seed* (target) user nodes per training batch.
        Val / test loaders use ``batch_size * 2``.
    num_neighbors:
        Per-hop fanout, e.g. ``[10, 5]``.  Applied uniformly across every
        edge type in the heterogeneous graph.  Defaults to ``[10, 5]``.

    Returns
    -------
    dict
        ``{'train_loader': NeighborLoader, 'val_loader': NeighborLoader,
           'test_loader': NeighborLoader}``

        Falls back to ``full_graph_loader`` for any split where
        ``NeighborLoader`` raises an exception.
    """
    if num_neighbors is None:
        num_neighbors = DEFAULT_NUM_NEIGHBORS

    def _make_loader(
        graph: HeteroData,
        bs: int,
        shuffle: bool,
        split: str,
    ) -> NeighborLoader | list[HeteroData]:
        # Per-edge-type fanout dict — cleaner than a bare list for hetero graphs.
        fanout: dict = {etype: num_neighbors for etype in graph.edge_types}
        try:
            loader = NeighborLoader(
                graph,
                num_neighbors=fanout,
                batch_size=bs,
                input_nodes=('user', None),  # seed from all user nodes
                shuffle=shuffle,
            )
            logger.info(
                '%s NeighborLoader: %d batches  (batch_size=%d  fanout=%s)',
                split, len(loader), bs, num_neighbors,
            )
            return loader
        except Exception as exc:
            logger.warning(
                '%s NeighborLoader failed (%s: %s) — using full-graph fallback',
                split, type(exc).__name__, exc,
            )
            return full_graph_loader(graph)

    return {
        'train_loader': _make_loader(graphs['train'], batch_size,     True,  'train'),
        'val_loader':   _make_loader(graphs['val'],   batch_size * 2, False, 'val'),
        'test_loader':  _make_loader(graphs['test'],  batch_size * 2, False, 'test'),
    }


def full_graph_loader(graph: HeteroData) -> list[HeteroData]:
    """Return the full graph as a single-element list (trivial loader).

    Iterating this yields exactly one "batch" — the entire graph.  Useful
    for debugging or when the graph fits comfortably in GPU memory.

    Note: unlike NeighborLoader batches, ``graph['user'].batch_size`` is NOT
    set.  Downstream code should use ``graph['user'].x.shape[0]`` as the node
    count and ``graph['user'].y`` directly for all labels.
    """
    return [graph]


def print_batch_info(loader, split: str = '') -> None:
    """Print node/edge counts and label distribution for the first batch.

    Parameters
    ----------
    loader:
        Any iterable that yields HeteroData batches (NeighborLoader or list).
    split:
        Label to display in the header (e.g. ``'train'``).
    """
    batch = next(iter(loader))
    W = 72
    header = f'First batch — {split} loader' if split else 'First batch'

    print(f'\n{"=" * W}')
    print(header)
    print(f'{"=" * W}')

    # Seed node count (only set by NeighborLoader, not full-graph loader).
    batch_size_attr = getattr(batch['user'], 'batch_size', None)
    if batch_size_attr is not None:
        print(f'\n  Seed (target) user nodes : {batch_size_attr:,}')
    else:
        print(f'\n  Full-graph loader — all nodes are targets')

    print(f'\n  Node types in subgraph:')
    print(f'  {"type":<14} {"nodes":>8}  {"feat_dim":>9}')
    print(f'  {"-" * 34}')
    for ntype in batch.node_types:
        n   = batch[ntype].x.shape[0]
        dim = batch[ntype].x.shape[1]
        print(f'  {ntype:<14} {n:>8,}  {dim:>9}')

    print(f'\n  Edge types in subgraph:')
    print(f'  {"(src, rel, dst)":<50} {"edges":>7}')
    print(f'  {"-" * 58}')
    for etype in batch.edge_types:
        n_e = batch[etype].edge_index.shape[1]
        print(f'  {str(etype):<50} {n_e:>7,}')

    if hasattr(batch['user'], 'y'):
        y     = batch['user'].y
        n_all = y.shape[0]
        # For NeighborLoader batches, only seed nodes have ground-truth labels.
        n_seed = batch_size_attr if batch_size_attr is not None else n_all
        y_seed = y[:n_seed]
        n_pos  = int(y_seed.sum().item())
        print(f'\n  Labels (seed nodes):')
        print(f'    positive : {n_pos:>5}  /  {n_seed}  ({n_pos / max(n_seed, 1):.3%})')
        print(f'    negative : {n_seed - n_pos:>5}')

    print(f'{"=" * W}')


# ── Smoke test / inspection ───────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import time

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    ap = argparse.ArgumentParser(description='Build and inspect graph DataLoaders.')
    ap.add_argument('--model-dir', default=os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
                    help='Directory containing cached graph .pt files')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Seed node batch size for NeighborLoader')
    ap.add_argument('--num-neighbors', nargs='+', type=int, default=[10, 5],
                    help='Per-hop fanout, e.g. --num-neighbors 10 5')
    args = ap.parse_args()

    os.environ['INNERSIGHT_MODEL_DIR'] = args.model_dir

    from innersight.models.graph_builder import load_temporal_graphs

    print(f'\nLoading cached temporal graphs from: {args.model_dir}')
    t0     = time.perf_counter()
    graphs = load_temporal_graphs()
    print(f'  Loaded in {time.perf_counter() - t0:.2f}s')

    for split, g in graphs.items():
        n_u = g['user'].x.shape[0]
        n_pos = int(g['user'].y.sum().item()) if hasattr(g['user'], 'y') else '?'
        print(f'  {split:<6}: {n_u} users ({n_pos} positive)  '
              f'{g.num_nodes:,} total nodes  {g.num_edges:,} total edges')

    print(f'\nBuilding DataLoaders  (batch_size={args.batch_size}  '
          f'num_neighbors={args.num_neighbors}) ...')
    t0      = time.perf_counter()
    loaders = build_graph_dataloaders(
        graphs,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
    )
    print(f'  Built in {time.perf_counter() - t0:.2f}s')

    # ── One batch from each split ─────────────────────────────────────────────
    for split, key in [('train', 'train_loader'), ('val', 'val_loader'), ('test', 'test_loader')]:
        print_batch_info(loaders[key], split=split)

    # ── Full-graph fallback demo ──────────────────────────────────────────────
    print(f'\n{"=" * 72}')
    print('Full-graph fallback loader (train graph)')
    print(f'{"=" * 72}')
    fg_loader = full_graph_loader(graphs['train'])
    print(f'  len(full_graph_loader) = {len(fg_loader)}  (always 1)')
    print_batch_info(fg_loader, split='train (full-graph)')

    # ── Loader length summary ─────────────────────────────────────────────────
    print(f'\n{"=" * 72}')
    print('DataLoader summary')
    print(f'{"=" * 72}')
    print(f'  {"split":<12} {"batches":>8}  {"batch_size":>12}  {"num_neighbors"}')
    print(f'  {"-" * 60}')
    for split, key, bs in [
        ('train',  'train_loader', args.batch_size),
        ('val',    'val_loader',   args.batch_size * 2),
        ('test',   'test_loader',  args.batch_size * 2),
    ]:
        loader = loaders[key]
        n_batches = len(loader) if hasattr(loader, '__len__') else '?'
        print(f'  {split:<12} {n_batches!s:>8}  {bs:>12}  {args.num_neighbors}')
    print(f'{"=" * 72}')
