"""Node2Vec trainer for the InnerSight heterogeneous CERT graph.

Converts the HeteroData graph to a homogeneous representation (manually
tracking user node offsets), trains Node2Vec with biased random walks (q<1
for a DFS-lean that captures cross-department behaviour), then slices out
user-only embeddings for downstream anomaly scoring.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import Node2Vec

_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND  = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PKG_ROOT = os.path.abspath(os.path.join(_BACKEND, '..', '..'))
for _p in (_PKG_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# Canonical ordering for the manual hetero→homo merge.
# Users are first so that row i in user_to_idx == homogeneous index i.
_NODE_ORDER = ['user', 'pc', 'url', 'file']


# ── Homogeneous conversion ────────────────────────────────────────────────────

def _hetero_to_homo(
    graph: HeteroData,
) -> tuple[torch.Tensor, int, torch.Tensor, dict[str, int]]:
    """Merge all edge types into one homogeneous edge_index with offset tracking.

    Node types are concatenated in _NODE_ORDER. Each type keeps its own
    0-based local indices, offset by the cumulative count of prior types:

        user   → indices  0 .. num_users-1
        pc     → indices  num_users .. num_users+num_pcs-1
        url    → indices  num_users+num_pcs .. (+ num_urls) - 1
        file   → indices  num_users+num_pcs+num_urls .. (+ num_files) - 1

    Isolated nodes (degree 0 in the merged graph) are pruned from the index
    space. Only user nodes that retain edges after pruning are returned in
    user_indices; the mapping is consistent because user nodes are always first.

    Returns
    -------
    homo_edge_index : (2, total_edges) int64 — edges in compact index space
    total_nodes     : number of nodes with at least one edge
    user_indices    : 1-D int64 tensor of compact indices that are users
    offsets         : {node_type: original offset} — for logging
    """
    node_counts: dict[str, int] = {
        ntype: (graph[ntype].x.shape[0] if ntype in graph.node_types else 0)
        for ntype in _NODE_ORDER
    }

    # ── Original (un-pruned) offsets ─────────────────────────────────────────
    offsets: dict[str, int] = {}
    cumulative = 0
    for ntype in _NODE_ORDER:
        offsets[ntype] = cumulative
        cumulative += node_counts[ntype]
    total_nodes_raw = cumulative

    # ── Collect all edges in the raw homogeneous space ───────────────────────
    parts: list[torch.Tensor] = []
    for src_type, rel, dst_type in graph.edge_types:
        ei = graph[src_type, rel, dst_type].edge_index
        if ei.shape[1] == 0:
            continue
        src_off = offsets.get(src_type, 0)
        dst_off = offsets.get(dst_type, 0)
        shift = torch.tensor([[src_off], [dst_off]], dtype=torch.long)
        parts.append(ei + shift)

    if not parts:
        raise ValueError('No edges found in the graph — cannot run Node2Vec.')

    raw_edge_index = torch.cat(parts, dim=1)

    # ── Prune isolated nodes (degree-0 in the merged graph) ──────────────────
    # Find every node that appears at least once in the edge list.
    connected = torch.unique(raw_edge_index.view(-1))          # sorted compact node ids
    total_nodes = int(connected.shape[0])

    # Build a remapping from raw → compact index.
    remap = torch.full((total_nodes_raw,), -1, dtype=torch.long)
    remap[connected] = torch.arange(total_nodes, dtype=torch.long)

    homo_edge_index = remap[raw_edge_index]                    # (2, E) in compact space

    # ── User indices in compact space ─────────────────────────────────────────
    # Raw user indices are offsets['user'] .. offsets['user'] + n_users - 1.
    n_users = node_counts['user']
    raw_user_ids = torch.arange(offsets['user'], offsets['user'] + n_users, dtype=torch.long)
    # Keep only users that have at least one edge.
    valid_mask = remap[raw_user_ids] != -1
    user_indices = remap[raw_user_ids[valid_mask]]

    return homo_edge_index, total_nodes, user_indices, offsets


# ── Public API ────────────────────────────────────────────────────────────────

def train_node2vec(graph: HeteroData, config: dict | None = None) -> torch.Tensor:
    """Train Node2Vec on the CERT graph and return user-only embeddings.

    Parameters
    ----------
    graph:  HeteroData as returned by load_temporal_graphs()['train'].
    config: Hyperparameter overrides (all optional).

    Returns
    -------
    user_embeddings : float32 tensor, shape (num_users, embedding_dim)
    """
    if config is None:
        config = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)

    # ── 1. Homogeneous conversion ─────────────────────────────────────────────
    homo_edge_index, total_nodes, user_indices, offsets = _hetero_to_homo(graph)

    n_users = int(user_indices.shape[0])
    if n_users == 0:
        raise ValueError('Graph has 0 user nodes — cannot produce user embeddings.')

    raw_counts = {nt: (graph[nt].x.shape[0] if nt in graph.node_types else 0) for nt in _NODE_ORDER}
    raw_total  = sum(raw_counts.values())
    print(f'Heterogeneous : {raw_total:,} raw nodes  '
          f'({raw_counts["user"]:,} users, {raw_counts["pc"]:,} pcs, '
          f'{raw_counts["url"]:,} urls, {raw_counts["file"]:,} files)')
    print(f'Homogeneous   : {total_nodes:,} nodes after pruning isolated  |  {homo_edge_index.shape[1]:,} edges')
    print(f'User indices  : {n_users:,} users active in homogeneous space'
          f'  (compact idx {int(user_indices[0])}..{int(user_indices[-1])})')

    # ── 2. Node2Vec model ─────────────────────────────────────────────────────
    model = Node2Vec(
        edge_index=homo_edge_index,
        embedding_dim=config.get('embedding_dim', 128),
        walk_length=config.get('walk_length', 20),
        context_size=config.get('context_size', 10),
        walks_per_node=config.get('walks_per_node', 20),
        num_negative_samples=config.get('num_negative_samples', 1),
        p=config.get('p', 1.0),
        q=config.get('q', 0.8),
        sparse=True,
        num_nodes=total_nodes,
    ).to(device)

    # ── 3. Training ───────────────────────────────────────────────────────────
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    epochs = config.get('epochs', 50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}: loss = {total_loss:.4f}')

    # ── 4. User embeddings ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_embeddings = model().detach().cpu()  # (total_nodes, emb_dim)

    user_embeddings = all_embeddings[user_indices]  # (num_users, emb_dim)
    logger.info('user_embeddings: %s', tuple(user_embeddings.shape))
    return user_embeddings


def train_metapath2vec(graph: HeteroData, config: dict | None = None) -> torch.Tensor:
    """Train MetaPath2Vec on the CERT graph and return user-only embeddings.

    One model is trained per valid metapath (all edge types in the path must
    exist and have > 0 edges). User embeddings from every metapath are
    concatenated: shape is (num_users, embedding_dim × num_valid_metapaths).

    Parameters
    ----------
    graph:  HeteroData as returned by load_temporal_graphs()['train'].
    config: Hyperparameter overrides (all optional).

    Returns
    -------
    user_embeddings : float32 tensor, shape (num_users, emb_dim * n_paths)
    """
    from torch_geometric.nn import MetaPath2Vec  # noqa: PLC0415

    if config is None:
        config = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)

    # ── Candidate metapaths ───────────────────────────────────────────────────
    CANDIDATE_METAPATHS = [
        ('http',  [('user', 'http_request',  'url'),  ('url',  'rev_http_request', 'user')]),
        ('logon', [('user', 'logon',          'pc'),   ('pc',   'rev_logon',        'user')]),
        ('email', [('user', 'email_to',       'user')]),
    ]

    # Only keep paths whose every edge type exists and has edges.
    available: set[tuple] = {
        etype for etype in graph.edge_types
        if graph[etype].edge_index.shape[1] > 0
    }

    valid_metapaths = []
    for name, path in CANDIDATE_METAPATHS:
        if all(tuple(e) in available for e in path):
            valid_metapaths.append((name, path))
            logger.info('Metapath "%s" is valid.', name)
        else:
            missing = [e for e in path if tuple(e) not in available]
            logger.info('Metapath "%s" skipped — missing edge types: %s', name, missing)

    if not valid_metapaths:
        raise ValueError('No valid metapaths found in graph — cannot run MetaPath2Vec.')

    print(f'Valid metapaths : {[n for n, _ in valid_metapaths]}')

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    embedding_dim        = config.get('embedding_dim',        64)
    walk_length          = config.get('walk_length',          20)
    context_size         = config.get('context_size',         7)
    walks_per_node       = config.get('walks_per_node',       10)
    num_negative_samples = config.get('num_negative_samples', 1)
    batch_size           = config.get('batch_size',           128)
    lr                   = config.get('lr',                   0.01)
    epochs               = config.get('epochs',               50)

    num_nodes_dict = {
        ntype: graph[ntype].x.shape[0] if ntype in graph.node_types else 0
        for ntype in graph.node_types
    }

    n_users = num_nodes_dict.get('user', 0)
    if n_users == 0:
        raise ValueError('Graph has 0 user nodes.')

    print(f'Users           : {n_users:,}')
    print(f'embedding_dim   : {embedding_dim}  (per metapath)')

    # ── Train one model per metapath, collect user embeddings ─────────────────
    all_user_embeddings: list[torch.Tensor] = []

    for name, metapath in valid_metapaths:
        print(f'\nTraining MetaPath2Vec — metapath "{name}" ...')

        model = MetaPath2Vec(
            edge_index_dict=graph.edge_index_dict,
            embedding_dim=embedding_dim,
            metapath=metapath,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            num_nodes_dict=num_nodes_dict,
            sparse=True,
        ).to(device)

        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
        loader    = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f'  [{name}] Epoch {epoch:3d} | loss = {total_loss:.4f}')

        model.eval()
        with torch.no_grad():
            user_emb = model('user').detach().cpu()   # (num_users, embedding_dim)

        print(f'  [{name}] user embedding : {tuple(user_emb.shape)}')
        all_user_embeddings.append(user_emb)

    user_embeddings = torch.cat(all_user_embeddings, dim=1)
    logger.info('MetaPath2Vec user_embeddings: %s', tuple(user_embeddings.shape))
    return user_embeddings


def save_embeddings(embeddings: torch.Tensor, user_to_idx: dict, path: str) -> None:
    """Persist user embeddings and the user→index mapping to a .pt file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'embeddings': embeddings, 'user_to_idx': user_to_idx}, path)
    logger.info('Saved embeddings → %s', path)


def load_embeddings(path: str) -> tuple[torch.Tensor, dict]:
    """Load embeddings and user→index mapping from a .pt file."""
    data = torch.load(path, weights_only=False)
    return data['embeddings'], data['user_to_idx']


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    ap = argparse.ArgumentParser(description='Train Node2Vec on the CERT training graph.')
    ap.add_argument('--data-dir',  default=os.environ.get('INNERSIGHT_DATA_DIR', ''),
                    help='Path to CERT log CSVs (INNERSIGHT_DATA_DIR)')
    ap.add_argument('--model-dir', default=os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
                    help='Checkpoint directory (INNERSIGHT_MODEL_DIR)')
    ap.add_argument('--epochs',    type=int,   default=50,  help='Training epochs')
    ap.add_argument('--emb-dim',   type=int,   default=128, help='Embedding dimension')
    ap.add_argument('--q',         type=float, default=0.8, help='Node2Vec in-out parameter')
    args = ap.parse_args()
    if not args.data_dir:
        ap.error('--data-dir or INNERSIGHT_DATA_DIR required')

    os.environ['INNERSIGHT_DATA_DIR']  = args.data_dir
    os.environ['INNERSIGHT_MODEL_DIR'] = args.model_dir

    from innersight.backend.models.graph_builder import load_temporal_graphs

    wall = time.perf_counter()

    print(f'\nLoading training graph ...')
    print(f'  data-dir  : {args.data_dir}')
    print(f'  model-dir : {args.model_dir}')
    t0 = time.perf_counter()
    graph = load_temporal_graphs()['train']
    print(f'  loaded in {time.perf_counter() - t0:.1f}s')
    print(f'  user : {graph["user"].x.shape[0]:,} nodes  (feat_dim={graph["user"].x.shape[1]})')
    print(f'  pc   : {graph["pc"].x.shape[0]:,} nodes')
    print(f'  url  : {graph["url"].x.shape[0]:,} nodes')
    print(f'  file : {graph["file"].x.shape[0]:,} nodes')
    for etype in graph.edge_types:
        n_e = graph[etype].edge_index.shape[1]
        if n_e > 0:
            print(f'  {str(etype):<45} {n_e:>9,} edges')

    config = {
        'embedding_dim':        args.emb_dim,
        'walk_length':          20,
        'context_size':         10,
        'walks_per_node':       20,
        'num_negative_samples': 1,
        'p':                    1.0,
        'q':                    args.q,
        'epochs':               args.epochs,
    }

    print(f'\nTraining Node2Vec  (epochs={args.epochs}  emb_dim={args.emb_dim}  q={args.q}) ...')
    t_train = time.perf_counter()
    embeddings = train_node2vec(graph, config)
    train_secs = time.perf_counter() - t_train

    e = embeddings
    print(f'\nEmbedding stats:')
    print(f'  shape : {tuple(e.shape)}')
    print(f'  min   : {e.min().item():.4f}')
    print(f'  max   : {e.max().item():.4f}')
    print(f'  mean  : {e.mean().item():.4f}')
    print(f'  std   : {e.std().item():.4f}')

    out_path = os.path.join(args.model_dir, 'node2vec_embeddings.pt')
    save_embeddings(embeddings, graph.user_to_idx, out_path)
    print(f'\nSaved → {out_path}')
    print(f'Training time : {train_secs:.1f}s')
    print(f'Total time    : {time.perf_counter() - wall:.1f}s')
