#!/usr/bin/env python
"""Comprehensive validation and visualization of InnerSight temporal graphs.

Usage
-----
    python scripts/validate_graphs.py [--model-dir DIR] [--data-dir DIR] [--out-dir DIR]

Reads cached train/val/test HeteroData objects, runs structural and semantic
checks, cross-checks edge counts against raw CSVs (if DATA_DIR is provided),
and saves a neighbourhood plot for one malicious user to outputs/.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import warnings

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND  = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PKG_ROOT = os.path.abspath(os.path.join(_BACKEND, '..', '..'))
for _p in (_PKG_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from innersight.backend.models.graph_schema import (
    NODE_USER, NODE_PC, NODE_URL, NODE_FILE,
    EDGE_LOGON, EDGE_USB, EDGE_EMAIL, EDGE_HTTP, EDGE_FILE_COPY,
    REV_EDGE_LOGON, REV_EDGE_USB, REV_EDGE_EMAIL, REV_EDGE_HTTP, REV_EDGE_FILE_COPY,
    NODE_FEATURE_DIMS, EDGE_FEATURE_DIMS, ALL_EDGE_TYPES, ALL_REV_EDGE_TYPES,
)

logger = logging.getLogger(__name__)

# ── Display constants ─────────────────────────────────────────────────────────
W  = 76        # table width
W2 = 60        # narrow table width

# Expected counts for the real CERT r4.2 dataset (used for warnings only).
EXPECTED_USERS   = 1000
EXPECTED_POS_MAX = 100   # upper bound on malicious users in any single window


# ── Check registry ────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warned: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warned.append(msg)
        warnings.warn(msg, stacklevel=2)

    def check(self, condition: bool, ok_msg: str, fail_msg: str, warn: bool = False) -> bool:
        if condition:
            self.ok(ok_msg)
            return True
        if warn:
            self.warn(fail_msg)
        else:
            self.fail(fail_msg)
        return False


# ── Stats table ───────────────────────────────────────────────────────────────

def _row(label: str, value, width: int = 36) -> None:
    print(f'  {label:<{width}} {value}')


def print_graph_stats(split: str, g) -> None:
    """Print a detailed stats table for one HeteroData graph."""
    print(f'\n{"=" * W}')
    print(f'  {split.upper()} GRAPH')
    print(f'{"=" * W}')

    # ── Node counts ──────────────────────────────────────────────────────────
    print(f'\n  NODE COUNTS')
    print(f'  {"-" * (W - 2)}')
    print(f'  {"type":<10} {"count":>8}  {"feat_dim":>9}')
    print(f'  {"-" * 30}')
    for ntype in (NODE_USER, NODE_PC, NODE_URL, NODE_FILE):
        n   = g[ntype].x.shape[0]
        dim = g[ntype].x.shape[1]
        print(f'  {ntype:<10} {n:>8,}  {dim:>9}')
    print(f'  {"-" * 30}')
    print(f'  {"TOTAL":<10} {g.num_nodes:>8,}')

    # ── Edge counts ──────────────────────────────────────────────────────────
    print(f'\n  EDGE COUNTS')
    print(f'  {"-" * (W - 2)}')
    print(f'  {"edge type":<42} {"count":>8}  {"edge_dim":>9}')
    print(f'  {"-" * 62}')
    fwd_total = 0
    for etype in ALL_EDGE_TYPES:
        n   = g[etype].edge_index.shape[1]
        dim = g[etype].edge_attr.shape[1] if g[etype].edge_index.shape[1] > 0 else EDGE_FEATURE_DIMS.get(etype, '?')
        fwd_total += n
        print(f'  {str(etype):<42} {n:>8,}  {dim:>9}')
    print(f'  {"-" * 62}')
    print(f'  {"forward total":<42} {fwd_total:>8,}')

    print(f'\n  {"reverse edge type":<42} {"count":>8}')
    print(f'  {"-" * 52}')
    rev_total = 0
    for retype in ALL_REV_EDGE_TYPES:
        n = g[retype].edge_index.shape[1]
        rev_total += n
        print(f'  {str(retype):<42} {n:>8,}')
    print(f'  {"-" * 52}')
    print(f'  {"reverse total":<42} {rev_total:>8,}')

    # ── Label stats ──────────────────────────────────────────────────────────
    print(f'\n  LABEL STATS')
    print(f'  {"-" * (W - 2)}')
    y = g[NODE_USER].y
    n_pos  = int(y.sum().item())
    n_neg  = y.shape[0] - n_pos
    ratio  = n_pos / max(y.shape[0], 1)
    print(f'  {"total users":<28} {y.shape[0]:>8,}')
    print(f'  {"malicious (positive)":<28} {n_pos:>8,}  ({ratio:.4%})')
    print(f'  {"benign   (negative)":<28} {n_neg:>8,}')

    # ── Feature value stats (NaN/Inf summary) ────────────────────────────────
    print(f'\n  FEATURE HEALTH')
    print(f'  {"-" * (W - 2)}')
    any_bad = False
    for ntype in (NODE_USER, NODE_PC, NODE_URL, NODE_FILE):
        x = g[ntype].x
        n_nan = torch.isnan(x).sum().item()
        n_inf = torch.isinf(x).sum().item()
        status = 'OK' if n_nan == 0 and n_inf == 0 else f'NaN={n_nan} Inf={n_inf}'
        if status != 'OK':
            any_bad = True
        print(f'  node/{ntype:<10} x{str(tuple(x.shape)):<16} {status}')
    for etype in ALL_EDGE_TYPES:
        ea = g[etype].edge_attr
        if ea.shape[0] == 0:
            print(f'  edge/{str(etype):<38} (empty)')
            continue
        n_nan = torch.isnan(ea).sum().item()
        n_inf = torch.isinf(ea).sum().item()
        status = 'OK' if n_nan == 0 and n_inf == 0 else f'NaN={n_nan} Inf={n_inf}'
        if status != 'OK':
            any_bad = True
        print(f'  edge/{str(etype):<38} {status}')
    if not any_bad:
        print(f'  All feature tensors are finite.')


# ── Validation checks ─────────────────────────────────────────────────────────

def validate_graph(split: str, g, results: CheckResult) -> None:
    """Run all structural and semantic checks; record pass/fail in *results*."""
    pfx = f'[{split}]'

    # 1. Required node types present
    for ntype in (NODE_USER, NODE_PC, NODE_URL, NODE_FILE):
        results.check(
            ntype in g.node_types,
            f'{pfx} node type {ntype!r} present',
            f'{pfx} MISSING node type {ntype!r}',
        )

    # 2. Required edge types present (forward + reverse)
    for etype in ALL_EDGE_TYPES + ALL_REV_EDGE_TYPES:
        results.check(
            etype in g.edge_types,
            f'{pfx} edge type {etype[1]!r} present',
            f'{pfx} MISSING edge type {str(etype)!r}',
        )

    # 3. Feature dimension matches schema
    for ntype, expected_dim in NODE_FEATURE_DIMS.items():
        if ntype not in g.node_types:
            continue
        actual = g[ntype].x.shape[1]
        results.check(
            actual == expected_dim,
            f'{pfx} {ntype} feat_dim={actual} matches schema',
            f'{pfx} {ntype} feat_dim={actual} != schema {expected_dim}',
        )

    for etype, expected_dim in EDGE_FEATURE_DIMS.items():
        if etype not in g.edge_types:
            continue
        actual = g[etype].edge_attr.shape[1]
        results.check(
            actual == expected_dim,
            f'{pfx} edge {etype[1]!r} feat_dim={actual} matches schema',
            f'{pfx} edge {etype[1]!r} feat_dim={actual} != schema {expected_dim}',
        )

    # 4. Edge index bounds
    n_user = g[NODE_USER].x.shape[0]
    n_pc   = g[NODE_PC].x.shape[0]
    n_url  = g[NODE_URL].x.shape[0]
    n_file = g[NODE_FILE].x.shape[0]
    bound_checks = [
        (EDGE_LOGON,     n_user, n_pc),
        (EDGE_USB,       n_user, n_pc),
        (EDGE_EMAIL,     n_user, n_user),
        (EDGE_HTTP,      n_user, n_url),
        (EDGE_FILE_COPY, n_user, n_file),
    ]
    for etype, src_n, dst_n in bound_checks:
        if etype not in g.edge_types:
            continue
        ei = g[etype].edge_index
        if ei.shape[1] == 0:
            results.ok(f'{pfx} {etype[1]!r} edge_index empty (no bounds to check)')
            continue
        ok = (ei[0].min() >= 0 and ei[0].max() < src_n
              and ei[1].min() >= 0 and ei[1].max() < dst_n)
        results.check(
            ok,
            f'{pfx} {etype[1]!r} edge_index in bounds',
            f'{pfx} {etype[1]!r} edge_index OUT OF BOUNDS '
            f'(src_max={ei[0].max().item()} vs {src_n}  '
            f'dst_max={ei[1].max().item()} vs {dst_n})',
        )

    # 5. No NaN / Inf in node features
    for ntype in (NODE_USER, NODE_PC, NODE_URL, NODE_FILE):
        x = g[ntype].x
        finite = torch.isfinite(x).all().item()
        results.check(
            finite,
            f'{pfx} node/{ntype} features are all finite',
            f'{pfx} node/{ntype} has NaN/Inf in features',
        )

    # 6. No NaN / Inf in edge features
    for etype in ALL_EDGE_TYPES:
        ea = g[etype].edge_attr
        if ea.shape[0] == 0:
            results.ok(f'{pfx} edge/{etype[1]!r} feature check skipped (empty)')
            continue
        finite = torch.isfinite(ea).all().item()
        results.check(
            finite,
            f'{pfx} edge/{etype[1]!r} features are all finite',
            f'{pfx} edge/{etype[1]!r} has NaN/Inf in features',
        )

    # 7. Reverse edge counts match forward edge counts
    fwd_rev_pairs = list(zip(ALL_EDGE_TYPES, ALL_REV_EDGE_TYPES))
    for ftype, rtype in fwd_rev_pairs:
        if ftype not in g.edge_types or rtype not in g.edge_types:
            continue
        n_fwd = g[ftype].edge_index.shape[1]
        n_rev = g[rtype].edge_index.shape[1]
        results.check(
            n_fwd == n_rev,
            f'{pfx} {ftype[1]!r} fwd/rev counts match ({n_fwd})',
            f'{pfx} {ftype[1]!r} fwd={n_fwd} rev={n_rev} MISMATCH',
        )

    # 8. Label vector length matches user count
    y = g[NODE_USER].y
    results.check(
        y.shape[0] == n_user,
        f'{pfx} label vector length {y.shape[0]} matches n_users',
        f'{pfx} label vector length {y.shape[0]} != n_users {n_user}',
    )

    # 9. Label dtype is float32
    results.check(
        y.dtype == torch.float32,
        f'{pfx} label dtype is float32',
        f'{pfx} label dtype {y.dtype} is not float32',
    )

    # 10. Plausible user count (warn, not fail, for synthetic data)
    results.check(
        n_user >= 10,
        f'{pfx} n_users={n_user} >= 10',
        f'{pfx} n_users={n_user} seems too small (< 10)',
        warn=True,
    )
    if n_user < EXPECTED_USERS:
        results.warn(
            f'{pfx} n_users={n_user} < expected {EXPECTED_USERS} '
            f'(real CERT r4.2 has ~1000 users; this may be synthetic data)',
        )

    # 11. Plausible malicious count
    n_pos = int(y.sum().item())
    results.check(
        0 < n_pos <= EXPECTED_POS_MAX,
        f'{pfx} n_positives={n_pos} in plausible range [1, {EXPECTED_POS_MAX}]',
        f'{pfx} n_positives={n_pos} outside plausible range',
        warn=True,
    )


# ── CSV cross-check ───────────────────────────────────────────────────────────

def crosscheck_csv_counts(graphs: dict, data_dir: str | None) -> None:
    """Compare graph edge counts against raw CSV row counts."""
    print(f'\n{"=" * W}')
    print('  CSV CROSS-CHECK')
    print(f'{"=" * W}')

    if not data_dir or not os.path.exists(data_dir):
        print(f'  (skipped — data_dir not set or not found: {data_dir!r})')
        return

    import pandas as pd

    csv_counts: dict[str, int] = {}
    for name in ('logon', 'device', 'file', 'email', 'http'):
        path = os.path.join(data_dir, f'{name}.csv')
        if os.path.exists(path):
            csv_counts[name] = sum(1 for _ in open(path)) - 1  # subtract header
        else:
            csv_counts[name] = -1

    print(f'  {"CSV file":<14} {"raw rows":>10}')
    print(f'  {"-" * 26}')
    for name, n in csv_counts.items():
        s = f'{n:,}' if n >= 0 else 'not found'
        print(f'  {name:<14} {s:>10}')

    print(f'\n  {"split":<8}  {"logon_edges":>12}  {"usb_edges":>10}  '
          f'{"http_edges":>10}  {"file_edges":>10}  {"email_edges":>11}')
    print(f'  {"-" * 66}')
    for split, g in graphs.items():
        logon = g[EDGE_LOGON].edge_index.shape[1]
        usb   = g[EDGE_USB].edge_index.shape[1]
        http  = g[EDGE_HTTP].edge_index.shape[1]
        fcp   = g[EDGE_FILE_COPY].edge_index.shape[1]
        eml   = g[EDGE_EMAIL].edge_index.shape[1]
        print(f'  {split:<8}  {logon:>12,}  {usb:>10,}  '
              f'{http:>10,}  {fcp:>10,}  {eml:>11,}')

    print(f'\n  Notes:')
    print(f'    logon/usb/http/file edges  ≤ CSV rows (time-window filtering)')
    print(f'    email edges may differ: one email → one edge per internal recipient')


# ── Neighbourhood visualization ───────────────────────────────────────────────

def _node_type_offsets(g) -> dict[str, int]:
    """Return the global node-index offset for each node type (as in to_networkx)."""
    offsets: dict[str, int] = {}
    cum = 0
    for ntype in g.node_types:          # PyG preserves insertion order
        offsets[ntype] = cum
        cum += g[ntype].x.shape[0]
    return offsets


def visualize_neighborhood(g, user_idx: int, user_name: str, out_path: str) -> None:
    """Plot the 1-hop neighbourhood of *user_idx* and save as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError as exc:
        print(f'  Visualization skipped ({exc})')
        return

    offsets = _node_type_offsets(g)

    # ── Collect 1-hop neighbourhood ──────────────────────────────────────────
    NX = nx.MultiDiGraph()

    # Node colours and labels.
    type_colour = {
        NODE_USER: '#4878CF',   # blue
        NODE_PC:   '#6ACC65',   # green
        NODE_URL:  '#EE854A',   # orange
        NODE_FILE: '#D65F5F',   # red
    }
    malicious_colour = '#B22222'  # firebrick for the target user

    def _add_node(ntype: str, local_idx: int, is_target: bool = False) -> int:
        gid = offsets[ntype] + local_idx
        if not NX.has_node(gid):
            colour = malicious_colour if is_target else type_colour[ntype]
            label  = (
                g.idx_to_user.get(local_idx, f'{ntype}[{local_idx}]')
                if ntype == NODE_USER
                else f'{ntype}[{local_idx}]'
            )
            NX.add_node(gid, ntype=ntype, colour=colour, label=label)
        return gid

    # Add target user.
    target_gid = _add_node(NODE_USER, user_idx, is_target=True)

    # Edge type → display colour.
    rel_colour = {
        'logon':         '#4878CF',
        'usb_connect':   '#6ACC65',
        'email_to':      '#EE854A',
        'http_request':  '#D65F5F',
        'file_copy':     '#956CB4',
        'rev_logon':         '#4878CF',
        'rev_usb_connect':   '#6ACC65',
        'rev_email_to':      '#EE854A',
        'rev_http_request':  '#D65F5F',
        'rev_file_copy':     '#956CB4',
    }

    edge_count = 0
    for etype in g.edge_types:
        src_type, rel, dst_type = etype
        ei = g[etype].edge_index

        # Forward edges from target user.
        if src_type == NODE_USER:
            mask = ei[0] == user_idx
            for j in ei[1, mask].tolist():
                s_gid = target_gid
                d_gid = _add_node(dst_type, j)
                NX.add_edge(s_gid, d_gid, rel=rel, colour=rel_colour.get(rel, '#888'))
                edge_count += 1

        # Reverse edges pointing TO target user.
        if dst_type == NODE_USER:
            mask = ei[1] == user_idx
            for j in ei[0, mask].tolist():
                s_gid = _add_node(src_type, j)
                d_gid = target_gid
                NX.add_edge(s_gid, d_gid, rel=rel, colour=rel_colour.get(rel, '#888'))
                edge_count += 1

    n_neigh = NX.number_of_nodes() - 1
    print(f'\n  Neighbourhood of {user_name!r} (user_idx={user_idx}):')
    print(f'    Neighbour nodes  : {n_neigh}')
    print(f'    Neighbourhood edges : {edge_count}')
    for ntype in (NODE_PC, NODE_URL, NODE_FILE, NODE_USER):
        cnt = sum(1 for _, d in NX.nodes(data=True) if d.get('ntype') == ntype)
        print(f'    {ntype:<8}: {cnt}')

    if NX.number_of_nodes() == 0:
        print('    (no edges found — skipping plot)')
        return

    # ── Layout & plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(
        f'1-hop neighbourhood of malicious user {user_name!r}\n'
        f'({NX.number_of_nodes()} nodes, {edge_count} edges)',
        fontsize=13,
    )
    ax.axis('off')

    # Spring layout with target user pinned to centre.
    pos = nx.spring_layout(NX, seed=42, k=2.5)
    pos[target_gid] = (0.0, 0.0)

    node_colours = [d['colour'] for _, d in NX.nodes(data=True)]
    node_sizes   = [
        600 if n == target_gid else 250
        for n in NX.nodes()
    ]
    labels = {n: d['label'] for n, d in NX.nodes(data=True)}

    # Draw nodes.
    nx.draw_networkx_nodes(NX, pos, ax=ax,
                           node_color=node_colours, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(NX, pos, labels=labels, ax=ax, font_size=6)

    # Draw edges grouped by relation (so each colour is applied together).
    relations = set(d['rel'] for _, _, d in NX.edges(data=True))
    for rel in relations:
        elist = [(u, v) for u, v, d in NX.edges(data=True) if d['rel'] == rel]
        colour = rel_colour.get(rel, '#888')
        nx.draw_networkx_edges(NX, pos, edgelist=elist, ax=ax,
                               edge_color=colour, alpha=0.55,
                               arrows=True, arrowsize=10,
                               connectionstyle='arc3,rad=0.1')

    # Legend.
    legend_items = [
        mpatches.Patch(color=type_colour[NODE_USER],  label='user (benign)'),
        mpatches.Patch(color=malicious_colour,          label='user (malicious)'),
        mpatches.Patch(color=type_colour[NODE_PC],    label='pc'),
        mpatches.Patch(color=type_colour[NODE_URL],   label='url'),
        mpatches.Patch(color=type_colour[NODE_FILE],  label='file'),
    ]
    for rel, col in rel_colour.items():
        if not rel.startswith('rev_'):
            legend_items.append(
                mpatches.Patch(color=col, label=f'edge: {rel}', alpha=0.7)
            )
    ax.legend(handles=legend_items, loc='upper right', fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Plot saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description='Validate InnerSight temporal graphs.')
    ap.add_argument('--model-dir', default=os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
                    help='Directory containing cached graph .pt files')
    ap.add_argument('--data-dir',  default=os.environ.get('INNERSIGHT_DATA_DIR', ''),
                    help='Raw CERT CSV directory (for cross-check; optional)')
    ap.add_argument('--out-dir',   default=os.path.join(_BACKEND, 'outputs'),
                    help='Directory for output files (PNG, etc.)')
    ap.add_argument('--no-viz',    action='store_true',
                    help='Skip neighbourhood visualization')
    args = ap.parse_args()

    os.environ['INNERSIGHT_MODEL_DIR'] = args.model_dir
    if args.data_dir:
        os.environ['INNERSIGHT_DATA_DIR'] = args.data_dir

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    from innersight.backend.models.graph_builder import load_temporal_graphs

    print(f'\n{"=" * W}')
    print('  InnerSight Graph Validation')
    print(f'  model_dir : {args.model_dir}')
    print(f'  data_dir  : {args.data_dir or "(not set)"}')
    print(f'{"=" * W}')

    print('\nLoading temporal graphs...')
    graphs = load_temporal_graphs()
    print(f'  Loaded: {list(graphs.keys())}')

    results = CheckResult()

    # ── Per-graph stats + validation ─────────────────────────────────────────
    for split, g in graphs.items():
        print_graph_stats(split, g)
        validate_graph(split, g, results)

    # ── CSV cross-check ───────────────────────────────────────────────────────
    crosscheck_csv_counts(graphs, args.data_dir or None)

    # ── Neighbourhood visualization ───────────────────────────────────────────
    if not args.no_viz:
        print(f'\n{"=" * W}')
        print('  NEIGHBOURHOOD VISUALIZATION')
        print(f'{"=" * W}')
        # Use the train graph; pick the first malicious user.
        g_train = graphs['train']
        y       = g_train[NODE_USER].y
        pos_idx = (y == 1).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            print('  No malicious users in train graph — skipping visualization.')
        else:
            target_idx  = pos_idx[0].item()
            target_name = g_train.idx_to_user.get(target_idx, f'user[{target_idx}]')
            out_path    = os.path.join(args.out_dir, 'sample_subgraph.png')
            visualize_neighborhood(g_train, target_idx, target_name, out_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    total   = len(results.passed) + len(results.failed)
    n_pass  = len(results.passed)
    n_fail  = len(results.failed)
    n_warn  = len(results.warned)

    print(f'\n{"=" * W}')
    if n_fail == 0:
        print(f'  VALIDATION PASSED: {n_pass}/{total} checks passed', end='')
        print(f'  ({n_warn} warning(s))' if n_warn else '')
    else:
        print(f'  VALIDATION FAILED: {n_pass}/{total} checks passed  '
              f'({n_fail} failure(s), {n_warn} warning(s))')
        print(f'\n  Failures:')
        for msg in results.failed:
            print(f'    FAIL  {msg}')

    if n_warn:
        print(f'\n  Warnings:')
        for msg in results.warned:
            print(f'    WARN  {msg}')

    print(f'{"=" * W}')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
