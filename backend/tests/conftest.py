import json
import os
import pytest
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import HeteroData


@pytest.fixture()
def tmp_dir(tmp_path):
    """Temporary directory that pytest cleans up automatically."""
    return tmp_path


@pytest.fixture()
def logon_csv(tmp_path):
    """Tiny logon.csv (5 rows) matching CERT schema."""
    df = pd.DataFrame({
        'id':       [f'L{i}' for i in range(5)],
        'date':     pd.to_datetime(['2010-06-01', '2010-06-01', '2010-07-15',
                                    '2010-10-10', '2011-01-20']),
        'user':     ['u1', 'u2', 'u1', 'u3', 'u1'],
        'pc':       ['PC1', 'PC2', 'PC1', 'PC3', 'PC1'],
        'activity': ['Logon', 'Logon', 'Logoff', 'Logon', 'Logon'],
    })
    path = tmp_path / 'logon.csv'
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def small_mlp():
    """Tiny InsiderThreatMLP for fast tests."""
    from innersight.models.mlp import InsiderThreatMLP
    return InsiderThreatMLP([18, 8, 1])


@pytest.fixture()
def tmp_model_checkpoint(tmp_path):
    """Save a fresh model + fitted standardizer to tmp_path.

    Returns a dict with 'pt_path', 'std_path' so tests can patch
    module-level path constants.
    """
    from innersight.models.mlp import InsiderThreatMLP
    from innersight.models.dataset import Standardizer

    layer_sizes = [18, 8, 1]
    model = InsiderThreatMLP(layer_sizes)

    pt_path  = str(tmp_path / 'best_model.pt')
    std_path = str(tmp_path / 'standardizer.pt')

    torch.save({"state_dict": model.state_dict(), "layer_sizes": layer_sizes,
                "model_type": "mlp"}, pt_path)

    std = Standardizer()
    std.fit(torch.randn(100, 18))
    std.save(std_path)

    return {'pt_path': pt_path, 'std_path': std_path, 'layer_sizes': layer_sizes}


@pytest.fixture()
def synthetic_loaders():
    """Lightweight DataLoaders + Standardizer suitable for train() smoke tests."""
    from innersight.models.dataset import Standardizer

    N_FEAT = 18
    gen    = torch.Generator().manual_seed(0)

    def _make(n, shuffle):
        X = torch.randn(n, N_FEAT, generator=gen)
        y = (torch.rand(n, generator=gen) > 0.95).float().unsqueeze(1)
        return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=shuffle, drop_last=False)

    std = Standardizer()
    std.fit(torch.randn(80, N_FEAT, generator=gen))

    return {
        'train_loader': _make(40, shuffle=True),
        'val_loader':   _make(16, shuffle=False),
        'test_loader':  _make(16, shuffle=False),
        'standardizer': std,
    }


@pytest.fixture()
def small_hetero_graph():
    """Tiny HeteroData for GNN tests: 10 users, 5 PCs, 3 URLs, no file nodes.

    Feature dimensions match the production schema from graph_schema.py:
      user=16, pc=8, url=8

    Edge types:
      (user, logon, pc) / (pc, rev_logon, user) — 20 edges
      (user, email_to, user) / (user, rev_email_to, user) — 5 edges
      (user, http_request, url) / (url, rev_http_request, user) — 10 edges

    Labels: user nodes 0 and 1 are malicious (y=1), rest benign (y=0).
    """
    rng = torch.Generator().manual_seed(42)

    N_USER, N_PC, N_URL = 10, 5, 3
    U_DIM, PC_DIM, URL_DIM = 16, 8, 8

    g = HeteroData()

    # ── Node features ──────────────────────────────────────────────────────────
    g['user'].x = torch.randn(N_USER, U_DIM,   generator=rng)
    g['pc'].x   = torch.randn(N_PC,   PC_DIM,  generator=rng)
    g['url'].x  = torch.randn(N_URL,  URL_DIM, generator=rng)

    # ── Labels (2 malicious users) ─────────────────────────────────────────────
    y = torch.zeros(N_USER)
    y[0] = 1.0
    y[1] = 1.0
    g['user'].y = y

    # ── Logon edges (user → pc) ────────────────────────────────────────────────
    src_l = torch.randint(0, N_USER, (20,), generator=rng)
    dst_l = torch.randint(0, N_PC,   (20,), generator=rng)
    g['user', 'logon',     'pc'  ].edge_index = torch.stack([src_l, dst_l])
    g['pc',   'rev_logon', 'user'].edge_index = torch.stack([dst_l, src_l])

    # ── Email edges (user → user) ──────────────────────────────────────────────
    src_e = torch.randint(0, N_USER, (5,), generator=rng)
    dst_e = torch.randint(0, N_USER, (5,), generator=rng)
    g['user', 'email_to',     'user'].edge_index = torch.stack([src_e, dst_e])
    g['user', 'rev_email_to', 'user'].edge_index = torch.stack([dst_e, src_e])

    # ── HTTP edges (user → url) ────────────────────────────────────────────────
    src_h = torch.randint(0, N_USER, (10,), generator=rng)
    dst_h = torch.randint(0, N_URL,  (10,), generator=rng)
    g['user', 'http_request',     'url' ].edge_index = torch.stack([src_h, dst_h])
    g['url',  'rev_http_request', 'user'].edge_index = torch.stack([dst_h, src_h])

    return g
