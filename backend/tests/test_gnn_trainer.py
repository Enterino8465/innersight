"""Tests for the GNN training loop (models/gnn_trainer.py)."""

import copy
import importlib
import os
import pytest
import torch

# NeighborLoader requires pyg-lib or torch-sparse; skip the whole module if absent.
_pyg_sparse = importlib.util.find_spec('pyg_lib') or importlib.util.find_spec('torch_sparse')
pytestmark = pytest.mark.skipif(
    _pyg_sparse is None,
    reason='pyg-lib or torch-sparse required for NeighborLoader (GNN tests)',
)

from innersight.backend.models.gnn_trainer import train_gnn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_config(model_path: str) -> dict:
    """Smallest valid config for a 2-epoch GNN smoke test."""
    return {
        'model': {
            'type':        'graphsage',
            'hidden_dim':  16,
            'num_layers':  1,
            'dropout':     0.0,
            'head_layers': [8],
        },
        'training': {
            'epochs':        2,
            'batch_size':    4,
            'lr':            0.001,
            'pos_weight':    10.0,
            'patience':      10,
            'num_neighbors': [2],   # 1-hop only, 2 neighbours — keeps test fast
        },
        'output': {
            'model_path': model_path,
            'log_file':   '/dev/null',
        },
    }


def _split_graph(g):
    """Return three shallow copies of g for train/val/test."""
    return {'train': copy.deepcopy(g), 'val': copy.deepcopy(g), 'test': copy.deepcopy(g)}


# ── Smoke test ────────────────────────────────────────────────────────────────

def test_train_gnn_smoke(tmp_path, small_hetero_graph):
    """train_gnn completes 2 epochs, emits events, and saves a checkpoint."""
    model_path = str(tmp_path / 'gnn_model.pt')
    config     = _minimal_config(model_path)
    graphs     = _split_graph(small_hetero_graph)

    events = []
    result = train_gnn(config, event_callback=events.append, graphs=graphs)

    # Checkpoint written
    assert os.path.exists(model_path), 'Model checkpoint not saved'

    # Result keys
    for key in ('best_val_f1', 'test_loss', 'test_precision', 'test_recall', 'test_f1'):
        assert key in result, f'Missing result key: {key}'

    # Losses are valid floats
    assert isinstance(result['test_loss'], float)
    assert not (result['test_loss'] != result['test_loss'])   # not NaN


def test_train_gnn_events(tmp_path, small_hetero_graph):
    """Event stream contains class_imbalance, epoch, and done events."""
    model_path = str(tmp_path / 'gnn.pt')
    config     = _minimal_config(model_path)
    graphs     = _split_graph(small_hetero_graph)

    events = []
    train_gnn(config, event_callback=events.append, graphs=graphs)

    assert any('class_imbalance' in e for e in events), 'No class_imbalance event'
    assert any('epoch' in e and 'val_loss' in e for e in events), 'No epoch summary event'
    assert any(e.get('status') == 'done' for e in events), 'No done event'


def test_train_gnn_class_imbalance_ratio(tmp_path, small_hetero_graph):
    """class_imbalance ratio reflects the 2/10 positive rate in the fixture."""
    model_path = str(tmp_path / 'gnn.pt')
    config     = _minimal_config(model_path)
    graphs     = _split_graph(small_hetero_graph)

    events = []
    train_gnn(config, event_callback=events.append, graphs=graphs)

    ci_event = next(e for e in events if 'class_imbalance' in e)
    ratio    = ci_event['class_imbalance']['ratio']
    assert abs(ratio - 0.2) < 1e-4, f'Expected ~0.2, got {ratio}'


def test_train_gnn_checkpoint_loadable(tmp_path, small_hetero_graph):
    """Saved checkpoint can be re-loaded as a valid InsiderThreatGNN."""
    from innersight.backend.models.graphsage import InsiderThreatGNN

    model_path = str(tmp_path / 'gnn.pt')
    config     = _minimal_config(model_path)
    graphs     = _split_graph(small_hetero_graph)

    train_gnn(config, graphs=graphs)

    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    assert ck.get('model_type') == 'graphsage'

    model = InsiderThreatGNN(
        metadata=ck['metadata'],
        **{k: ck['config'][k]
           for k in ('hidden_dim', 'num_layers', 'dropout', 'head_layers')},
    )
    model.load_state_dict(ck['state_dict'])
    model.eval()

    g  = small_hetero_graph
    with torch.no_grad():
        logits = model(g.x_dict, g.edge_index_dict)
    assert logits.shape == (10, 1)


def test_train_gnn_no_data_dir_needed(tmp_path, small_hetero_graph):
    """When graphs= is supplied, train_gnn never touches the filesystem for data."""
    model_path = str(tmp_path / 'gnn.pt')
    config     = _minimal_config(model_path)
    # Intentionally point data.dir at a non-existent path — must be ignored
    config['data'] = {'dir': '/nonexistent/path/that/does/not/exist'}

    graphs = _split_graph(small_hetero_graph)
    # Should not raise even though data dir is wrong
    train_gnn(config, graphs=graphs)
    assert os.path.exists(model_path)


def test_train_gnn_batch_loss_events(tmp_path, small_hetero_graph):
    """Per-batch loss events are emitted with epoch/batch/loss keys."""
    model_path = str(tmp_path / 'gnn.pt')
    config     = _minimal_config(model_path)
    graphs     = _split_graph(small_hetero_graph)

    events = []
    train_gnn(config, event_callback=events.append, graphs=graphs)

    batch_events = [e for e in events if 'batch' in e and 'loss' in e and 'val_loss' not in e]
    assert len(batch_events) > 0, 'No per-batch loss events emitted'
    for e in batch_events:
        assert isinstance(e['loss'], float), 'Batch loss must be float'
        assert e['loss'] >= 0.0,              'Batch loss must be non-negative'
