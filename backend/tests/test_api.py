import json
import uuid
import pytest

from innersight.backend.config import ALERTS_FILE, DEFAULT_TRAINING_CONFIG


@pytest.fixture(autouse=True)
def seed_alerts(tmp_path, monkeypatch):
    """Point ALERTS_FILE at a tmp copy so tests don't touch real data."""
    import innersight.backend.scoring.scoring as sc
    import innersight.backend.feedback.feedback as fb

    alerts_path = str(tmp_path / 'alerts.json')
    sample = [
        {'id': str(uuid.uuid4()), 'user': 'u1', 'date': '2011-03-15',
         'score': 0.92, 'status': 'open', 'created_at': '', 'top_features': []},
    ]
    with open(alerts_path, 'w') as f:
        json.dump(sample, f)

    monkeypatch.setattr(sc, '_ALERTS_PATH', alerts_path)
    monkeypatch.setattr(fb, '_ALERTS_PATH', alerts_path)
    return alerts_path


@pytest.fixture()
def client():
    from innersight.backend import api as api_mod
    api_mod.app.config['TESTING'] = True
    with api_mod.app.test_client() as c:
        yield c


def test_status_returns_200(client):
    r = client.get('/api/status')
    assert r.status_code == 200
    assert r.get_json() == {'status': 'idle'}


def test_config_returns_layer_sizes(client):
    r = client.get('/api/config')
    assert r.status_code == 200
    body = r.get_json()
    assert 'layer_sizes' in body
    assert body['layer_sizes'] == DEFAULT_TRAINING_CONFIG['layer_sizes']


def test_train_empty_body_uses_defaults(client):
    # Empty body → defaults applied → valid → 200
    r = client.post('/api/train', json={})
    assert r.status_code == 200
    assert r.get_json()['status'] == 'started'


def test_train_invalid_epochs_returns_400(client):
    bad = dict(DEFAULT_TRAINING_CONFIG, epochs=0)
    r = client.post('/api/train', json=bad)
    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_train_invalid_layer_sizes_returns_400(client):
    bad = dict(DEFAULT_TRAINING_CONFIG, layer_sizes=[18, 2])
    r = client.post('/api/train', json=bad)
    assert r.status_code == 400


def test_train_invalid_lr_returns_400(client):
    bad = dict(DEFAULT_TRAINING_CONFIG, lr=-0.001)
    r = client.post('/api/train', json=bad)
    assert r.status_code == 400


def test_alerts_returns_list(client):
    r = client.get('/api/alerts')
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)


def test_alerts_status_filter(client):
    r = client.get('/api/alerts?status=open')
    assert r.status_code == 200
    body = r.get_json()
    assert all(a['status'] == 'open' for a in body)


def test_alerts_sorted_by_score_desc(client, seed_alerts):
    # Add a second alert with lower score
    import innersight.backend.scoring.scoring as sc
    alerts = sc._read_alerts_file()
    alerts.append({'id': str(uuid.uuid4()), 'user': 'u2', 'date': '2011-03-16',
                   'score': 0.5, 'status': 'open', 'created_at': '', 'top_features': []})
    sc._write_alerts_file(alerts)

    r = client.get('/api/alerts')
    body = r.get_json()
    scores = [a['score'] for a in body]
    assert scores == sorted(scores, reverse=True)


def test_alert_mute_unknown_id_returns_404(client):
    r = client.post(f'/api/alert/{uuid.uuid4()}/mute')
    assert r.status_code == 404
    assert r.get_json()['error'] == 'Alert not found'


def test_alert_block_unknown_id_returns_404(client):
    r = client.post(f'/api/alert/{uuid.uuid4()}/block')
    assert r.status_code == 404


def test_alert_mute_known_id_returns_200(client, seed_alerts):
    import innersight.backend.scoring.scoring as sc
    alert_id = sc._read_alerts_file()[0]['id']
    r = client.post(f'/api/alert/{alert_id}/mute')
    assert r.status_code == 200
    assert r.get_json()['alert']['status'] == 'muted'
