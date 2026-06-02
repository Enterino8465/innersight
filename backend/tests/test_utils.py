import json
import os
import pytest
from innersight.utils import safe_json_read, safe_json_write


def test_write_creates_file(tmp_path):
    path = str(tmp_path / 'out.json')
    safe_json_write(path, [1, 2, 3])
    assert os.path.exists(path)


def test_write_no_tmp_leftover(tmp_path):
    path = str(tmp_path / 'out.json')
    safe_json_write(path, {'key': 'val'})
    assert not os.path.exists(path + '.tmp')


def test_write_creates_parent_dirs(tmp_path):
    path = str(tmp_path / 'nested' / 'deep' / 'out.json')
    safe_json_write(path, [])
    assert os.path.exists(path)


def test_round_trip(tmp_path):
    path = str(tmp_path / 'data.json')
    original = [{'id': 1, 'score': 0.9}, {'id': 2, 'score': 0.5}]
    safe_json_write(path, original)
    loaded = safe_json_read(path)
    assert loaded == original


def test_read_missing_file_returns_default(tmp_path):
    path = str(tmp_path / 'nonexistent.json')
    assert safe_json_read(path) is None
    assert safe_json_read(path, default=[]) == []
    assert safe_json_read(path, default={'x': 1}) == {'x': 1}


def test_read_corrupt_file_returns_default(tmp_path):
    path = str(tmp_path / 'bad.json')
    with open(path, 'w') as f:
        f.write('{not: valid json}}}')
    result = safe_json_read(path, default='fallback')
    assert result == 'fallback'


def test_write_is_atomic_via_replace(tmp_path, monkeypatch):
    """Verify os.replace is called (atomic swap), not a direct open+write."""
    path = str(tmp_path / 'atomic.json')
    replaced = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replaced.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', spy_replace)
    safe_json_write(path, [42])
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == path
    assert src == path + '.tmp'


def test_overwrite_preserves_data(tmp_path):
    path = str(tmp_path / 'over.json')
    safe_json_write(path, ['first'])
    safe_json_write(path, ['second'])
    assert safe_json_read(path) == ['second']
