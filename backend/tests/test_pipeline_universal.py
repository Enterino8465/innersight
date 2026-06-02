"""Unit tests for backend/data/pipeline.py universal orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from innersight.backend.data.pipeline import CertDataset, load_version, time_split


@pytest.fixture()
def r4_dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'r4.2'
    d.mkdir()
    (d / 'logon.csv').write_text(
        'id,date,user,pc,activity\n{L},01/02/2010 06:49:00,AAM0658,PC-1,Logon\n'
    )
    (d / 'device.csv').write_text(
        'id,date,user,pc,activity\n{D},01/02/2010 07:00:00,AAM0658,PC-1,Connect\n'
    )
    (d / 'http.csv').write_text(
        'id,date,user,pc,url,content\n{H},01/02/2010 06:55:16,AAM0658,PC-1,http://x,news\n'
    )
    (d / 'email.csv').write_text(
        'id,date,user,pc,to,cc,bcc,from,size,attachments,content\n'
        '{E},01/02/2010 07:11:45,AAM0658,PC-1,a@dtaa.com,,,AAM0658@dtaa.com,2500,0,hi\n'
    )
    (d / 'file.csv').write_text(
        'id,date,user,pc,filename,content\n{F},01/02/2010 07:23:14,AAM0658,PC-1,a.doc,D0-CF data\n'
    )
    ldap = d / 'LDAP'
    ldap.mkdir()
    (ldap / '2010-01.csv').write_text(
        'employee_name,user_id,email,role\nA M Person,AAM0658,a@dtaa.com,Eng\n'
    )
    (d / 'psychometric.csv').write_text(
        'employee_name,user_id,O,C,E,A,N\nA M Person,AAM0658,40,39,36,19,40\n'
    )
    ans = d / 'answers'
    ans.mkdir()
    (ans / 'insiders.csv').write_text(
        'dataset,scenario,details,user,start,end\n'
        '4.2,1,f.csv,AAM0658,1/2/2010 6:00:00,1/3/2010 18:00:00\n'
    )
    return d


def test_load_version_all_fields_populated(r4_dataset_dir: Path) -> None:
    ds = load_version(r4_dataset_dir, 'r4.2')
    assert isinstance(ds, CertDataset)
    assert ds.version == 'r4.2'
    assert ds.data_dir == r4_dataset_dir
    assert ds.logs
    assert not ds.ldap.empty
    assert not ds.psychometric.empty


def test_logs_dict_has_expected_keys(r4_dataset_dir: Path) -> None:
    ds = load_version(r4_dataset_dir, 'r4.2')
    assert set(ds.logs.keys()) == {'logon', 'device', 'email', 'http', 'file'}


def test_insiders_count(r4_dataset_dir: Path) -> None:
    ds = load_version(r4_dataset_dir, 'r4.2')
    assert len(ds.insiders) == 1


def test_attack_windows_keyed_by_user(r4_dataset_dir: Path) -> None:
    ds = load_version(r4_dataset_dir, 'r4.2')
    assert set(ds.attack_windows.keys()) == {'AAM0658'}


def test_time_split_preserves_columns(r4_dataset_dir: Path) -> None:
    ds = load_version(r4_dataset_dir, 'r4.2')
    splits = time_split(ds.logs, train_end='2010-06-30', val_end='2010-09-30')
    assert set(splits.keys()) == {'train', 'val', 'test'}
    for split in splits.values():
        assert list(split['logon'].columns) == list(ds.logs['logon'].columns)
