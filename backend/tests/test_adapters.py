"""Unit tests for backend/data/adapters.py (synthetic CERT directories)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from innersight.data.adapters import (
    R1Adapter,
    R4xAdapter,
    auto_detect_version,
    get_adapter,
)


@pytest.fixture()
def r4_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'r4.2'
    d.mkdir()
    (d / 'logon.csv').write_text(
        'id,date,user,pc,activity\n{L},01/02/2010 06:49:00,NGF0157,PC-6056,Logon\n'
    )
    (d / 'device.csv').write_text(
        'id,date,user,pc,activity\n{D},01/02/2010 07:00:00,NGF0157,PC-6056,Connect\n'
    )
    (d / 'http.csv').write_text(
        'id,date,user,pc,url,content\n{H},01/02/2010 06:55:16,NGF0157,PC-6056,http://msn.com,news\n'
    )
    (d / 'email.csv').write_text(
        'id,date,user,pc,to,cc,bcc,from,size,attachments,content\n'
        '{E},01/02/2010 07:11:45,NGF0157,PC-6056,a@dtaa.com,,,NGF0157@dtaa.com,2500,0,hello\n'
    )
    (d / 'file.csv').write_text(
        'id,date,user,pc,filename,content\n{F},01/02/2010 07:23:14,NGF0157,PC-6056,a.doc,D0-CF data\n'
    )
    return d


@pytest.fixture()
def r1_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'r1'
    d.mkdir()
    (d / 'logon.csv').write_text(
        'id,date,user,pc,activity\n{L},01/02/2010 06:49:00,DTAA/KEE0997,PC-6056,Logon\n'
    )
    # r1 http.csv has NO header row, 5 columns, DTAA/-prefixed user
    (d / 'http.csv').write_text('{H},01/02/2010 06:55:16,DTAA/KEE0997,PC-6056,http://x.com\n')
    return d


def test_get_adapter_r4_returns_r4x() -> None:
    assert isinstance(get_adapter('r4.2'), R4xAdapter)


def test_get_adapter_r1_returns_r1() -> None:
    assert isinstance(get_adapter('r1'), R1Adapter)


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_adapter('r99')


def test_r1_strips_dtaa_prefix(r1_dir: Path) -> None:
    adapter = get_adapter('r1')
    logon = adapter.load_logon(r1_dir)
    assert logon.iloc[0]['user'] == 'KEE0997'
    http = adapter.load_http(r1_dir)
    assert http.iloc[0]['user'] == 'KEE0997'
    assert 'content' in http.columns  # r1 http gains a content column (None)


def test_r2_normalizes_device_activity(tmp_path: Path) -> None:
    d = tmp_path / 'r2'
    d.mkdir()
    (d / 'device.csv').write_text(
        'id,date,user,pc,activity\n'
        '{D1},01/02/2010 07:00:00,U1,PC-1,Insert\n'
        '{D2},01/02/2010 08:00:00,U1,PC-1,Remove\n'
    )
    df = get_adapter('r2').load_device(d)
    assert set(df['activity']) == {'Connect', 'Disconnect'}


def test_r4_loads_all_log_types(r4_dir: Path) -> None:
    adapter = get_adapter('r4.2')
    assert len(adapter.load_logon(r4_dir)) == 1
    assert len(adapter.load_device(r4_dir)) == 1
    assert len(adapter.load_email(r4_dir)) == 1
    assert len(adapter.load_http(r4_dir)) == 1
    assert len(adapter.load_file(r4_dir)) == 1


def test_adapter_missing_file_returns_empty(tmp_path: Path) -> None:
    d = tmp_path / 'empty'
    d.mkdir()
    df = get_adapter('r4.2').load_email(d)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_auto_detect_identifies_r4x(r4_dir: Path) -> None:
    assert auto_detect_version(r4_dir) == 'r4x'
