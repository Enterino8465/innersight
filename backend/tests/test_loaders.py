"""Unit tests for backend/data/loaders.py (no real CERT data required)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from innersight.data.loaders import (
    load_csv,
    load_decoy_files,
    load_ldap_directory,
    load_psychometric,
)


@pytest.fixture()
def logon_csv(tmp_path: Path) -> Path:
    path = tmp_path / 'logon.csv'
    path.write_text(
        'id,date,user,pc,activity\n'
        '{L3},01/03/2010 09:00:00,U1,PC-1,Logon\n'
        '{L1},01/01/2010 08:00:00,U1,PC-1,Logon\n'
        '{L2},01/02/2010 07:00:00,U2,PC-2,Logoff\n'
    )
    return path


def test_load_csv_parses_and_sorts(logon_csv: Path) -> None:
    df = load_csv(logon_csv)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert list(df['date']) == sorted(df['date'])  # ascending by date
    assert df.iloc[0]['id'] == '{L1}'  # earliest date first


def test_load_csv_no_header_with_columns(tmp_path: Path) -> None:
    path = tmp_path / 'http.csv'
    path.write_text('{H1},01/01/2010 08:00:00,U1,PC-1,http://x.com\n')
    cols = ['id', 'date', 'user', 'pc', 'url']
    df = load_csv(path, columns=cols, has_header=False)
    assert list(df.columns) == cols
    assert df.iloc[0]['url'] == 'http://x.com'


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / 'nope.csv')


def test_load_csv_no_header_without_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / 'x.csv'
    path.write_text('a,b\n1,2\n')
    with pytest.raises(ValueError, match='columns'):
        load_csv(path, has_header=False)


def test_load_ldap_directory_latest_snapshot(tmp_path: Path) -> None:
    ldap = tmp_path / 'LDAP'
    ldap.mkdir()
    (ldap / '2009-12.csv').write_text(
        'employee_name,user_id,email,role\nOld Name,U1,u1@x,Eng\n'
    )
    (ldap / '2010-01.csv').write_text(
        'employee_name,user_id,email,role\n'
        'New Name,U1,u1@x,Eng\nNew Two,U2,u2@x,Mgr\n'
    )
    df = load_ldap_directory(ldap)
    assert len(df) == 2  # latest (2010-01) snapshot, not the older one
    assert 'New Name' in set(df['employee_name'])


def test_load_ldap_directory_missing_returns_empty(tmp_path: Path) -> None:
    df = load_ldap_directory(tmp_path / 'no_such_ldap')
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_psychometric(tmp_path: Path) -> None:
    path = tmp_path / 'psychometric.csv'
    path.write_text('employee_name,user_id,O,C,E,A,N\nAlice,U1,40,39,36,19,40\n')
    df = load_psychometric(path)
    assert len(df) == 1
    assert df.iloc[0]['O'] == 40
    assert df.iloc[0]['N'] == 40


def test_load_decoy_files(tmp_path: Path) -> None:
    path = tmp_path / 'decoy_file.csv'
    path.write_text('decoy_filename,pc\ndecoy1.doc,PC-1\ndecoy2.xls,PC-2\n')
    df = load_decoy_files(path)
    assert len(df) == 2
    assert list(df.columns) == ['decoy_filename', 'pc']
