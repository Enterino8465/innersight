"""Unit tests for backend/data/answers.py insider-label parsing."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from innersight.backend.data.answers import (
    InsiderRecord,
    get_attack_windows,
    get_malicious_dates,
    load_insiders,
)


@pytest.fixture()
def answers_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'answers'
    d.mkdir()
    (d / 'insiders.csv').write_text(
        'dataset,scenario,details,user,start,end\n'
        '4.2,1,f1.csv,AAM0658,1/4/2010 8:47:00,1/8/2010 17:00:00\n'
        '4.2,2,f2.csv,CDE1846,3/1/2010 9:00:00,3/3/2010 18:00:00\n'
        '5.2,3,f3.csv,XYZ9999,6/1/2010 9:00:00,6/2/2010 18:00:00\n'
    )
    return d


def test_load_insiders_count(answers_dir: Path) -> None:
    assert len(load_insiders(answers_dir, 'r4.2')) == 2


def test_load_insiders_filters_by_version(answers_dir: Path) -> None:
    users = {r.user_id for r in load_insiders(answers_dir, 'r5.2')}
    assert users == {'XYZ9999'}


def test_load_insiders_family_prefix_match(answers_dir: Path) -> None:
    # 'r4x' family must prefix-match the '4.2' rows
    assert len(load_insiders(answers_dir, 'r4x')) == 2


def test_load_insiders_zero_match_raises(answers_dir: Path) -> None:
    with pytest.raises(ValueError, match='zero-labels'):
        load_insiders(answers_dir, 'r9.9')


def test_get_attack_windows_keyed_by_user(answers_dir: Path) -> None:
    windows = get_attack_windows(answers_dir, 'r4.2')
    assert set(windows.keys()) == {'AAM0658', 'CDE1846'}
    assert isinstance(windows['AAM0658'], InsiderRecord)


def test_get_malicious_dates(answers_dir: Path) -> None:
    md = get_malicious_dates(answers_dir, 'r4.2')
    assert len(md) == 8  # AAM0658: 5 days + CDE1846: 3 days (inclusive)
    user, date = next(iter(md))
    assert isinstance(user, str)
    assert isinstance(date, dt.date)


def test_insider_record_is_frozen() -> None:
    r = InsiderRecord(
        'U1', 1, '4.2',
        pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-02'), 'f.csv',
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.user_id = 'changed'  # type: ignore[misc]
