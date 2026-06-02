"""End-to-end integration test for the Phase 1 universal pipeline.

Proves that load_version() output flows through the existing feature
extraction (features.build_user_day_features), the Parquet feature store,
and the backward-compatible time_split — all with synthetic data, no real
CERT dataset required.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from innersight.backend.data.answers import get_malicious_dates
from innersight.backend.data.feature_store import FeatureStore
from innersight.backend.data.pipeline import load_version, time_split
from innersight.backend.features.features import build_user_day_features

INSIDER = 'AAF0535'
NORMALS = ('NORM0001', 'NORM0002')
# The insider's malicious day; their after-hours logon + USB connect land here.
ATTACK_DAY = dt.date(2010, 1, 5)


@pytest.fixture()
def r4_full_dir(tmp_path: Path) -> Path:
    """A comprehensive synthetic r4.2 directory: 3 users over multiple days."""
    d = tmp_path / 'r4.2'
    d.mkdir()

    # logon: normals during business hours; insider has an after-hours logon (01:34).
    (d / 'logon.csv').write_text(
        'id,date,user,pc,activity\n'
        '{L01},01/04/2010 08:00:00,NORM0001,PC-1,Logon\n'
        '{L02},01/04/2010 17:00:00,NORM0001,PC-1,Logoff\n'
        '{L03},01/05/2010 08:05:00,NORM0001,PC-1,Logon\n'
        '{L04},01/05/2010 17:02:00,NORM0001,PC-1,Logoff\n'
        '{L05},01/04/2010 09:00:00,NORM0002,PC-2,Logon\n'
        '{L06},01/04/2010 18:00:00,NORM0002,PC-2,Logoff\n'
        '{L07},01/05/2010 01:34:00,AAF0535,PC-9,Logon\n'
        '{L08},01/05/2010 03:10:00,AAF0535,PC-9,Logoff\n'
    )
    # device: insider connects a USB after-hours (02:00) on the attack day.
    (d / 'device.csv').write_text(
        'id,date,user,pc,activity\n'
        '{D01},01/04/2010 10:00:00,NORM0001,PC-1,Connect\n'
        '{D02},01/04/2010 10:30:00,NORM0001,PC-1,Disconnect\n'
        '{D03},01/05/2010 02:00:00,AAF0535,PC-9,Connect\n'
        '{D04},01/05/2010 02:45:00,AAF0535,PC-9,Disconnect\n'
    )
    # http: insider visits a job site (job_search keyword 'job').
    (d / 'http.csv').write_text(
        'id,date,user,pc,url,content\n'
        '{H01},01/04/2010 11:00:00,NORM0001,PC-1,http://msn.com/news,news\n'
        '{H02},01/04/2010 12:00:00,NORM0002,PC-2,http://google.com/search,search\n'
        '{H03},01/05/2010 02:10:00,AAF0535,PC-9,http://monster.com/jobs,resume\n'
    )
    # email: insider sends an external email with a large attachment.
    (d / 'email.csv').write_text(
        'id,date,user,pc,to,cc,bcc,from,size,attachments,content\n'
        '{E01},01/04/2010 13:00:00,NORM0001,PC-1,coworker@dtaa.com,,,NORM0001@dtaa.com,1500,0,hi\n'
        '{E02},01/05/2010 02:20:00,AAF0535,PC-9,outside@gmail.com,,,AAF0535@dtaa.com,2000000,1,leak\n'
    )
    # file: insider copies a file (r4.x => implicit removable-media copy).
    (d / 'file.csv').write_text(
        'id,date,user,pc,filename,content\n'
        '{F01},01/04/2010 14:00:00,NORM0001,PC-1,report.doc,D0-CF data\n'
        '{F02},01/05/2010 02:25:00,AAF0535,PC-9,secret.doc,D0-CF secret\n'
    )

    # LDAP with role info for all three users.
    ldap = d / 'LDAP'
    ldap.mkdir()
    (ldap / '2010-01.csv').write_text(
        'employee_name,user_id,email,role,business_unit,functional_unit,department,team,supervisor\n'
        'Norm One,NORM0001,NORM0001@dtaa.com,Engineer,1,2 - R&E,2 - SW,3 - Software,Boss A\n'
        'Norm Two,NORM0002,NORM0002@dtaa.com,Salesman,1,5 - Sales,4 - Sales,5 - SalesDept,Boss B\n'
        'Aaf Insider,AAF0535,AAF0535@dtaa.com,Engineer,1,2 - R&E,2 - SW,3 - Software,Boss A\n'
    )
    # psychometric OCEAN scores for all three users.
    (d / 'psychometric.csv').write_text(
        'employee_name,user_id,O,C,E,A,N\n'
        'Norm One,NORM0001,40,39,36,19,40\n'
        'Norm Two,NORM0002,30,45,50,33,20\n'
        'Aaf Insider,AAF0535,25,20,15,18,48\n'
    )

    # answers: one insider entry covering the attack day.
    ans = d / 'answers'
    ans.mkdir()
    (ans / 'insiders.csv').write_text(
        'dataset,scenario,details,user,start,end\n'
        '4.2,2,r4.2-2-AAF0535.csv,AAF0535,1/5/2010 0:00:00,1/5/2010 23:59:00\n'
    )
    return d


def test_load_version_all_logs_non_empty(r4_full_dir: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    for name, df in ds.logs.items():
        assert not df.empty, f"log {name!r} should have rows"


def test_insider_in_records_and_windows(r4_full_dir: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    assert INSIDER in {r.user_id for r in ds.insiders}
    assert INSIDER in ds.attack_windows
    assert ds.attack_windows[INSIDER].scenario == 2


def test_ldap_and_psychometric_counts(r4_full_dir: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    assert len(ds.ldap) == 3
    assert len(ds.psychometric) == 3


def test_provenance_manifest(r4_full_dir: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    prov = ds.provenance
    for key in ('version', 'adapter', 'data_dir', 'loaded_at', 'row_counts',
                'insider_count', 'ldap_count', 'psychometric_count', 'file_hashes'):
        assert key in prov, f"provenance missing {key!r}"
    assert prov['version'] == 'r4.2'
    assert prov['row_counts']['logon'] == 8
    assert prov['insider_count'] == 1
    # file_hashes: one 64-char SHA-256 hex digest per top-level CSV.
    assert set(prov['file_hashes']) >= {'logon.csv', 'device.csv', 'http.csv', 'email.csv', 'file.csv'}
    assert all(len(h) == 64 for h in prov['file_hashes'].values())
    # loaded_at is a parseable ISO timestamp.
    dt.datetime.fromisoformat(prov['loaded_at'])


def test_features_compatible_with_pipeline_output(r4_full_dir: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    malicious = get_malicious_dates(r4_full_dir / 'answers', 'r4.2')
    feats = build_user_day_features(ds.logs, malicious)

    assert not feats.empty
    assert list(feats.columns).count('date') == 1  # no duplicate date column
    for col in ('user', 'date', 'after_hours_logons', 'usb_connect_count', 'is_malicious'):
        assert col in feats.columns

    insider_attack = feats[(feats['user'] == INSIDER) & (feats['date'] == pd.Timestamp(ATTACK_DAY))]
    assert len(insider_attack) == 1
    row = insider_attack.iloc[0]
    assert row['is_malicious'] == 1
    assert row['after_hours_logons'] >= 1
    assert row['usb_connect_count'] >= 1


def test_feature_store_roundtrip_from_pipeline(r4_full_dir: Path, tmp_path: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    malicious = get_malicious_dates(r4_full_dir / 'answers', 'r4.2')
    feats = build_user_day_features(ds.logs, malicious)

    store = FeatureStore(tmp_path / 'fstore')
    store.save_features(ds.version, feats)
    loaded = store.load_features(ds.version)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, feats)


def test_backward_compat_time_split(r4_full_dir: Path) -> None:
    ds = load_version(r4_full_dir, 'r4.2')
    # End-of-day boundary so all 01/04 timestamps fall in train (split is on <=).
    splits = time_split(ds.logs, train_end='2010-01-04 23:59:59', val_end='2010-01-04 23:59:59')
    assert set(splits.keys()) == {'train', 'val', 'test'}
    # everything on 01/04 lands in train; the insider's 01/05 activity in test.
    assert len(splits['train']['logon']) > 0
    assert len(splits['test']['logon']) > 0
    for split in splits.values():
        assert list(split['logon'].columns) == list(ds.logs['logon'].columns)
