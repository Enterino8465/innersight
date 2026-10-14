"""The Stage 7 shape guard, which now derives its expectation instead of
remembering one release's answer.

Three jobs, in order of how much they matter.

**One: r4.2's verdict has not moved.** The guard used to compare every run
against ``EXPECTED_ROWS_PER_SEED = 93_000`` and two friends, hand-typed from
r4.2's Stage 4 output. Those are gone. What replaces them must reach the *same
verdict* on the r4.2 test pass that is already frozen in
``outputs/phase2_test.json``, and :func:`test_the_old_constants_and_the_new_derivation_agree`
asserts exactly that, by running the old arithmetic here beside the new.

**Two: the numbers are pinned as literals.** Every count below was read on
2026-08-22 from ``data/splits.json``, ``releases/r52/data/splits.json`` and the
frozen ``outputs/phase2_test.json``, and is written out rather than recomputed.
A test that derives its expectation from the code it is testing passes when
both are wrong together.

**Three: this file lives at the project root on purpose.** ``pytest
evaluation/ modeling/ -q`` is used elsewhere as a fingerprint of the frozen
result and must keep printing **101**. Adding tests here rather than under
``evaluation/`` leaves it at 101, which is the same choice brief R3 made with
``test_scenario_coverage.py``.

Run it with::

    python -m pytest test_shape_guard.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
import torch

from config import (
    DAY_COLUMN,
    FEATURE_NAMES,
    LABEL_COLUMN,
    N_FEATURES,
    PROFILES,
    RELEASE_R42,
    RELEASE_R52,
    SCENARIO_COLUMN,
    SPLITS_PATH,
    USER_COLUMN,
    WINDOW_DAYS,
)
from evaluation.corpus import Corpus, EvalSplit
from evaluation.testpass import expected_shape, shape_guard
from modeling.data import SplitData

ROOT = Path(__file__).resolve().parent

#: The mask drops each user's first ``WINDOW_DAYS - 1`` rows. Written out here
#: rather than imported so that a change to WINDOW_DAYS breaks this file loudly
#: instead of silently moving every expectation in it.
WARM_UP_DAYS = 27


# --------------------------------------------------------------------------
# The measured shapes, as literals
#
# `rows`, `users` and `positives_by_scenario` come from each release's
# splits.json summary block. `evaluated_rows` is what Stage 7 actually scored:
# for r4.2 it is quoted from the frozen phase2_test.json, and for r5.2 it is
# the arithmetic, which is why the r4.2 rows are the ones that prove the
# arithmetic is right.
# --------------------------------------------------------------------------

R42_TEST: Mapping[int, dict[str, Any]] = {
    42: {
        "users": 200, "rows": 93_287, "evaluated_rows": 87_887,
        "positives_by_scenario": {"0": 0, "1": 40, "2": 339, "3": 4},
        "users_by_scenario": {"0": 186, "1": 6, "2": 6, "3": 2},
    },
    123: {
        "users": 200, "rows": 92_767, "evaluated_rows": 87_367,
        "positives_by_scenario": {"0": 0, "1": 32, "2": 323, "3": 4},
        "users_by_scenario": {"0": 186, "1": 6, "2": 6, "3": 2},
    },
    456: {
        "users": 200, "rows": 94_120, "evaluated_rows": 88_720,
        "positives_by_scenario": {"0": 0, "1": 56, "2": 327, "3": 4},
        "users_by_scenario": {"0": 186, "1": 6, "2": 6, "3": 2},
    },
}

R52_TEST: Mapping[int, dict[str, Any]] = {
    42: {
        "users": 400, "rows": 196_302, "evaluated_rows": 185_502,
        "positives_by_scenario": {"0": 0, "1": 44, "2": 307, "3": 4, "4": 437},
        "users_by_scenario": {"0": 380, "1": 6, "2": 6, "3": 2, "4": 6},
    },
    123: {
        "users": 400, "rows": 194_787, "evaluated_rows": 183_987,
        "positives_by_scenario": {"0": 0, "1": 32, "2": 321, "3": 4, "4": 413},
        "users_by_scenario": {"0": 380, "1": 6, "2": 6, "3": 2, "4": 6},
    },
    456: {
        "users": 400, "rows": 194_419, "evaluated_rows": 183_619,
        "positives_by_scenario": {"0": 0, "1": 39, "2": 318, "3": 4, "4": 528},
        "users_by_scenario": {"0": 380, "1": 6, "2": 6, "3": 2, "4": 6},
    },
}

#: The exact line the guard prints for r4.2 seed 42, pinned. Brief R7 §9.2:
#: the verdict must not move and the message must not move unnoticed.
R42_SEED42_DETAIL = (
    "87,887 rows (expected 87,887 = 93,287 - 27 warm-up days x 200 users), "
    "383 positives (expected 383, 0.4358%), scenario-3 positives 4 "
    "(expected exactly 4; scenario 3 is the scarcest in this split)"
)


# --------------------------------------------------------------------------
# Fixtures: a splits.json, and a split shaped like one
# --------------------------------------------------------------------------


def _summary_block(shape: Mapping[str, Any]) -> dict[str, Any]:
    """One ``summary[seed][split]`` block, in Stage 4's format."""
    positives = sum(shape["positives_by_scenario"].values())
    return {
        "users": shape["users"],
        "by_scenario": dict(shape["users_by_scenario"]),
        "rows": shape["rows"],
        "positive_rows": positives,
        "positive_rate": positives / shape["rows"],
        "positives_by_scenario": dict(shape["positives_by_scenario"]),
    }


def _write_splits(
    path: Path, shapes: Mapping[int, Mapping[str, Any]], split: str = "test"
) -> Path:
    """Write a splits.json carrying only what the guard reads: the summary."""
    payload = {
        "summary": {
            str(seed): {split: _summary_block(shape)}
            for seed, shape in shapes.items()
        }
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _eval_split(
    name: str, *, n_rows: int, positives_by_scenario: Mapping[str, int]
) -> EvalSplit:
    """An :class:`EvalSplit` with the right counts and no real data in it.

    The guard reads five things — the name, the row count, the positive count,
    the positive rate and the (label, scenario) pair of every row — so those
    five are real here and the feature tensors are zeros of the correct shape.
    Building it as a genuine EvalSplit rather than a stand-in means a change to
    that class breaks this file, which is the point.
    """
    labels = np.zeros(n_rows, dtype=np.int64)
    scenarios = np.zeros(n_rows, dtype=np.int64)
    cursor = 0
    for scenario, count in sorted(positives_by_scenario.items()):
        if int(scenario) == 0:
            continue
        labels[cursor : cursor + count] = 1
        scenarios[cursor : cursor + count] = int(scenario)
        cursor += count

    users = np.full(n_rows, "USER0000", dtype="U16")
    data = SplitData(
        name=name,
        features=torch.zeros((n_rows, N_FEATURES), dtype=torch.float32),
        target_positions=torch.arange(n_rows, dtype=torch.int64),
        labels=torch.from_numpy(labels.astype(np.float32)),
        scenarios=torch.from_numpy(scenarios),
        users=users,
        user_ids=("USER0000",),
    )
    return EvalSplit(
        data=data,
        days=np.zeros(n_rows, dtype="datetime64[ns]"),
        users=users,
        labels=labels,
        scenarios=scenarios,
        raw_features=np.zeros((n_rows, N_FEATURES), dtype=np.float32),
    )


def _split_for(shape: Mapping[str, Any], name: str = "test") -> EvalSplit:
    return _eval_split(
        name,
        n_rows=shape["evaluated_rows"],
        positives_by_scenario=shape["positives_by_scenario"],
    )


def _guard(tmp_path: Path, seed: int, shapes: Mapping[int, Mapping[str, Any]]):
    """Run the guard for one seed over a written splits file."""
    splits_file = _write_splits(tmp_path / "splits.json", shapes)
    split = _split_for(shapes[seed])
    return shape_guard(seed, split, expected_shape(splits_file, seed, "test"))


# --------------------------------------------------------------------------
# 1. r4.2 has not moved
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", sorted(R42_TEST))
def test_r42_still_passes_on_every_seed(tmp_path: Path, seed: int) -> None:
    result = _guard(tmp_path, seed, R42_TEST)
    assert result.passed
    assert result.fatal is False
    assert result.name == f"seed {seed}: test split matches splits.json"


def test_r42_detail_line_is_pinned(tmp_path: Path) -> None:
    """The message a reader checks the guard against, fixed as a literal."""
    assert _guard(tmp_path, 42, R42_TEST).detail == R42_SEED42_DETAIL


#: The pre-R7 guard, reproduced exactly so the two can be compared rather than
#: assumed equivalent. Deleted from `testpass.py`; kept here as the baseline.
_OLD_ROWS = 93_000
_OLD_POSITIVES = 376
_OLD_SCENARIO_3 = 4
_OLD_TOLERANCE = 0.10


def _old_verdict(split: EvalSplit) -> bool:
    scenario_3 = int(((split.labels == 1) & (split.scenarios == 3)).sum())
    off = (
        abs(split.n_rows - _OLD_ROWS) > _OLD_TOLERANCE * _OLD_ROWS
        or abs(split.n_positives - _OLD_POSITIVES) > _OLD_TOLERANCE * _OLD_POSITIVES
        or scenario_3 != _OLD_SCENARIO_3
    )
    return not off


@pytest.mark.parametrize("seed", sorted(R42_TEST))
def test_the_old_constants_and_the_new_derivation_agree(
    tmp_path: Path, seed: int
) -> None:
    """Brief R7 §9.2, as the identity it was actually protecting.

    The detail string could not survive deriving the expectation — it contains
    the expectation. The verdict could, and does, on all three seeds.
    """
    shape = R42_TEST[seed]
    assert _old_verdict(_split_for(shape)) is True
    assert _guard(tmp_path, seed, R42_TEST).passed is True


# --------------------------------------------------------------------------
# 2. r5.2, which is what the whole exercise is for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", sorted(R52_TEST))
def test_r52_passes_where_the_old_constants_would_have_failed(
    tmp_path: Path, seed: int
) -> None:
    """The guard that cried wolf, and the one that does not.

    r5.2's test splits hold roughly twice r4.2's rows, so the old constants
    reject every seed with a message about the splits file and the dataset
    disagreeing about who exists — which was false.
    """
    shape = R52_TEST[seed]
    assert _old_verdict(_split_for(shape)) is False
    result = _guard(tmp_path, seed, R52_TEST)
    assert result.passed
    assert f"{shape['evaluated_rows']:,} rows" in result.detail


def test_the_r52_expectation_is_the_arithmetic(tmp_path: Path) -> None:
    """185,502 = 196,302 - 27 x 400, and the guard says so in as many words."""
    splits_file = _write_splits(tmp_path / "splits.json", R52_TEST)
    expected = expected_shape(splits_file, 42, "test")
    assert expected is not None
    assert expected.warm_up_rows == WARM_UP_DAYS * 400
    assert expected.evaluated_rows == 196_302 - WARM_UP_DAYS * 400 == 185_502
    assert expected.positives == 792
    assert expected.scarcest_scenario == 3
    assert expected.scarcest_positives == 4


# --------------------------------------------------------------------------
# 3. What the guard is supposed to catch
# --------------------------------------------------------------------------


def test_eight_lost_users_are_caught_where_the_old_guard_passed(
    tmp_path: Path,
) -> None:
    """The failure the old tolerance was wide enough to swallow.

    Eight benign users vanish between Stage 4 and Stage 7 — 466 rows each,
    439 of them evaluated. The old constants still pass, because the loss lands
    inside a +/-10% band centred 5.8% away from the truth to begin with. The
    derived guard fails, because the splits file said 200 users and 93,287
    rows and the dataset no longer holds them.
    """
    shape = R42_TEST[42]
    lost = 8 * (466 - WARM_UP_DAYS)
    split = _eval_split(
        "test",
        n_rows=shape["evaluated_rows"] - lost,
        positives_by_scenario=shape["positives_by_scenario"],
    )
    assert _old_verdict(split) is True

    splits_file = _write_splits(tmp_path / "splits.json", R42_TEST)
    result = shape_guard(42, split, expected_shape(splits_file, 42, "test"))
    assert result.passed is False
    assert "expected 87,887" in result.detail


def test_one_missing_positive_is_caught(tmp_path: Path) -> None:
    """Positives are compared exactly: the mask costs zero on both releases."""
    shape = R42_TEST[42]
    thinner = dict(shape["positives_by_scenario"]) | {"2": 338}
    split = _eval_split(
        "test",
        n_rows=shape["evaluated_rows"],
        positives_by_scenario=thinner,
    )
    splits_file = _write_splits(tmp_path / "splits.json", R42_TEST)
    result = shape_guard(42, split, expected_shape(splits_file, 42, "test"))
    assert result.passed is False
    assert "382 positives (expected 383" in result.detail


def test_users_with_short_records_do_not_trip_the_guard(tmp_path: Path) -> None:
    """A user with fewer than 28 days loses their whole record, not 27 rows.

    So the evaluated count can sit *above* the exact expectation without
    anything being wrong, and the guard says which it saw rather than failing.
    Neither release contains such a user today — the minimum record is 31 days
    on r4.2 and 30 on r5.2 — so this is about not crying wolf on the next one.
    """
    shape = R42_TEST[42]
    split = _eval_split(
        "test",
        n_rows=shape["evaluated_rows"] + 60,
        positives_by_scenario=shape["positives_by_scenario"],
    )
    splits_file = _write_splits(tmp_path / "splits.json", R42_TEST)
    result = shape_guard(42, split, expected_shape(splits_file, 42, "test"))
    assert result.passed is True
    assert "60 rows above the exact expectation" in result.detail


def test_more_rows_than_the_split_holds_is_a_failure(tmp_path: Path) -> None:
    """The upper bound is a theorem too: the mask never adds a row."""
    shape = R42_TEST[42]
    split = _eval_split(
        "test",
        n_rows=shape["rows"] + 1,
        positives_by_scenario=shape["positives_by_scenario"],
    )
    splits_file = _write_splits(tmp_path / "splits.json", R42_TEST)
    assert shape_guard(42, split, expected_shape(splits_file, 42, "test")).passed is False


# --------------------------------------------------------------------------
# 4. The fragile scenario is derived, not assumed
# --------------------------------------------------------------------------


def test_the_scarcest_scenario_is_derived_not_assumed(tmp_path: Path) -> None:
    """Brief R7 §9.4. Scenario 3 is the answer on both real releases and it is
    the answer by coincidence — ten insiders with two attack days each, two of
    them per test split. Given a split where scenario 2 is the scarce one, the
    guard must name scenario 2."""
    shape = {
        "users": 200, "rows": 93_287, "evaluated_rows": 87_887,
        "positives_by_scenario": {"0": 0, "1": 40, "2": 3, "3": 340},
        "users_by_scenario": {"0": 186, "1": 6, "2": 2, "3": 6},
    }
    splits_file = _write_splits(tmp_path / "splits.json", {42: shape})
    expected = expected_shape(splits_file, 42, "test")
    assert expected is not None
    assert expected.scarcest_scenario == 2
    assert expected.scarcest_positives == 3

    result = shape_guard(42, _split_for(shape), expected)
    assert result.passed
    assert "scenario-2 positives 3" in result.detail
    assert "scenario 2 is the scarcest in this split" in result.detail


def test_a_scenario_the_split_does_not_contain_wins(tmp_path: Path) -> None:
    """Zero positive rows is the scarcest there is, and the most useful line
    the guard could print in that situation."""
    shape = {
        "users": 200, "rows": 93_287, "evaluated_rows": 87_887,
        "positives_by_scenario": {"0": 0, "1": 40, "2": 343, "3": 0},
        "users_by_scenario": {"0": 188, "1": 6, "2": 6, "3": 0},
    }
    splits_file = _write_splits(tmp_path / "splits.json", {42: shape})
    expected = expected_shape(splits_file, 42, "test")
    assert expected is not None
    assert expected.scarcest_scenario == 3
    assert expected.scarcest_positives == 0


# --------------------------------------------------------------------------
# 5. A guard that cannot check must not crash the run it is watching
# --------------------------------------------------------------------------


def test_a_splits_file_without_a_summary_is_not_a_failure(tmp_path: Path) -> None:
    """`evaluation/test_evaluation.py` writes exactly this shape of file."""
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({"splits": {"7": {"test": ["USER0000"]}}}))
    result = shape_guard(7, _split_for(R42_TEST[42]), expected_shape(splits_file, 7, "test"))
    assert result.passed is True
    assert result.fatal is False
    assert "shape not checked" in result.detail


def test_a_missing_splits_file_is_not_a_failure(tmp_path: Path) -> None:
    assert expected_shape(tmp_path / "absent.json", 42, "test") is None


def test_an_unknown_seed_is_not_a_failure(tmp_path: Path) -> None:
    splits_file = _write_splits(tmp_path / "splits.json", R42_TEST)
    assert expected_shape(splits_file, 999, "test") is None


# --------------------------------------------------------------------------
# 5b. The whole expression, against rows the real code path produced
# --------------------------------------------------------------------------


def _dense_frame(n_users: int = 10, days: int = 60) -> pd.DataFrame:
    """A small dense corpus: every user, every calendar day, two insiders."""
    start = pd.Timestamp("2010-01-04")
    scenario_of = {"INS0001": 1, "INS0002": 2}
    rows: list[dict[str, Any]] = []
    users = [f"BEN{i:04d}" for i in range(n_users - 2)] + ["INS0001", "INS0002"]
    for user in users:
        scenario = scenario_of.get(user, 0)
        for day in range(days):
            attack = scenario and 40 <= day < 45
            rows.append(
                {
                    USER_COLUMN: user,
                    DAY_COLUMN: start + pd.Timedelta(days=day),
                    LABEL_COLUMN: int(bool(attack)),
                    SCENARIO_COLUMN: scenario,
                    **dict(zip(FEATURE_NAMES, np.zeros(N_FEATURES, dtype=np.float32))),
                }
            )
    return pd.DataFrame(rows)


def test_the_expression_phase1_runs_holds_on_a_real_corpus(tmp_path: Path) -> None:
    """The call site, end to end, on rows `Corpus.build` actually produced.

    Everything above hands the guard a split built to order. This one goes
    through `modeling.data`'s real mask, so it is the test that would catch the
    27-rows-per-user arithmetic being wrong — `expected.evaluated_rows` is
    arithmetic on the splits file and `split.n_rows` is what the mask left.
    """
    frame = _dense_frame()
    users = sorted(frame[USER_COLUMN].unique())
    val_users = users[-4:]
    per_user = len(frame) // len(users)

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps(
            {
                "splits": {"7": {"train": users[:-4], "val": val_users, "test": []}},
                "summary": {
                    "7": {
                        "val": {
                            "users": len(val_users),
                            "rows": per_user * len(val_users),
                            "positive_rows": int(
                                frame.loc[
                                    frame[USER_COLUMN].isin(val_users), LABEL_COLUMN
                                ].sum()
                            ),
                            "by_scenario": {"0": 2, "1": 1, "2": 1},
                            "positives_by_scenario": {"0": 0, "1": 5, "2": 5},
                        }
                    }
                },
            }
        )
    )

    corpus = Corpus.load(frame=frame, splits_file=splits_file, window_days=WINDOW_DAYS)
    split = corpus.build("val", 7, corpus.refit_train_standardiser(7))
    expected = expected_shape(corpus.splits_file, 7, split.name)

    assert expected is not None
    assert expected.evaluated_rows == split.n_rows
    assert expected.positives == split.n_positives
    result = shape_guard(7, split, expected)
    assert result.passed, result.detail


# --------------------------------------------------------------------------
# 6. The real splits files, when this checkout has them
#
# `data/` and `releases/` are both gitignored, so neither file is in the
# repository. These skip rather than fail when they are absent, and name the
# file they wanted so a skip never reads as a pass.
# --------------------------------------------------------------------------


R52_SPLITS_PATH = PROFILES[RELEASE_R52].data_dir / "splits.json"


@pytest.mark.parametrize("seed", sorted(R42_TEST))
def test_the_real_r42_splits_file_matches_the_frozen_test_pass_if_present(
    seed: int,
) -> None:
    if not SPLITS_PATH.is_file():
        pytest.skip(f"{SPLITS_PATH} is not in this checkout (data/ is gitignored)")
    expected = expected_shape(SPLITS_PATH, seed, "test")
    assert expected is not None
    shape = R42_TEST[seed]
    assert expected.rows == shape["rows"]
    assert expected.users == shape["users"]
    # The frozen phase2_test.json recorded these row counts on 2026-08-21.
    assert expected.evaluated_rows == shape["evaluated_rows"]
    assert expected.positives == sum(shape["positives_by_scenario"].values())
    assert expected.scarcest_scenario == 3
    assert expected.scarcest_positives == 4


@pytest.mark.parametrize("seed", sorted(R52_TEST))
def test_the_real_r52_splits_file_gives_the_expected_shape_if_present(
    seed: int,
) -> None:
    if not R52_SPLITS_PATH.is_file():
        pytest.skip(f"{R52_SPLITS_PATH} is not in this checkout (releases/ is gitignored)")
    expected = expected_shape(R52_SPLITS_PATH, seed, "test")
    assert expected is not None
    shape = R52_TEST[seed]
    assert expected.rows == shape["rows"]
    assert expected.evaluated_rows == shape["evaluated_rows"]
    assert expected.positives == sum(shape["positives_by_scenario"].values())
    assert expected.scarcest_scenario == 3
    assert expected.scarcest_positives == 4


# --------------------------------------------------------------------------
# 7. The audit marker moves with the release, both directions
# --------------------------------------------------------------------------


_MARKER_PROBE = """
import json
from evaluation import testpass
import config
print(json.dumps({
    "marker": str(testpass.OPENED_MARKER),
    "config_path": str(config.TEST_SPLIT_OPENED_PATH),
    "release": config.ACTIVE_RELEASE,
}))
"""


def _probe_marker(release: str | None) -> dict[str, str]:
    """Import `evaluation.testpass` in a fresh process under one release.

    A subprocess for the same reason `test_config.py` uses one: `config` reads
    the environment once, at import, and `importlib.reload` would rebind names
    inside it while every `from config import X` kept the old value.
    """
    environment = dict(os.environ)
    environment.pop("INNERSIGHT_RELEASE", None)
    if release is not None:
        environment["INNERSIGHT_RELEASE"] = release
    completed = subprocess.run(
        [sys.executable, "-c", _MARKER_PROBE],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"importing evaluation.testpass under "
            f"INNERSIGHT_RELEASE={release!r} exited {completed.returncode}\n"
            f"{completed.stderr}"
        )
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def marker_r42() -> dict[str, str]:
    """Probed once: each call is a fresh interpreter importing torch."""
    return _probe_marker(None)


@pytest.fixture(scope="module")
def marker_r52() -> dict[str, str]:
    return _probe_marker("5.2")


def test_an_r52_opening_cannot_write_to_r42s_marker(
    marker_r42: dict[str, str], marker_r52: dict[str, str]
) -> None:
    """Brief R7 §9.5. The r4.2 test split was opened once, on
    2026-08-21T11:02:25+00:00, and nothing may append to that file again."""
    root = str(ROOT)
    r42 = marker_r42
    r52 = marker_r52

    assert r42["release"] == RELEASE_R42
    assert r42["marker"] == f"{root}/outputs/TEST_SPLIT_OPENED.json"
    assert r52["release"] == RELEASE_R52
    assert r52["marker"] == f"{root}/releases/r52/outputs/TEST_SPLIT_OPENED.json"
    assert r42["marker"] != r52["marker"]


def test_the_marker_is_configs_path_and_not_a_second_copy_of_it(
    marker_r42: dict[str, str], marker_r52: dict[str, str]
) -> None:
    """It used to be `OUTPUT_DIR / "TEST_SPLIT_OPENED.json"` — the same value,
    written twice. `report.py` held a duplicate path like this one until brief
    R1, where the copy was anchored to PROJECT_ROOT and would have written
    r5.2's report over r4.2's frozen one."""
    for probe in (marker_r42, marker_r52):
        assert probe["marker"] == probe["config_path"]
