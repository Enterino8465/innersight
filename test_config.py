"""What ``config.py`` must keep being true now that it knows about releases.

Three jobs, in order of how much they matter.

**One: r4.2 has not moved.** Every path constant and every path-returning
function is asserted against a *literal* here -- ``<root>/data``, not
``RELEASE_ROOT / "data"``. That distinction is the whole point. A test that
recomputes its expectation from the profile it is testing passes even when the
profile is wrong, and the frozen r4.2 numbers (Model A AUPRC 0.1903 +/- 0.0300,
Model B 0.6663 +/- 0.1277, from the single test pass on 2026-08-21) are only
worth anything while they describe the code in this repository.

**Two: r5.2 selects correctly.** ``config`` reads ``INNERSIGHT_RELEASE`` once,
at import, so every assertion here runs in a subprocess with the variable set
or removed. ``importlib.reload`` would be faster and would also be a lie: it
rebinds names inside ``config`` while every module that did ``from config
import X`` keeps the old value, which is not what a real run looks like. It
also means this file does not import ``config`` itself, so it asserts the same
things whatever ``INNERSIGHT_RELEASE`` the person running pytest happens to
have exported.

**Three: nothing else had to change.** The six pipeline packages import exactly
the names from ``config`` they imported before, and none of them has learned
what a release is.

Run it with::

    python -m pytest test_config.py -q
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Importing config in a fresh process, under a chosen release
# --------------------------------------------------------------------------


def _run(release: str | None, body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` from the project root with INNERSIGHT_RELEASE set or removed.

    ``release=None`` removes the variable rather than setting it empty, so the
    default path is exercised the way a clean shell exercises it.
    """
    environment = dict(os.environ)
    environment.pop("INNERSIGHT_RELEASE", None)
    if release is not None:
        environment["INNERSIGHT_RELEASE"] = release
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _probe(release: str | None, body: str) -> dict[str, Any]:
    """Run ``body`` -- which must print one JSON object -- and parse it."""
    completed = _run(release, body)
    if completed.returncode != 0:
        pytest.fail(
            f"importing config under INNERSIGHT_RELEASE={release!r} exited "
            f"{completed.returncode}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


_SNAPSHOT = """
import json
import config

print(json.dumps({
    "active_release": config.ACTIVE_RELEASE,
    "selected_by_env": config.RELEASE_SELECTED_BY_ENV,
    "release_id": config.RELEASE_ID,
    "scenarios": list(config.SCENARIOS),
    "expected_insider_count": config.EXPECTED_INSIDER_COUNT,
    "expected_employee_count": config.EXPECTED_EMPLOYEE_COUNT,
    "answer_scenario_dirs": list(config.ANSWER_SCENARIO_DIRS),
    "banner": config.release_banner(),

    "release_root": str(config.RELEASE_ROOT),
    "data_dir": str(config.DATA_DIR),
    "output_dir": str(config.OUTPUT_DIR),
    "checkpoint_dir": str(config.CHECKPOINT_DIR),
    "figure_dir": str(config.FIGURE_DIR),
    "daily_counts_path": str(config.DAILY_COUNTS_PATH),
    "daily_counts_sample_path": str(config.DAILY_COUNTS_SAMPLE_PATH),
    "labelled_counts_path": str(config.LABELLED_COUNTS_PATH),
    "labelled_counts_sample_path": str(config.LABELLED_COUNTS_SAMPLE_PATH),
    "dataset_path": str(config.DATASET_PATH),
    "dataset_sample_path": str(config.DATASET_SAMPLE_PATH),
    "splits_path": str(config.SPLITS_PATH),
    "splits_sample_path": str(config.SPLITS_SAMPLE_PATH),
    "final_results_path": str(config.FINAL_RESULTS_PATH),
    "test_split_opened_path": str(config.TEST_SPLIT_OPENED_PATH),
    "stage7_report_path": str(config.STAGE7_REPORT_PATH),
    "model_store_dir": str(config.MODEL_STORE_DIR),
    "ui_run_dir": str(config.UI_RUN_DIR),

    "checkpoint_a42": str(config.checkpoint_path("model_a", 42)),
    "checkpoint_b456_sample": str(
        config.checkpoint_path("model_b", 456, sample=True)
    ),
    "metrics_a": str(config.metrics_path("model_a")),
    "metrics_a_nopw": str(config.metrics_path("model_a", tag="nopw")),
    "metrics_b_sample": str(config.metrics_path("model_b", sample=True)),
    "figure_b": str(config.figure_path("model_b")),

    "dataset_full": str(config.DATASET_FULL),
    "dataset_full_mirror": str(config.DATASET_FULL_MIRROR),
    "dataset_sample": str(config.DATASET_SAMPLE),
    "dataset_r52": str(config.DATASET_R52),
    "dataset_r52_mirror": str(config.DATASET_R52_MIRROR),
    "dataset_r52_sample": str(config.DATASET_R52_SAMPLE),
    "answers_dir": str(config.ANSWERS_DIR),
    "answers_dir_mirror": str(config.ANSWERS_DIR_MIRROR),

    "log_names": list(config.LOG_NAMES),
    "log_columns": {n: list(s.columns) for n, s in config.LOG_SPECS.items()},
    "log_filenames": {n: s.filename for n, s in config.LOG_SPECS.items()},
    "sample_rows": {
        n: s.expected_sample_rows for n, s in config.LOG_SPECS.items()
    },
    "release_log_columns": {
        r: {n: list(c) for n, c in t.items()}
        for r, t in config.RELEASE_LOG_COLUMNS.items()
    },
    "release_capability": dict(config.RELEASE_CAPABILITY),

    "all_checkpoint_dirs": [str(p) for p in config.ALL_CHECKPOINT_DIRS],
    "all_final_results": [str(p) for p in config.ALL_FINAL_RESULTS_PATHS],
    "all_test_split_opened": [
        str(p) for p in config.ALL_TEST_SPLIT_OPENED_PATHS
    ],

    "seeds": list(config.SEEDS),
    "window_days": config.WINDOW_DAYS,
    "min_history_days": config.MIN_HISTORY_DAYS,
    "z_clip": config.Z_CLIP,
    "std_floor_multiplier": config.STD_FLOOR_MULTIPLIER,
    "std_floor_active_users_only": config.STD_FLOOR_ACTIVE_USERS_ONLY,
    "std_floor_fallback": config.STD_FLOOR_FALLBACK,
    "validation_z_threshold": config.VALIDATION_Z_THRESHOLD,
    "split": [config.SPLIT.train, config.SPLIT.val, config.SPLIT.test],
    "n_features": config.N_FEATURES,
    "feature_names": list(config.FEATURE_NAMES),
    "job_search_domains": list(config.JOB_SEARCH_DOMAINS),
    "cloud_upload_domains": list(config.CLOUD_UPLOAD_DOMAINS),

    "profiles": {
        r: {
            "release_id": p.release_id,
            "scenarios": list(p.scenarios),
            "answer_scenario_dirs": list(p.answer_scenario_dirs),
            "log_names": [s.name for s in p.log_specs],
            "checkpoint_dir": str(p.checkpoint_dir),
            "output_dir": str(p.output_dir),
        }
        for r, p in config.PROFILES.items()
    },
}))
"""


@pytest.fixture(scope="module")
def r42() -> dict[str, Any]:
    """Everything ``config`` exposes with INNERSIGHT_RELEASE removed."""
    return _probe(None, _SNAPSHOT)


@pytest.fixture(scope="module")
def r52() -> dict[str, Any]:
    """Everything ``config`` exposes with INNERSIGHT_RELEASE=5.2."""
    return _probe("5.2", _SNAPSHOT)


# --------------------------------------------------------------------------
# 1. The r4.2 identity test -- literals, never recomputed
# --------------------------------------------------------------------------


def test_r42_is_the_default_when_the_variable_is_unset(r42: dict) -> None:
    assert r42["active_release"] == "4.2"
    assert r42["selected_by_env"] is False
    assert r42["release_id"] == "4.2"


def test_r42_scalars_are_exactly_what_they_were(r42: dict) -> None:
    assert r42["scenarios"] == [1, 2, 3]
    assert r42["expected_insider_count"] == 70
    assert r42["expected_employee_count"] == 1_000
    assert r42["answer_scenario_dirs"] == ["r4.2-1", "r4.2-2", "r4.2-3"]


def test_r42_paths_are_exactly_what_they_were(r42: dict) -> None:
    """Written out one by one. No loop, no derivation, no cleverness."""
    root = str(ROOT)
    assert r42["release_root"] == root
    assert r42["data_dir"] == f"{root}/data"
    assert r42["output_dir"] == f"{root}/outputs"
    assert r42["checkpoint_dir"] == f"{root}/outputs/checkpoints"
    assert r42["figure_dir"] == f"{root}/outputs/figures"
    assert r42["daily_counts_path"] == f"{root}/data/daily_counts.parquet"
    assert r42["daily_counts_sample_path"] == (
        f"{root}/data/daily_counts.sample.parquet"
    )
    assert r42["labelled_counts_path"] == f"{root}/data/labelled_counts.parquet"
    assert r42["labelled_counts_sample_path"] == (
        f"{root}/data/labelled_counts.sample.parquet"
    )
    assert r42["dataset_path"] == f"{root}/data/dataset.parquet"
    assert r42["dataset_sample_path"] == f"{root}/data/dataset.sample.parquet"
    assert r42["splits_path"] == f"{root}/data/splits.json"
    assert r42["splits_sample_path"] == f"{root}/data/splits.sample.json"
    assert r42["final_results_path"] == f"{root}/outputs/final_results.json"
    assert r42["test_split_opened_path"] == (
        f"{root}/outputs/TEST_SPLIT_OPENED.json"
    )
    assert r42["stage7_report_path"] == f"{root}/STAGE7_REPORT.md"
    assert r42["model_store_dir"] == f"{root}/models"
    assert r42["ui_run_dir"] == f"{root}/runs"


def test_r42_path_functions_are_exactly_what_they_were(r42: dict) -> None:
    root = str(ROOT)
    assert r42["checkpoint_a42"] == (
        f"{root}/outputs/checkpoints/model_a_seed42.pt"
    )
    assert r42["checkpoint_b456_sample"] == (
        f"{root}/outputs/checkpoints/model_b_seed456.sample.pt"
    )
    assert r42["metrics_a"] == f"{root}/outputs/model_a_metrics.json"
    assert r42["metrics_a_nopw"] == f"{root}/outputs/model_a_nopw_metrics.json"
    assert r42["metrics_b_sample"] == (
        f"{root}/outputs/model_b_metrics.sample.json"
    )
    assert r42["figure_b"] == f"{root}/outputs/figures/model_b_curves.png"


def test_r42_log_specs_are_exactly_what_they_were(r42: dict) -> None:
    assert r42["log_names"] == ["logon", "device", "file", "email", "http"]
    assert r42["log_columns"] == {
        "logon": ["id", "date", "user", "pc", "activity"],
        "device": ["id", "date", "user", "pc", "activity"],
        "file": ["id", "date", "user", "pc", "filename", "content"],
        "email": [
            "id", "date", "user", "pc", "to", "cc", "bcc",
            "from", "size", "attachments", "content",
        ],
        "http": ["id", "date", "user", "pc", "url", "content"],
    }
    assert r42["sample_rows"] == {
        "logon": 67_953,
        "device": 26_342,
        "file": 26_222,
        "email": 89_577,
        "http": 967_805,
    }


def test_dataset_locations_are_unchanged(r42: dict) -> None:
    assert r42["dataset_full"] == "/Volumes/dataset hard drive/dataset/r4.2"
    assert r42["dataset_full_mirror"] == (
        "/Users/michaelkuksov/Developer/innersight_dataset/r4.2 copy"
    )
    assert r42["dataset_sample"] == (
        "/Users/michaelkuksov/Developer/innersight_dataset/r4.2_sample"
    )
    assert r42["answers_dir"] == "/Volumes/dataset hard drive/dataset/answers"
    assert r42["answers_dir_mirror"] == (
        "/Users/michaelkuksov/Developer/innersight_dataset/answers"
    )


def test_modelling_constants_are_not_release_scoped(r42: dict, r52: dict) -> None:
    """They are the same under both releases, and this says so out loud.

    If a later brief ever makes one of these vary by release, this is where
    that decision has to be argued rather than slipped in.
    """
    shared = (
        "seeds", "window_days", "min_history_days", "z_clip",
        "std_floor_multiplier", "std_floor_active_users_only",
        "std_floor_fallback", "validation_z_threshold", "split",
        "n_features", "feature_names", "job_search_domains",
        "cloud_upload_domains",
    )
    for key in shared:
        assert r42[key] == r52[key], f"{key} must not vary by release"

    assert r42["seeds"] == [42, 123, 456]
    assert r42["window_days"] == 28
    assert r42["min_history_days"] == 30
    assert r42["z_clip"] == 10.0
    assert r42["std_floor_multiplier"] == 0.1
    assert r42["std_floor_active_users_only"] is True
    assert r42["std_floor_fallback"] == 1.0
    assert r42["validation_z_threshold"] == 2.0
    assert r42["split"] == [0.6, 0.2, 0.2]
    assert r42["n_features"] == 18
    assert r42["feature_names"][0] == "logon_count"
    assert r42["feature_names"][9] == "file_to_removable_count"
    assert r42["feature_names"][17] == "cloud_upload_visits"


# --------------------------------------------------------------------------
# 2. The r5.2 profile
# --------------------------------------------------------------------------


def test_r52_scalars(r52: dict) -> None:
    assert r52["active_release"] == "5.2"
    assert r52["selected_by_env"] is True
    assert r52["release_id"] == "5.2"
    assert r52["scenarios"] == [1, 2, 3, 4]
    assert r52["expected_insider_count"] == 99
    assert r52["expected_employee_count"] == 2_000
    assert r52["answer_scenario_dirs"] == [
        "r5.2-1", "r5.2-2", "r5.2-3", "r5.2-4"
    ]


def test_r52_roots_live_under_releases_r52(r52: dict) -> None:
    root = str(ROOT)
    assert r52["release_root"] == f"{root}/releases/r52"
    assert r52["data_dir"] == f"{root}/releases/r52/data"
    assert r52["output_dir"] == f"{root}/releases/r52/outputs"
    assert r52["checkpoint_dir"] == f"{root}/releases/r52/outputs/checkpoints"
    assert r52["figure_dir"] == f"{root}/releases/r52/outputs/figures"
    assert r52["dataset_path"] == f"{root}/releases/r52/data/dataset.parquet"
    assert r52["splits_path"] == f"{root}/releases/r52/data/splits.json"
    assert r52["stage7_report_path"] == f"{root}/releases/r52/STAGE7_REPORT.md"
    assert r52["final_results_path"] == (
        f"{root}/releases/r52/outputs/final_results.json"
    )
    assert r52["checkpoint_a42"] == (
        f"{root}/releases/r52/outputs/checkpoints/model_a_seed42.pt"
    )


def test_r52_does_not_move_the_ui_scratch_directories(r52: dict) -> None:
    """``models/`` and ``runs/`` are the write guard's fixed point.

    If they followed the release, the one function that is supposed to be
    unmovable would move with an environment variable.
    """
    root = str(ROOT)
    assert r52["model_store_dir"] == f"{root}/models"
    assert r52["ui_run_dir"] == f"{root}/runs"


def test_r52_log_specs_match_the_verified_headers(r52: dict) -> None:
    """The literals here were read from the shipped files with ``head -1``.

    Pinned as literals for the same reason r4.2's are: deriving them from
    ``RELEASE_LOG_COLUMNS`` would test that two expressions agree, not that
    either one matches the data.
    """
    assert r52["log_names"] == ["logon", "device", "file", "email", "http"]
    assert r52["log_columns"] == {
        "logon": ["id", "date", "user", "pc", "activity"],
        "device": ["id", "date", "user", "pc", "file_tree", "activity"],
        "file": [
            "id", "date", "user", "pc", "filename", "activity",
            "to_removable_media", "from_removable_media", "content",
        ],
        "email": [
            "id", "date", "user", "pc", "to", "cc", "bcc", "from",
            "activity", "size", "attachments", "content",
        ],
        "http": ["id", "date", "user", "pc", "url", "content"],
    }
    assert r52["log_filenames"] == {
        "logon": "logon.csv",
        "device": "device.csv",
        "file": "file.csv",
        "email": "email.csv",
        "http": "http.csv",
    }


def test_r52_sample_row_counts_are_the_measured_ones(r52: dict) -> None:
    """These were None until brief R2, because no r5.2 sample existed yet.

    `tools/build_r52_sample.py` has since been run, and these five counts were
    read off the sample it built with `wc -l` minus the header on 2026-08-21 --
    the convention `verify_setup.py` uses when it checks them.

    Pinned as literals for the same reason r4.2's are. Deriving them from
    anything would test that two expressions agree rather than that either one
    matches the file on disk, and the whole value of a checksum is that it was
    measured somewhere else.
    """
    assert r52["sample_rows"] == {
        "logon": 30_574,
        "device": 19_446,
        "file": 8_439,
        "email": 237_495,
        "http": 871_706,
    }


def test_r52_ships_no_spec_for_the_two_extra_files(r52: dict) -> None:
    """decoy_file.csv and psychometric.csv exist in r5.2 and are deliberately
    unused: they would give the models inputs the r4.2 result never had."""
    assert "decoy_file" not in r52["log_columns"]
    assert "psychometric" not in r52["log_columns"]


# --------------------------------------------------------------------------
# 3. Tables that describe every release, under either release
# --------------------------------------------------------------------------


def test_release_log_columns_names_both_releases(r42: dict, r52: dict) -> None:
    for snapshot in (r42, r52):
        table = snapshot["release_log_columns"]
        assert sorted(table) == ["4.2", "5.2"]
        assert table["4.2"]["file"] == [
            "id", "date", "user", "pc", "filename", "content"
        ]
        assert table["4.2"]["device"] == ["id", "date", "user", "pc", "activity"]
        # The dangerous one: r5.2 inserts `activity` at position 8, in the
        # middle of the row, where r4.2 has `size`.
        assert table["5.2"]["email"][8] == "activity"
        assert table["4.2"]["email"][8] == "size"
        assert table["5.2"]["device"] == [
            "id", "date", "user", "pc", "file_tree", "activity"
        ]
        assert table["4.2"]["logon"] == table["5.2"]["logon"]
        assert table["4.2"]["http"] == table["5.2"]["http"]


def test_both_releases_are_supported(r42: dict) -> None:
    """Pinned as literals, so that un-supporting a release shows up in a diff.

    This was ``test_release_capability_still_gates_r52`` and asserted r5.2 was
    NOT_YET_ALIGNED. r5.2 became SUPPORTED on 2026-08-22 under brief R8, and a
    test whose *name* asserts a state the project has left is worse than no
    test: the next reader takes the name as current and the assertion as
    incidental.

    Both rows are pinned rather than only the changed one. Flipping either back
    is a deliberate act -- it stops a release's whole pipeline -- and this is
    the line that makes it a deliberate *visible* one.
    """
    assert r42["release_capability"] == {
        "4.2": "SUPPORTED",
        "5.2": "SUPPORTED",
    }


def test_the_frozen_path_lists_cover_both_releases(r42: dict, r52: dict) -> None:
    root = str(ROOT)
    for snapshot in (r42, r52):
        assert sorted(snapshot["all_checkpoint_dirs"]) == sorted([
            f"{root}/outputs/checkpoints",
            f"{root}/releases/r52/outputs/checkpoints",
        ])
        assert sorted(snapshot["all_final_results"]) == sorted([
            f"{root}/outputs/final_results.json",
            f"{root}/releases/r52/outputs/final_results.json",
        ])
        assert sorted(snapshot["all_test_split_opened"]) == sorted([
            f"{root}/outputs/TEST_SPLIT_OPENED.json",
            f"{root}/releases/r52/outputs/TEST_SPLIT_OPENED.json",
        ])


def test_profiles_are_internally_consistent(r42: dict) -> None:
    for release, profile in r42["profiles"].items():
        assert profile["release_id"] == release
        assert profile["answer_scenario_dirs"] == [
            f"r{release}-{s}" for s in profile["scenarios"]
        ]
        assert profile["log_names"] == [
            "logon", "device", "file", "email", "http"
        ]
        assert profile["checkpoint_dir"] == (
            profile["output_dir"] + "/checkpoints"
        )


# --------------------------------------------------------------------------
# 4. Bad values fail loudly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["6.1", "52", "r5.2", "5.20", "4.3", "five"])
def test_an_unsupported_release_raises_at_import(value: str) -> None:
    completed = _run(value, "import config")
    assert completed.returncode != 0, (
        f"INNERSIGHT_RELEASE={value!r} was accepted; it must not be. Falling "
        f"back to r4.2 silently would let a typo write into the frozen tree."
    )
    assert "UnknownReleaseError" in completed.stderr
    assert "4.2" in completed.stderr and "5.2" in completed.stderr


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_release_means_unset_not_broken(value: str) -> None:
    snapshot = _probe(value, _SNAPSHOT)
    assert snapshot["active_release"] == "4.2"
    assert snapshot["selected_by_env"] is False


# --------------------------------------------------------------------------
# 5. The banner
# --------------------------------------------------------------------------


def test_the_banner_says_which_release_and_where_it_writes(
    r42: dict, r52: dict
) -> None:
    assert r42["banner"] == (
        "release 4.2 (INNERSIGHT_RELEASE unset) -- data/ outputs/"
    )
    assert r52["banner"] == (
        "release 5.2 (INNERSIGHT_RELEASE=5.2) -- "
        "releases/r52/data/ releases/r52/outputs/"
    )


_BANNER_SCRIPT = """
import logging
import config

# The order every stage's main() uses: config is already imported, and only
# now does logging get configured. A banner emitted at import time would have
# gone nowhere.
logging.basicConfig(level=logging.INFO, format="%(message)s")
config.ensure_directories()
config.resolve_device("cpu")
config.release_banner()
"""


def test_the_banner_reaches_a_stage_that_configured_logging_first() -> None:
    completed = _run(None, _BANNER_SCRIPT)
    assert completed.returncode == 0, completed.stderr
    banners = [
        line for line in completed.stderr.splitlines()
        if line.startswith("release ")
    ]
    assert banners == [
        "release 4.2 (INNERSIGHT_RELEASE unset) -- data/ outputs/"
    ], (
        "the banner must appear exactly once per process, from whichever "
        f"config function the stage happened to call first. Got: "
        f"{completed.stderr!r}"
    )


_CAPABILITY_SCRIPT = """
import json
import config

print(json.dumps({
    "capability": config.RELEASE_CAPABILITY.get(config.ACTIVE_RELEASE),
    "supported": config.RELEASE_SUPPORTED,
    "releases": sorted(config.PROFILES),
}))
"""


@pytest.mark.parametrize("release", ["4.2", "5.2"])
def test_the_alignment_warning_tracks_the_capability_table(release: str) -> None:
    """A release warns if and only if its capability row is not SUPPORTED.

    This used to assert that r5.2 warns, full stop -- which was true, and which
    would have turned the day r5.2 became SUPPORTED into a failing test that
    looked like a regression instead of the finish line. The property is what
    the mechanism actually promises, and it holds on both sides of that flip.

    Written as a property on 2026-08-21, under brief R3, while r5.2 was still
    NOT_YET_ALIGNED -- so it was verified in both states before it was trusted.
    """
    probe = _probe(release, _CAPABILITY_SCRIPT)
    assert probe["releases"] == ["4.2", "5.2"], (
        "a third release exists; add it to this test's parameters so its "
        "warning is covered too"
    )

    completed = _run(release, _BANNER_SCRIPT)
    assert completed.returncode == 0, completed.stderr

    aligned = probe["capability"] == probe["supported"]
    warned = "smoke test, not a result" in completed.stderr
    assert warned == (not aligned), (
        f"r{release} is {probe['capability']!r} but the run "
        f"{'warned' if warned else 'did not warn'}"
    )
    if warned:
        assert probe["capability"] in completed.stderr, (
            "the warning must name the state, so a reader can grep for it"
        )


# --------------------------------------------------------------------------
# 6. Nothing else changed
# --------------------------------------------------------------------------

#: Every name each frozen pipeline module imports from ``config``, recorded on
#: 2026-08-21 before brief R1 touched anything. These six packages are the ones
#: §7 forbids changing, and this snapshot is the proof that it did not.
#:
#: ``evaluation/report.py`` is the ONE deliberate exception, taken under §7's
#: optional clause with michael's agreement. It previously imported
#: ``PROJECT_ROOT`` and defined ``REPORT_PATH`` and ``FINAL_RESULTS_PATH``
#: locally; it now imports ``STAGE7_REPORT_PATH`` and ``FINAL_RESULTS_PATH``
#: from ``config`` instead. Nothing outside that file referenced either name.
_EXPECTED_CONFIG_IMPORTS: dict[str, list[str]] = {
    # Four names added by brief R2, and each one is a decision worth naming.
    # ACTIVE_RELEASE and RELEASE_R42 are read by the two places that genuinely
    # cannot decide from the data -- the header contract and the sanity checks,
    # where the expected ANSWER differs per release. RELEASE_LOG_COLUMNS lets a
    # rejected header say which release the folder actually is.
    # EMAIL_SEND_ACTIVITY is the "Send" literal, which belongs in config like
    # every other value read out of a raw log.
    "extraction/extract.py": [
        "ACTIVE_RELEASE", "BUSINESS_HOUR_END", "BUSINESS_HOUR_START",
        "CHUNK_SIZE", "CLOUD_UPLOAD_DOMAINS", "DAILY_COUNTS_PATH",
        "DAILY_COUNTS_SAMPLE_PATH", "DATE_FORMAT", "DAY_COLUMN",
        "DEVICE_CONNECT", "DEVICE_DISCONNECT", "EMAIL_SEND_ACTIVITY",
        "FEATURE_NAMES", "INTERNAL_DOMAIN", "JOB_SEARCH_DOMAINS",
        "LARGE_ATTACHMENT_BYTES", "LOGOFF_ACTIVITY", "LOGON_ACTIVITY",
        "LOG_NAMES", "LOG_SPECS", "RELEASE_LOG_COLUMNS", "RELEASE_R42",
        "USER_COLUMN", "WEEKEND_WEEKDAYS",
        "ensure_directories", "resolve_dataset",
    ],
    "extraction/test_extract.py": [
        "FEATURE_NAMES", "LARGE_ATTACHMENT_BYTES", "RELEASE_LOG_COLUMNS",
    ],
    "extraction/test_extract_datasets.py": ["DAILY_COUNTS_PATH", "PROFILES"],
    "labeling/label.py": [
        "BENIGN_SCENARIO", "DAILY_COUNTS_PATH", "DAILY_COUNTS_SAMPLE_PATH",
        "DAY_COLUMN", "FEATURE_NAMES", "INSIDERS_COLUMNS",
        "INSIDERS_DATE_FORMAT", "LABELLED_COUNTS_PATH",
        "LABELLED_COUNTS_SAMPLE_PATH", "LABEL_COLUMN", "RELEASE_ID",
        "SCENARIOS", "SCENARIO_COLUMN", "USER_COLUMN", "ensure_directories",
        "insiders_csv_path",
    ],
    "labeling/test_label.py": ["FEATURE_NAMES"],
    "baseline/baseline.py": [
        "BENIGN_SCENARIO", "DATASET_PATH", "DATASET_SAMPLE_PATH", "DAY_COLUMN",
        "FEATURE_NAMES", "LABELLED_COUNTS_PATH", "LABELLED_COUNTS_SAMPLE_PATH",
        "LABEL_COLUMN", "MIN_HISTORY_DAYS", "SCENARIOS", "SCENARIO_COLUMN",
        "STD_FLOOR", "STD_FLOOR_ACTIVE_USERS_ONLY", "STD_FLOOR_FALLBACK",
        "STD_FLOOR_MULTIPLIER", "USER_COLUMN", "VALIDATION_Z_THRESHOLD",
        "Z_CLIP", "ensure_directories",
    ],
    "baseline/test_baseline.py": [
        "DAY_COLUMN", "FEATURE_NAMES", "LABEL_COLUMN", "MIN_HISTORY_DAYS",
        "SCENARIO_COLUMN", "STD_FLOOR_FALLBACK", "USER_COLUMN",
    ],
    # Two names added by brief R3. SCENARIOS so a scenario with no users in
    # this table still gets a column instead of vanishing from the header;
    # BENIGN_SCENARIO so "which column is the benign one" is a config decision
    # rather than a literal 0 in a format string. The report is now the union
    # of these and the values actually in the table -- see `scenario_columns`.
    "splitting/split.py": [
        "BENIGN_SCENARIO", "DATASET_PATH", "DATASET_SAMPLE_PATH",
        "LABEL_COLUMN", "SCENARIOS", "SCENARIO_COLUMN", "SEEDS", "SPLIT",
        "SPLITS_PATH", "SPLITS_SAMPLE_PATH", "SplitFractions", "USER_COLUMN",
        "ensure_directories",
    ],
    "splitting/test_split.py": [
        "LABEL_COLUMN", "SCENARIO_COLUMN", "SEEDS", "SPLIT", "SplitFractions",
        "USER_COLUMN",
    ],
    "modeling/data.py": [
        "DATASET_PATH", "DATASET_SAMPLE_PATH", "DAY_COLUMN", "FEATURE_NAMES",
        "LABEL_COLUMN", "N_FEATURES", "SCENARIO_COLUMN", "SPLITS_PATH",
        "SPLITS_SAMPLE_PATH", "STD_FLOOR", "USER_COLUMN", "WINDOW_DAYS",
        "Z_CLIP",
    ],
    "modeling/metrics.py": [
        "BENIGN_SCENARIO", "LEAKAGE_AUPRC", "MIN_DISTINCT_PREDICTIONS",
        "PRECISION_AT_K", "SCENARIOS", "THRESHOLD_SWEEP_POINTS",
    ],
    "modeling/models.py": [
        "DROPOUT", "MODEL_A_HIDDEN", "N_FEATURES", "WINDOW_DAYS",
    ],
    "modeling/run_model_a.py": [
        "BATCH_SIZE", "DEVICE_CHOICES", "DEVICE_PREFERENCE", "LEARNING_RATE",
        "MAX_EPOCHS", "MODEL_A_HIDDEN", "MODEL_A_NAME", "SEEDS",
        "resolve_device",
    ],
    "modeling/run_model_b.py": [
        "BATCH_SIZE", "DEVICE_CHOICES", "DEVICE_PREFERENCE", "LEARNING_RATE",
        "MAX_EPOCHS", "MODEL_B_NAME", "N_FEATURES", "SEEDS", "WINDOW_DAYS",
        "resolve_device",
    ],
    # SCENARIOS added by brief R3. `summarise` re-aggregated per-scenario
    # recall through a hardcoded (1, 2, 3) and dropped scenario 4 on its way
    # into the metrics JSON -- the only one of the four sites that lost data
    # rather than display. Nothing in the training path was touched.
    "modeling/runner.py": [
        "LEAKAGE_AUPRC", "PRECISION_AT_K", "SCENARIOS", "SEEDS", "WINDOW_DAYS",
        "checkpoint_path", "ensure_directories", "figure_path", "metrics_path",
    ],
    "modeling/test_model_b.py": [
        "FEATURE_NAMES", "LABEL_COLUMN", "N_FEATURES", "USER_COLUMN",
        "WINDOW_DAYS",
    ],
    "modeling/test_modeling.py": [
        "FEATURE_NAMES", "LABEL_COLUMN", "MIN_DISTINCT_PREDICTIONS",
        "N_FEATURES", "SCENARIO_COLUMN", "SPLITS_SAMPLE_PATH", "USER_COLUMN",
        "WINDOW_DAYS", "Z_CLIP",
    ],
    "modeling/train.py": [
        "BATCH_SIZE", "EARLY_STOP_PATIENCE", "LEARNING_RATE", "MAX_EPOCHS",
    ],
    "evaluation/checkpoints.py": [
        "CHECKPOINT_DIR", "MODEL_A_NAME", "MODEL_B_NAME", "WINDOW_DAYS",
    ],
    "evaluation/corpus.py": [
        "DAY_COLUMN", "FEATURE_NAMES", "LABEL_COLUMN", "SCENARIO_COLUMN",
        "USER_COLUMN", "WINDOW_DAYS",
    ],
    "evaluation/falsepositives.py": ["BENIGN_SCENARIO", "FEATURE_NAMES"],
    # Three names added by brief R3. SCENARIOS for the bars, SCENARIO_PROFILES
    # for the axis labels (which used to hard-type r4.2's day ranges), and
    # ACTIVE_RELEASE because the figure's title said "CERT r4.2" whatever it
    # was drawing.
    "evaluation/figures.py": [
        "ACTIVE_RELEASE", "FIGURE_DIR", "MODEL_A_NAME", "MODEL_B_NAME",
        "SCENARIOS", "SCENARIO_PROFILES",
    ],
    "evaluation/guards.py": ["LEAKAGE_AUPRC"],
    "evaluation/phase1.py": [
        "MODEL_A_NAME", "MODEL_B_NAME", "OUTPUT_DIR", "SCENARIOS", "SEEDS",
    ],
    # Four names added by brief R8, the day r5.2 became SUPPORTED. This module
    # wrote r4.2's name, r4.2's population and r4.2's findings whatever release
    # it was rendering. ACTIVE_RELEASE, EXPECTED_INSIDER_COUNT and
    # EXPECTED_EMPLOYEE_COUNT parameterise the release-varying facts;
    # RELEASE_R42 is what `_only_on_r42` compares against, so that "is this an
    # r4.2 measurement" is a config value rather than a literal in a frozen
    # package. Nothing new is measured -- the positive rate now comes off
    # `random_baseline`, which the report already loaded.
    # SCENARIO_PROFILES added by brief R9. §2's table was built from a
    # hardcoded three-key dict of scenario names, so it described r4.2's day
    # ranges whatever release it rendered and dropped r5.2's scenario 4 -- 56%
    # of that release's attack days -- out of the report entirely. The same
    # constant already backs `evaluation/figures.py`'s axis labels, which is
    # where this defect was fixed the first time.
    "evaluation/report.py": [
        "ACTIVE_RELEASE", "EXPECTED_EMPLOYEE_COUNT", "EXPECTED_INSIDER_COUNT",
        "FINAL_RESULTS_PATH", "MAX_EPOCHS", "MODEL_A_NAME", "MODEL_B_NAME",
        "OUTPUT_DIR", "RELEASE_R42", "SCENARIO_PROFILES", "STAGE7_REPORT_PATH",
        "metrics_path",
    ],
    "evaluation/run_b7.py": [
        "DEVICE_CHOICES", "DEVICE_PREFERENCE", "MODEL_B_NAME", "SEEDS",
        "metrics_path", "resolve_device",
    ],
    "evaluation/run_phase1.py": [
        "DEVICE_CHOICES", "DEVICE_PREFERENCE", "SEEDS", "resolve_device",
    ],
    "evaluation/run_phase2.py": [
        "DEVICE_CHOICES", "DEVICE_PREFERENCE", "SEEDS", "resolve_device",
    ],
    "evaluation/run_stage7.py": ["ensure_directories"],
    "evaluation/scoring.py": ["BATCH_SIZE"],
    "evaluation/test_evaluation.py": [
        "DAY_COLUMN", "FEATURE_NAMES", "LABEL_COLUMN", "MODEL_A_NAME",
        "MODEL_B_NAME", "N_FEATURES", "SCENARIO_COLUMN", "USER_COLUMN",
        "WINDOW_DAYS",
    ],
    # Four names added by brief R7, and each replaces something worse.
    # TEST_SPLIT_OPENED_PATH replaces a second, local definition of a path
    # config already owns -- the defect R1 found in report.py. WINDOW_DAYS and
    # SCENARIO_COLUMN are what the shape guard needs to derive its expectation
    # from splits.json instead of from three hand-typed r4.2 constants, which
    # are gone. ACTIVE_RELEASE is the test-results title, which said "full
    # r4.2" whatever release it was printing.
    "evaluation/testpass.py": [
        "ACTIVE_RELEASE", "MODEL_A_NAME", "MODEL_B_NAME", "OUTPUT_DIR",
        "SCENARIOS", "SCENARIO_COLUMN", "SEEDS", "TEST_SPLIT_OPENED_PATH",
        "WINDOW_DAYS",
    ],
}

_FROZEN_PACKAGES = (
    "extraction", "labeling", "baseline", "splitting", "modeling", "evaluation",
)


def _python_files(package: str) -> list[Path]:
    return [
        path
        for path in sorted((ROOT / package).rglob("*.py"))
        if "salvage" not in path.parts and "__pycache__" not in path.parts
    ]


def _config_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "config":
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names += ["<module>" for a in node.names if a.name == "config"]
    return sorted(names)


def test_no_pipeline_package_changed_what_it_imports_from_config() -> None:
    """§9.5, as a test rather than a grep somebody has to remember to run."""
    found = {
        path.relative_to(ROOT).as_posix(): names
        for package in _FROZEN_PACKAGES
        for path in _python_files(package)
        if (names := _config_imports(path))
    }
    assert found == _EXPECTED_CONFIG_IMPORTS


def test_every_name_those_packages_import_still_exists() -> None:
    """The snapshot proves nothing moved. This proves nothing broke."""
    wanted = sorted({
        name
        for names in _EXPECTED_CONFIG_IMPORTS.values()
        for name in names
        if name != "<module>"
    })
    body = (
        "import json, config\n"
        f"print(json.dumps([n for n in {wanted!r} if not hasattr(config, n)]))"
    )
    assert _probe(None, body) == []
    assert _probe("5.2", body) == []


def _code_string_literals(path: Path) -> list[str]:
    """Every string literal in the file that is not a comment or a docstring.

    Parsing rather than grepping, for one reason: reading an environment
    variable needs the name as a runtime value, while *explaining* the variable
    in a comment or a docstring is exactly what good code does. A text search
    cannot tell those apart and would punish the second. Comments never reach
    the AST at all; docstrings do, so they are filtered out here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_only_config_reads_the_release_variable() -> None:
    """One reader, one decision. Every other module asks ``config``.

    A stage that read the environment itself could disagree with the paths it
    was handed, which is the failure this whole design exists to prevent. Test
    files are exempt: setting the variable is exactly what they are for.

    ``pipeline`` was added on 2026-08-22. Brief S1 moved four modules --
    ``detect``, ``store``, ``inference``, ``results`` -- out of ``ui/`` and
    forbade touching this file, so for six days they were audited by nothing.
    The package is listed here and NOT in ``_FROZEN_PACKAGES`` or
    ``_EXPECTED_CONFIG_IMPORTS``: those two snapshot the six frozen stage
    packages, and freezing a week-old package's config imports would be a
    bug wearing a safeguard's clothes.
    """
    offenders = [
        path.relative_to(ROOT).as_posix()
        for package in (*_FROZEN_PACKAGES, "ui", "pipeline")
        for path in _python_files(package)
        if not path.name.startswith("test_")
        and any(
            "INNERSIGHT_RELEASE" in literal
            for literal in _code_string_literals(path)
        )
    ]
    assert offenders == [], (
        f"only config.py may read INNERSIGHT_RELEASE; also found in {offenders}"
    )
