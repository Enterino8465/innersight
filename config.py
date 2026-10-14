"""Every constant InnerSight uses.

No other module in this project defines a path, a threshold or a magic number.
If a number needs to change, it changes here once and every stage sees it.

The one rule worth stating twice: ``FEATURE_NAMES`` is the vector layout.
It is the CNN's channel order and the column order of every saved checkpoint.
Reordering it raises no error -- it silently makes every stored artifact score
the wrong feature.

**This file serves two CERT releases.** ``INNERSIGHT_RELEASE`` picks one --
``"4.2"`` when the variable is unset, ``"5.2"`` when it says so -- and every
constant below that differs between them is *derived from the selected
profile*. Nothing else in the project knows a release exists: ``baseline`` still
does ``from config import SCENARIOS`` and still gets ``(1, 2, 3)``, or
``(1, 2, 3, 4)`` when the variable says 5.2.

**With the variable unset, every constant and every function result in this
file is byte-identical to what it was before releases were introduced.** That
is not a nicety. The frozen r4.2 numbers -- Model A AUPRC 0.1903 +/- 0.0300,
Model B 0.6663 +/- 0.1277, from the single test pass on 2026-08-21 -- describe
the code in this repository. If r4.2's behaviour moves by one path or one
integer, those numbers stop describing the code. ``test_config.py`` asserts the
r4.2 values as literals, never as something recomputed from the profile, for
exactly this reason: a test that derives its expectation from the thing it is
testing passes even when the thing is wrong.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent


_ENV_FULL: Final[str] = "INNERSIGHT_DATASET_FULL"
_ENV_SAMPLE: Final[str] = "INNERSIGHT_DATASET_SAMPLE"
_ENV_ANSWERS: Final[str] = "INNERSIGHT_ANSWERS_DIR"
_ENV_RELEASE: Final[str] = "INNERSIGHT_RELEASE"

DATASET_FULL: Final[Path] = Path("/Volumes/dataset hard drive/dataset/r4.2")
DATASET_FULL_MIRROR: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/r4.2 copy"
)
DATASET_SAMPLE: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/r4.2_sample"
)
ANSWERS_DIR: Final[Path] = Path("/Volumes/dataset hard drive/dataset/answers")
ANSWERS_DIR_MIRROR: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/answers"
)

DATASET_R52: Final[Path] = Path("/Volumes/dataset hard drive/dataset/r5.2")
DATASET_R52_MIRROR: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/r5.2"
)
DATASET_R52_SAMPLE: Final[Path] = Path(
    "/Users/michaelkuksov/Developer/innersight_dataset/r5.2_sample"
)


@dataclass(frozen=True, slots=True)
class LogSpec:
    """The contract for one raw CSV: what it is called and what is inside it.

    ``expected_sample_rows`` is the row count in the 30-user sample release and
    is used only as a checksum -- a mismatch means the sample was rebuilt, not
    that anything is broken.

    It is ``None`` when no sample has been built and counted yet, which is the
    state r5.2 is in until ``tools/build_r52_sample.py`` has been run once. A
    fabricated checksum is worse than no checksum: it would fail on a correct
    sample and teach whoever sees it to ignore the check.
    """

    filename: str
    columns: tuple[str, ...]
    expected_sample_rows: int | None = None

    @property
    def name(self) -> str:
        return self.filename.removesuffix(".csv")


_LOG_SPECS_R42: Final[tuple[LogSpec, ...]] = (
    LogSpec(
        filename="logon.csv",
        columns=("id", "date", "user", "pc", "activity"),
        expected_sample_rows=67_953,
    ),
    LogSpec(
        filename="device.csv",
        columns=("id", "date", "user", "pc", "activity"),
        expected_sample_rows=26_342,
    ),
    LogSpec(
        filename="file.csv",
        columns=("id", "date", "user", "pc", "filename", "content"),
        expected_sample_rows=26_222,
    ),
    LogSpec(
        filename="email.csv",
        columns=(
            "id", "date", "user", "pc", "to", "cc", "bcc",
            "from", "size", "attachments", "content",
        ),
        expected_sample_rows=89_577,
    ),
    LogSpec(
        filename="http.csv",
        columns=("id", "date", "user", "pc", "url", "content"),
        expected_sample_rows=967_805,
    ),
)

_LOG_SPECS_R52: Final[tuple[LogSpec, ...]] = (
    LogSpec(
        filename="logon.csv",
        columns=("id", "date", "user", "pc", "activity"),
        expected_sample_rows=30_574,
    ),
    LogSpec(
        filename="device.csv",
        columns=("id", "date", "user", "pc", "file_tree", "activity"),
        expected_sample_rows=19_446,
    ),
    LogSpec(
        filename="file.csv",
        columns=(
            "id", "date", "user", "pc", "filename", "activity",
            "to_removable_media", "from_removable_media", "content",
        ),
        expected_sample_rows=8_439,
    ),
    LogSpec(
        filename="email.csv",
        columns=(
            "id", "date", "user", "pc", "to", "cc", "bcc", "from",
            "activity", "size", "attachments", "content",
        ),
        expected_sample_rows=237_495,
    ),
    LogSpec(
        filename="http.csv",
        columns=("id", "date", "user", "pc", "url", "content"),
        expected_sample_rows=871_706,
    ),
)


RELEASE_R42: Final[str] = "4.2"
RELEASE_R52: Final[str] = "5.2"

RELEASE_LOG_COLUMNS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    RELEASE_R42: {spec.name: spec.columns for spec in _LOG_SPECS_R42},
    RELEASE_R52: {spec.name: spec.columns for spec in _LOG_SPECS_R52},
}

RELEASE_SUPPORTED: Final[str] = "SUPPORTED"
RELEASE_NOT_YET_ALIGNED: Final[str] = "NOT_YET_ALIGNED"
RELEASE_CAPABILITY: Final[dict[str, str]] = {
    RELEASE_R42: RELEASE_SUPPORTED,
    RELEASE_R52: RELEASE_SUPPORTED,
}


class UnknownReleaseError(ValueError):
    """Raised when ``INNERSIGHT_RELEASE`` names a release with no profile."""


@dataclass(frozen=True, slots=True)
class ScenarioProfile:
    """One attack scenario: what shape it is, and how long its windows run.

    This exists because ``evaluation/figures.py`` used to hard-type the day
    ranges into its axis labels, which made the figure silently describe r4.2
    whichever release drew it. The ranges are a property of the release, so
    they live on the release profile like everything else that varies.

    ``kind`` is the one word that tells a reader how to read the bar next to
    it: a two-day burst and a two-month drift are not the same detection
    problem and the chart should not pretend they are.
    """

    kind: str
    min_days: int
    max_days: int

    def __post_init__(self) -> None:
        if self.min_days < 1 or self.max_days < self.min_days:
            raise ValueError(
                f"scenario window range {self.min_days}..{self.max_days} is not "
                f"a run of at least one day"
            )


_SCENARIO_PROFILES_R42: Final[dict[int, ScenarioProfile]] = {
    1: ScenarioProfile(kind="burst", min_days=1, max_days=12),
    2: ScenarioProfile(kind="drift", min_days=45, max_days=59),
    3: ScenarioProfile(kind="burst", min_days=2, max_days=2),
}

_SCENARIO_PROFILES_R52: Final[dict[int, ScenarioProfile]] = {
    1: ScenarioProfile(kind="burst", min_days=1, max_days=10),
    2: ScenarioProfile(kind="drift", min_days=44, max_days=60),
    3: ScenarioProfile(kind="burst", min_days=2, max_days=2),
    4: ScenarioProfile(kind="escalation", min_days=33, max_days=161),
}


@dataclass(frozen=True, slots=True)
class ReleaseProfile:
    """Everything about one CERT release that the pipeline has to know.

    The path properties are computed rather than stored so that a release's
    layout is stated once, as a rule, instead of six times as literals. Do not
    "optimise" them into fields: the rule *is* the documentation.
    """

    release_id: str
    scenario_profiles: dict[int, ScenarioProfile]
    expected_insider_count: int
    expected_employee_count: int
    release_root: Path
    full_candidates: tuple[Path, ...]
    sample_dir: Path
    log_specs: tuple[LogSpec, ...]

    @property
    def scenarios(self) -> tuple[int, ...]:
        """The attack scenario ids, ascending. Benign (0) is not one of them."""
        return tuple(sorted(self.scenario_profiles))

    @property
    def log_spec_map(self) -> dict[str, LogSpec]:
        """The specs keyed by short name -- "logon", "device", ... ."""
        return {spec.name: spec for spec in self.log_specs}

    @property
    def data_dir(self) -> Path:
        return self.release_root / "data"

    @property
    def output_dir(self) -> Path:
        return self.release_root / "outputs"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def final_results_path(self) -> Path:
        return self.output_dir / "final_results.json"

    @property
    def test_split_opened_path(self) -> Path:
        return self.output_dir / "TEST_SPLIT_OPENED.json"

    @property
    def stage7_report_path(self) -> Path:
        return self.release_root / "STAGE7_REPORT.md"

    @property
    def answer_scenario_dirs(self) -> tuple[str, ...]:
        """The per-scenario answer folders, e.g. ``("r4.2-1", ..., "r4.2-3")``.

        Derived rather than listed because the naming rule is exact for every
        release in `answers/`, and a list would be one more thing to keep in
        step with `scenarios`.
        """
        return tuple(f"r{self.release_id}-{s}" for s in self.scenarios)


PROFILES: Final[dict[str, ReleaseProfile]] = {
    RELEASE_R42: ReleaseProfile(
        release_id=RELEASE_R42,
        scenario_profiles=_SCENARIO_PROFILES_R42,
        expected_insider_count=70,
        expected_employee_count=1_000,
        release_root=PROJECT_ROOT,
        full_candidates=(DATASET_FULL, DATASET_FULL_MIRROR),
        sample_dir=DATASET_SAMPLE,
        log_specs=_LOG_SPECS_R42,
    ),
    RELEASE_R52: ReleaseProfile(
        release_id=RELEASE_R52,
        scenario_profiles=_SCENARIO_PROFILES_R52,
        expected_insider_count=99,
        expected_employee_count=2_000,
        release_root=PROJECT_ROOT / "releases" / "r52",
        full_candidates=(DATASET_R52, DATASET_R52_MIRROR),
        sample_dir=DATASET_R52_SAMPLE,
        log_specs=_LOG_SPECS_R52,
    ),
}

DEFAULT_RELEASE: Final[str] = RELEASE_R42


def _select_profile() -> tuple[ReleaseProfile, bool]:
    """Read ``INNERSIGHT_RELEASE`` once and return its profile.

    Returns the profile and whether the environment variable chose it, so the
    banner can say "unset" rather than leaving a reader to guess whether the
    default was deliberate.

    A blank value counts as unset, matching how `_env_path` treats the other
    three variables. An unrecognised value raises: falling back to r4.2 would
    mean a typo'd `INNERSIGHT_RELEASE=52` silently wrote into the frozen r4.2
    tree, which is the exact failure this whole mechanism exists to prevent.
    """
    raw = os.environ.get(_ENV_RELEASE, "").strip()
    if not raw:
        return PROFILES[DEFAULT_RELEASE], False
    if raw not in PROFILES:
        supported = ", ".join(sorted(PROFILES))
        raise UnknownReleaseError(
            f"{_ENV_RELEASE}={raw!r} is not a release this project has a "
            f"profile for. Supported: {supported}. Unset the variable for "
            f"r{DEFAULT_RELEASE}, which is the default and the release the "
            f"frozen results describe."
        )
    return PROFILES[raw], True


_ACTIVE, _FROM_ENV = _select_profile()

ACTIVE_RELEASE: Final[str] = _ACTIVE.release_id
RELEASE_SELECTED_BY_ENV: Final[bool] = _FROM_ENV


RELEASE_ROOT: Final[Path] = _ACTIVE.release_root
DATA_DIR: Final[Path] = _ACTIVE.data_dir
OUTPUT_DIR: Final[Path] = _ACTIVE.output_dir
CHECKPOINT_DIR: Final[Path] = _ACTIVE.checkpoint_dir
FIGURE_DIR: Final[Path] = _ACTIVE.figure_dir

DAILY_COUNTS_PATH: Final[Path] = DATA_DIR / "daily_counts.parquet"
DAILY_COUNTS_SAMPLE_PATH: Final[Path] = DATA_DIR / "daily_counts.sample.parquet"
LABELLED_COUNTS_PATH: Final[Path] = DATA_DIR / "labelled_counts.parquet"
LABELLED_COUNTS_SAMPLE_PATH: Final[Path] = DATA_DIR / "labelled_counts.sample.parquet"
DATASET_PATH: Final[Path] = DATA_DIR / "dataset.parquet"
DATASET_SAMPLE_PATH: Final[Path] = DATA_DIR / "dataset.sample.parquet"
SPLITS_PATH: Final[Path] = DATA_DIR / "splits.json"
SPLITS_SAMPLE_PATH: Final[Path] = DATA_DIR / "splits.sample.json"
MODEL_A_NAME: Final[str] = "model_a"
MODEL_B_NAME: Final[str] = "model_b"


FEATURE_NAMES: Final[tuple[str, ...]] = (
    "logon_count",
    "logoff_count",
    "after_hours_logons",
    "weekend_logons",
    "unique_pcs_used",
    "usb_connect_count",
    "usb_disconnect_count",
    "after_hours_usb",
    "file_count",
    "file_to_removable_count",
    "unique_filenames",
    "email_sent_count",
    "email_to_external_count",
    "large_attachment_count",
    "total_email_size",
    "http_request_count",
    "job_search_visits",
    "cloud_upload_visits",
)

N_FEATURES: Final[int] = len(FEATURE_NAMES)
FEATURE_INDEX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(FEATURE_NAMES)
}


DEGENERATE_FEATURES: Final[dict[str, dict[str, str]]] = {
    RELEASE_R42: {
        "file_to_removable_count": (
            "equals file_count on every row (445,581 of 445,581), because "
            "r4.2 has no to_removable_media column: every file.csv row there "
            "already is a copy to removable media"
        ),
        "unique_filenames": (
            "equals file_count (445,581 of 445,581), because CERT generates "
            "one fresh random filename per event, so the distinct count is "
            "the row count"
        ),
        "large_attachment_count": (
            "identically zero, because r4.2's largest of 2,629,979 mails is "
            "141,909 bytes against a 1,000,000-byte threshold; baseline.py's "
            "std floor falls back to 1.000 for it, the feature being constant "
            "for every user"
        ),
    },
    RELEASE_R52: {},
}

USER_COLUMN: Final[str] = "user"
DAY_COLUMN: Final[str] = "day"
LABEL_COLUMN: Final[str] = "label"
SCENARIO_COLUMN: Final[str] = "user_scenario"
BENIGN_SCENARIO: Final[int] = 0
SCENARIOS: Final[tuple[int, ...]] = _ACTIVE.scenarios
SCENARIO_PROFILES: Final[dict[int, ScenarioProfile]] = _ACTIVE.scenario_profiles


DATE_FORMAT: Final[str] = "%m/%d/%Y %H:%M:%S"
INTERNAL_DOMAIN: Final[str] = "dtaa.com"

LOG_SPECS: Final[dict[str, LogSpec]] = _ACTIVE.log_spec_map
LOG_NAMES: Final[tuple[str, ...]] = tuple(LOG_SPECS)

LOGON_ACTIVITY: Final[str] = "Logon"
LOGOFF_ACTIVITY: Final[str] = "Logoff"
DEVICE_CONNECT: Final[str] = "Connect"
DEVICE_DISCONNECT: Final[str] = "Disconnect"

EMAIL_SEND_ACTIVITY: Final[str] = "Send"

LARGE_ATTACHMENT_BYTES: Final[int] = 1_000_000


INSIDERS_FILENAME: Final[str] = "insiders.csv"
INSIDERS_COLUMNS: Final[tuple[str, ...]] = (
    "dataset", "scenario", "details", "user", "start", "end",
)

RELEASE_ID: Final[str] = _ACTIVE.release_id

INSIDERS_DATE_FORMAT: Final[str] = DATE_FORMAT

EXPECTED_INSIDER_COUNT: Final[int] = _ACTIVE.expected_insider_count
EXPECTED_EMPLOYEE_COUNT: Final[int] = _ACTIVE.expected_employee_count
ANSWER_SCENARIO_DIRS: Final[tuple[str, ...]] = _ACTIVE.answer_scenario_dirs

FULL_DATASET_CANDIDATES: Final[tuple[Path, ...]] = _ACTIVE.full_candidates
SAMPLE_DATASET_DIR: Final[Path] = _ACTIVE.sample_dir

BUSINESS_HOUR_START: Final[int] = 7
BUSINESS_HOUR_END: Final[int] = 19

CHUNK_SIZE: Final[int] = 50_000

WEEKEND_WEEKDAYS: Final[tuple[int, int]] = (5, 6)


JOB_SEARCH_DOMAINS: Final[tuple[str, ...]] = (
    "careerbuilder.com",
    "monster.com",
    "jobhuntersbible.com",
    "job-hunt.org",
    "indeed.com",
    "simplyhired.com",
    "aol.com/jobs",
)


CLOUD_UPLOAD_DOMAINS: Final[tuple[str, ...]] = (
    "wikileaks.org",
    "dropbox.com",
    "4shared.com",
)


WINDOW_DAYS: Final[int] = 28
MIN_HISTORY_DAYS: Final[int] = 30

Z_CLIP: Final[float] = 10.0
STD_FLOOR: Final[float] = 1e-6


STD_FLOOR_MULTIPLIER: Final[float] = 0.1

STD_FLOOR_ACTIVE_USERS_ONLY: Final[bool] = True

STD_FLOOR_FALLBACK: Final[float] = 1.0

VALIDATION_Z_THRESHOLD: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class SplitFractions:
    """Train/val/test proportions. Split by USER, never by day -- the same
    person's Monday must not train a model that is then tested on their Tuesday.
    """

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"split fractions must sum to 1.0, got {total!r} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )
        for name, value in (
            ("train", self.train), ("val", self.val), ("test", self.test)
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"split fraction {name} must be strictly between 0 and 1, "
                    f"got {value!r}"
                )


SPLIT: Final[SplitFractions] = SplitFractions(train=0.6, val=0.2, test=0.2)

SEEDS: Final[tuple[int, ...]] = (42, 123, 456)

BATCH_SIZE: Final[int] = 512
LEARNING_RATE: Final[float] = 1e-3
MAX_EPOCHS: Final[int] = 50
EARLY_STOP_PATIENCE: Final[int] = 7

MODEL_A_HIDDEN: Final[int] = 32
DROPOUT: Final[float] = 0.0

PRECISION_AT_K: Final[tuple[int, ...]] = (10, 50)
THRESHOLD_SWEEP_POINTS: Final[int] = 512
LEAKAGE_AUPRC: Final[float] = 0.95
MIN_DISTINCT_PREDICTIONS: Final[int] = 1_000


DEVICE_CHOICES: Final[tuple[str, ...]] = ("auto", "cpu", "mps")
DEVICE_PREFERENCE: Final[str] = "auto"
MPS_FALLBACK_ENV: Final[str] = "PYTORCH_ENABLE_MPS_FALLBACK"
os.environ.setdefault(MPS_FALLBACK_ENV, "1")

MIN_PYTHON: Final[tuple[int, int]] = (3, 11)


_ANNOUNCED: bool = False


def release_banner() -> str:
    """The one line every run prints, naming the release and where it writes."""
    source = (
        f"{_ENV_RELEASE}={ACTIVE_RELEASE}"
        if RELEASE_SELECTED_BY_ENV
        else f"{_ENV_RELEASE} unset"
    )
    try:
        data = DATA_DIR.relative_to(PROJECT_ROOT)
        outputs = OUTPUT_DIR.relative_to(PROJECT_ROOT)
    except ValueError:
        data, outputs = DATA_DIR, OUTPUT_DIR
    return f"release {ACTIVE_RELEASE} ({source}) -- {data}/ {outputs}/"


def _announce_release() -> None:
    """Log the banner, once per process. Idempotent and cheap to call."""
    global _ANNOUNCED
    if _ANNOUNCED:
        return
    _ANNOUNCED = True
    logger.info("%s", release_banner())
    if RELEASE_CAPABILITY.get(ACTIVE_RELEASE) != RELEASE_SUPPORTED:
        logger.warning(
            "r%s is %s: the feature computations in extraction/, labeling/ and "
            "baseline/ were written against r%s and have not been re-verified "
            "against this release. Anything this run produces is a smoke test, "
            "not a result.",
            ACTIVE_RELEASE,
            RELEASE_CAPABILITY.get(ACTIVE_RELEASE, "unknown to RELEASE_CAPABILITY"),
            DEFAULT_RELEASE,
        )


def child_environment(release: str | None = None) -> dict[str, str]:
    """This process's environment, plus the release a child process must be.

    A copy of ``os.environ`` with the release variable set to `release`, ready
    to hand to ``subprocess.Popen(env=...)``. The one way for a parent process
    to tell a child which release it is running as.

    **Why it lives here.** The variable's name is this module's business and
    nobody else's -- every other package asks `config` -- which is what
    `test_config.py`'s `test_only_config_reads_the_release_variable` enforces.
    A caller that has to launch a child as a particular release cannot name
    the variable, so this hands it the whole environment instead.

    **It COPIES the environment; it does not build one.**
    ``subprocess.Popen(env=...)`` REPLACES the child's environment rather than
    merging into it, so returning a one-key mapping would launch a child with
    no ``PATH``, no ``HOME`` and no ``VIRTUAL_ENV`` -- a failure that looks
    nothing like the one this function exists to stop.

    ``release=None`` means "this child has no release": nothing is added and
    nothing is removed, so the child inherits exactly what this process has.
    That is the honest answer for ``pipeline.score --sample``, which reads the
    frozen sample tables and never opens a release folder. The variable is
    never set to ``None`` or to ``""``: `_select_profile` reads a blank value
    as unset, so an empty string would quietly turn a chosen release back into
    the default, and ``None`` would not survive `Popen` at all.

    Written on 2026-08-22, after the first r5.2 training run launched from the
    web UI died inside ``extraction.extract`` with "device.csv is not r4.2's
    device log". `pipeline.train` had read the folder's five header lines and
    knew it was r5.2; it then launched its six stages with the environment it
    had inherited, and the UI's own process had no release chosen, so every
    stage fell back to r4.2 and read r5.2's files under r4.2's column
    contract. The parent knew and the children did not.
    """
    environment = dict(os.environ)
    if release is not None:
        if release not in PROFILES:
            supported = ", ".join(sorted(PROFILES))
            raise UnknownReleaseError(
                f"cannot run a child process as release {release!r}: this "
                f"project has no profile for it. Supported: {supported}."
            )
        environment[_ENV_RELEASE] = release
    return environment


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no candidate location for a dataset directory exists."""


def _first_existing(
    label: str, candidates: tuple[Path | None, ...]
) -> Path:
    tried = [c for c in candidates if c is not None]
    for candidate in tried:
        if candidate.is_dir():
            return candidate
    listing = "\n".join(f"  - {p}" for p in tried)
    raise DatasetNotFoundError(
        f"{label} not found. Expected a directory at one of:\n{listing}\n"
        f"If the external drive is unplugged, plug it in, or set the "
        f"environment variable for this location and re-run."
    )


def _env_path(variable: str) -> Path | None:
    raw = os.environ.get(variable)
    return Path(raw).expanduser() if raw else None


def resolve_dataset(sample: bool = True) -> Path:
    """Return the directory holding the five raw CSVs.

    ``sample=True`` gives the 30-user slice that development runs against;
    ``sample=False`` gives the full release, preferring the external drive and
    falling back to the local mirror.

    Both are the ACTIVE release's. INNERSIGHT_DATASET_FULL and
    INNERSIGHT_DATASET_SAMPLE still win over the built-in locations, and they
    now override whichever release is active -- "point the pipeline at this
    folder" rather than "point r4.2 at this folder". With INNERSIGHT_RELEASE
    unset, which is the default, this is exactly the old behaviour.
    """
    _announce_release()
    if sample:
        return _first_existing(
            f"Sample dataset (${_ENV_SAMPLE})",
            (_env_path(_ENV_SAMPLE), _ACTIVE.sample_dir),
        )
    return _first_existing(
        f"Full r{ACTIVE_RELEASE} dataset (${_ENV_FULL})",
        (_env_path(_ENV_FULL), *_ACTIVE.full_candidates),
    )


def resolve_answers() -> Path:
    """Return the directory holding the ground-truth answer files.

    Not release-scoped, and it must not become so: `answers/insiders.csv` holds
    all ten CERT releases in one file and is filtered by RELEASE_ID at read
    time. One directory, one file, ten releases.
    """
    return _first_existing(
        f"Answers directory (${_ENV_ANSWERS})",
        (_env_path(_ENV_ANSWERS), ANSWERS_DIR, ANSWERS_DIR_MIRROR),
    )


def insiders_csv_path() -> Path:
    """Return the path to the ground-truth window file.

    A function rather than a module-level constant because the answers
    directory may live on an external drive that is not mounted at import
    time; resolving eagerly would make importing ``config`` fail on a laptop
    with the drive unplugged.
    """
    return resolve_answers() / INSIDERS_FILENAME


def log_path(name: str, sample: bool = True) -> Path:
    """Return the path to one raw log, e.g. ``log_path("http", sample=True)``."""
    if name not in LOG_SPECS:
        raise KeyError(
            f"unknown log {name!r}; expected one of {', '.join(LOG_NAMES)}"
        )
    return resolve_dataset(sample=sample) / LOG_SPECS[name].filename


def external_drive_mounted() -> bool:
    """Whether the full dataset is reachable. Information, not an error --
    development happens on the sample."""
    return _ACTIVE.full_candidates[0].is_dir()


def ensure_directories() -> None:
    """Create the generated-output directories if they are missing."""
    _announce_release()
    for directory in (DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _sample_infix(sample: bool) -> str:
    return ".sample" if sample else ""


def checkpoint_path(model: str, seed: int, sample: bool = False) -> Path:
    """Where one trained model's weights, threshold and standardiser live.

    e.g. ``outputs/checkpoints/model_a_seed42.pt``. A checkpoint carries the
    standardiser's mean and std with it: a model whose input scaling lives in
    a different file is not reproducible, and Stage 7 must apply exactly the
    transform the weights were trained under.
    """
    return CHECKPOINT_DIR / f"{model}_seed{seed}{_sample_infix(sample)}.pt"


def metrics_path(model: str, sample: bool = False, tag: str = "") -> Path:
    """Where the per-seed and aggregated validation metrics land.

    ``tag`` names a variant run so the ablation cannot overwrite the real
    result: ``metrics_path("model_a", tag="nopw")`` is a different file.
    """
    suffix = f"_{tag}" if tag else ""
    return OUTPUT_DIR / f"{model}{suffix}_metrics{_sample_infix(sample)}.json"


def figure_path(model: str, sample: bool = False, tag: str = "") -> Path:
    """Where the training-curve figure lands."""
    suffix = f"_{tag}" if tag else ""
    return FIGURE_DIR / f"{model}{suffix}_curves{_sample_infix(sample)}.png"


def resolve_device(preference: str = DEVICE_PREFERENCE) -> str:
    """Return the torch device string to train on.

    ``"auto"`` prefers MPS and falls back to CPU; an explicit ``"mps"`` on a
    machine without Metal is a mistake worth an exception rather than a silent
    downgrade, because the whole point of asking for MPS is to test it.

    torch is imported inside the function: ``config`` is imported by stages
    that have nothing to do with modelling, and importing torch costs a couple
    of seconds every time.
    """
    _announce_release()
    if preference not in DEVICE_CHOICES:
        raise ValueError(
            f"device must be one of {', '.join(DEVICE_CHOICES)}, "
            f"got {preference!r}"
        )
    if preference == "cpu":
        return "cpu"

    import torch

    available = torch.backends.mps.is_available()
    if preference == "mps":
        if not available:
            raise RuntimeError(
                "device 'mps' was requested but torch.backends.mps."
                "is_available() is False. Use --device cpu, or --device auto "
                "to fall back automatically."
            )
        return "mps"
    return "mps" if available else "cpu"


STAGE7_REPORT_PATH: Final[Path] = _ACTIVE.stage7_report_path
FINAL_RESULTS_PATH: Final[Path] = _ACTIVE.final_results_path
TEST_SPLIT_OPENED_PATH: Final[Path] = _ACTIVE.test_split_opened_path

ALL_TEST_SPLIT_OPENED_PATHS: Final[tuple[Path, ...]] = tuple(
    profile.test_split_opened_path for profile in PROFILES.values()
)
ALL_FINAL_RESULTS_PATHS: Final[tuple[Path, ...]] = tuple(
    profile.final_results_path for profile in PROFILES.values()
)
ALL_CHECKPOINT_DIRS: Final[tuple[Path, ...]] = tuple(
    profile.checkpoint_dir for profile in PROFILES.values()
)


MODEL_STORE_DIR: Final[Path] = PROJECT_ROOT / "models"
UI_RUN_DIR: Final[Path] = PROJECT_ROOT / "runs"

UI_DATASET_CHOICES: Final[dict[str, tuple[Path, ...]]] = {
    RELEASE_R42: (DATASET_FULL, DATASET_FULL_MIRROR),
    RELEASE_R52: (DATASET_R52, DATASET_R52_MIRROR),
}


def resolve_release_folder(release: str) -> Path | None:
    """Where one release's raw logs are, or None if none of its places exist.

    Unlike :func:`resolve_dataset` this does not raise. It backs a dropdown,
    and a drive that is currently unplugged is a thing to grey out and say so
    about, not an exception to serve a 500 page with. The environment variable
    still wins for the ACTIVE release, so a machine that is not Michael's
    laptop can point the menu somewhere else without editing this file.
    """
    if release not in UI_DATASET_CHOICES:
        raise KeyError(
            f"unknown release {release!r}; the menu offers "
            f"{', '.join(UI_DATASET_CHOICES)}"
        )
    override = _env_path(_ENV_FULL) if release == ACTIVE_RELEASE else None
    for candidate in (override, *UI_DATASET_CHOICES[release]):
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


UI_HOST: Final[str] = "127.0.0.1"
UI_PORT: Final[int] = 5001

UI_ALLOWED_COMMANDS: Final[tuple[str, ...]] = (
    "pipeline.train",
    "pipeline.score",
)

UI_REPRODUCTION_TOLERANCE: Final[float] = 5e-5

UI_JOB_TIMEOUT_SECONDS: Final[int] = 3_600

UI_LOG_BUFFER_LINES: Final[int] = 5_000
