"""
The leaker — never used USB or worked late, suddenly does both, uploads to wikileaks.org, resigns. 
A burst, 1–12 days.

The thief — browses job sites, contacts a competitor, 
and over weeks uses a thumb drive far more than they used to. A slow drift,
45–59 days. 88% of r4.2's attack days.

The rogue sysadmin — keylogger onto his supervisor's machine, 
logs in as him next day, sends an alarming mass email, leaves. 
2 days. Only 12 days total in your data — too few to measure.

The hunter — logs into other people's machines, hunts for files, 
mails them to a personal address, escalating over three months. 
33–161 days. r5.2 only. Model B's best result, ~90%.




Stage 2: daily counts -> the same counts on a complete calendar, labelled.

Two jobs, in this order, because the second depends on the first.

**Calendar fill.** Stage 1 emits a row only for a day that produced at least one
event, so its table is much shorter than one row per person per day. On r4.2 it
has 330,452 rows where 1,000 users x 501 days would be 501,000; on r5.2, 693,649
rows against 2,000 users x 517 days. Most of the gap is weekends, on both
releases. That is fine for Stage 1 and fatal later:
Model B reads the 28 days ending on day *d*, and if the rows are positional
rather than calendar-dense, "28 rows" quietly becomes anything from 28 to 40
calendar days depending on how many weekends the person skipped.

The deeper reason is that **a day of no activity is an observation, not missing
data**. The scenario-2 story is an employee who resigns and stops showing up;
their silence is the strongest thing they do. Positional windowing would delete
that signal by construction rather than fail to learn it.

The fill is per user, from that user's first observed day to their last -- never
to the global range. People are hired and terminated throughout the 17 months,
and expanding a user who left in month 4 out to month 17 would fabricate 300
days of absence that nothing in the world produced.

**Labelling.** ``answers/insiders.csv`` gives one attack window per insider as a
pair of timestamps. Two columns come out of it:

``label``           per DAY  -- is this day inside an attack window (0/1)
``user_scenario``   per USER -- 0 for benign, otherwise the person's scenario id

The scenario ids come from ``config.SCENARIOS`` and depend on the release: 1-3
on r4.2, 1-4 on r5.2. Nothing in this module is told how many there are.

They are separate because they answer different questions. An insider's ordinary
Tuesday in March is ``label=0`` but still ``user_scenario=2``: Stage 4 needs the
user-level value to stratify the split, and Stage 7 needs it to break results
down per scenario. Averaging over scenarios would hide how lopsided the mix is,
and it is lopsided differently in each release: on r4.2 scenario 2 is ~89% of
all positive days, on r5.2 the new scenario 4 is ~56% of them and scenario 2
drops to ~40%.

Two things this stage does NOT do: it drops no rows, and it computes no
z-scores. Both belong to Stage 3.

Run it from the project root::

    python -m labeling.label                      # full table
    python -m labeling.label --sample             # 30-user sample, ~1 s
    python -m labeling.label --user KPC0073       # inspect one user's window
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from config import (
    BENIGN_SCENARIO,
    DAILY_COUNTS_PATH,
    DAILY_COUNTS_SAMPLE_PATH,
    DAY_COLUMN,
    FEATURE_NAMES,
    INSIDERS_COLUMNS,
    INSIDERS_DATE_FORMAT,
    LABEL_COLUMN,
    LABELLED_COUNTS_PATH,
    LABELLED_COUNTS_SAMPLE_PATH,
    RELEASE_ID,
    SCENARIO_COLUMN,
    SCENARIOS,
    USER_COLUMN,
    ensure_directories,
    insiders_csv_path,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS: Final[tuple[str, str]] = (USER_COLUMN, DAY_COLUMN)
_RULE: Final[str] = "=" * 78

WINDOW_START: Final[str] = "start_day"
WINDOW_END: Final[str] = "end_day"


class GroundTruthError(ValueError):
    """The answer key is not shaped the way this stage requires."""


class LabellingError(AssertionError):
    """A labelled table violates an invariant that cannot be true of real data."""


def load_insiders(path: Path, release: str = RELEASE_ID) -> pd.DataFrame:
    """Read ``insiders.csv`` and return the windows for one release.

    Args:
        path: The answer file. It contains every CERT release, not just ours.
        release: The ``dataset`` value to keep, as a string.

    Returns:
        One row per attack window with columns ``user``, ``scenario`` (int),
        ``start_day`` and ``end_day`` (both midnight-normalised timestamps).

    Raises:
        FileNotFoundError: The answer file is missing.
        GroundTruthError: The file lacks an expected column, the release filter
            matched nothing, or one user is attributed to two scenarios.
    """
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth file not found: {path}")

    answers = pd.read_csv(path, dtype="str")

    missing = [c for c in INSIDERS_COLUMNS if c not in answers.columns]
    if missing:
        raise GroundTruthError(
            f"{path} is missing column(s) {missing}; expected "
            f"{list(INSIDERS_COLUMNS)} and found {list(answers.columns)}"
        )

    windows = answers.loc[answers["dataset"].str.strip() == release].copy()
    if windows.empty:
        found = sorted(answers["dataset"].dropna().unique())
        raise GroundTruthError(
            f"no rows in {path} have dataset == {release!r}. Values present: "
            f"{found}. Note this must be compared as a string."
        )

    parsed = pd.DataFrame(
        {
            USER_COLUMN: windows["user"].str.strip(),
            "scenario": windows["scenario"].astype("int64"),
            WINDOW_START: _to_day(windows["start"]),
            WINDOW_END: _to_day(windows["end"]),
        }
    ).reset_index(drop=True)

    _validate_windows(parsed, path)
    logger.info(
        "loaded %d attack window(s) for %d insider(s) in release %s",
        len(parsed), parsed[USER_COLUMN].nunique(), release,
    )
    return parsed


def _to_day(stamps: pd.Series) -> pd.Series:
    """Parse a window timestamp column down to midnight of its calendar day.

    Normalising here rather than at comparison time is what makes the labelling
    rule "day in [start_day, end_day]" literally true of the values, instead of
    true only if every later comparison remembers to strip the time.
    """
    return pd.to_datetime(
        stamps.str.strip(), format=INSIDERS_DATE_FORMAT
    ).dt.normalize()


def _validate_windows(windows: pd.DataFrame, path: Path) -> None:
    """Reject an answer key that cannot be turned into per-user scenarios."""
    reversed_rows = windows.loc[windows[WINDOW_END] < windows[WINDOW_START]]
    if not reversed_rows.empty:
        raise GroundTruthError(
            f"{path}: {len(reversed_rows)} window(s) end before they start, "
            f"first is {reversed_rows.iloc[0][USER_COLUMN]}"
        )

    unknown = sorted(set(windows["scenario"]) - set(SCENARIOS))
    if unknown:
        raise GroundTruthError(
            f"{path}: unknown scenario id(s) {unknown}; expected {list(SCENARIOS)}"
        )

    per_user = windows.groupby(USER_COLUMN)["scenario"].nunique()
    conflicted = sorted(per_user[per_user > 1].index)
    if conflicted:
        raise GroundTruthError(
            f"{path}: user(s) {conflicted} appear under more than one scenario, "
            f"so a single per-user scenario cannot be assigned"
        )

    duplicated = windows[USER_COLUMN].duplicated().sum()
    if duplicated:
        logger.warning(
            "%d user(s) have more than one attack window; windows will be unioned",
            duplicated,
        )


def calendar_fill(daily: pd.DataFrame) -> pd.DataFrame:
    """Reindex each user onto every calendar day they were observed across.

    Inserted days get 0 for all 18 features, in the original dtype. Rows that
    already existed are untouched, and no row is ever removed -- the output is a
    strict superset of the input.

    Raises:
        LabellingError: The input already contains duplicate ``(user, day)``
            pairs, which would silently multiply during the merge.
    """
    _require_unique_keys(daily, "daily counts")
    if daily.empty:
        return daily.copy()

    features = [name for name in FEATURE_NAMES if name in daily.columns]
    dtypes = {name: daily[name].dtype for name in features}

    complete = _calendar_index(daily)
    filled = complete.merge(daily, on=list(KEY_COLUMNS), how="left")

    for name in features:
        filled[name] = filled[name].fillna(0).astype(dtypes[name])

    ordered = filled[[*KEY_COLUMNS, *features]].sort_values(
        list(KEY_COLUMNS), ignore_index=True
    )
    logger.info("calendar fill: %d rows -> %d rows", len(daily), len(ordered))
    return ordered


def _calendar_index(daily: pd.DataFrame) -> pd.DataFrame:
    """Every ``(user, day)`` from each user's first observed day to their last.

    Built by arithmetic rather than by concatenating one ``date_range`` frame
    per user -- 1,000 of them on r4.2, 2,000 on r5.2. One ``repeat`` plus one
    ``arange`` is both faster and, more usefully, has no per-user branch that
    could get a boundary wrong for one person and not the others.
    """
    span = daily.groupby(USER_COLUMN, sort=True)[DAY_COLUMN].agg(["min", "max"])
    lengths = ((span["max"] - span["min"]).dt.days + 1).to_numpy()

    starts = span["min"].to_numpy()
    run_starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    offsets = np.arange(int(lengths.sum())) - run_starts

    return pd.DataFrame(
        {
            USER_COLUMN: pd.Series(
                np.repeat(span.index.to_numpy(), lengths), dtype=daily[USER_COLUMN].dtype
            ),
            DAY_COLUMN: np.repeat(starts, lengths) + offsets.astype("timedelta64[D]"),
        }
    )


def attach_labels(daily: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Add ``label`` (per day) and ``user_scenario`` (per user).

    Args:
        daily: One row per ``(user, day)``; calendar-filled or not.
        windows: The output of :func:`load_insiders`.

    Returns:
        A copy of *daily* with two ``int8`` columns appended. Row count, row
        order and every feature value are unchanged.
    """
    labelled = daily.copy()

    scenario_by_user = (
        windows.drop_duplicates(USER_COLUMN).set_index(USER_COLUMN)["scenario"]
    )
    labelled[SCENARIO_COLUMN] = (
        labelled[USER_COLUMN]
        .map(scenario_by_user)
        .fillna(BENIGN_SCENARIO)
        .astype("int8")
    )
    labelled[LABEL_COLUMN] = _window_mask(labelled, windows).astype("int8")
    return labelled


def _window_mask(daily: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """A boolean per row: does its day fall inside one of that user's windows.

    Walks the ~70 windows rather than the ~465,000 rows, touching only the rows
    belonging to each window's user. Overlapping windows union, and a window
    naming a user who is absent from the table is skipped here -- the report
    calls that out separately, because it means an insider produced no rows at
    all and that is a data problem, not a labelling one.
    """
    inside = np.zeros(len(daily), dtype=bool)
    if daily.empty:
        return inside

    rows_by_user = daily.groupby(USER_COLUMN).indices
    days = daily[DAY_COLUMN].to_numpy()

    for window in windows.itertuples(index=False):
        positions = rows_by_user.get(getattr(window, USER_COLUMN))
        if positions is None:
            continue
        window_days = days[positions]
        in_window = (
            window_days >= np.datetime64(getattr(window, WINDOW_START))
        ) & (window_days <= np.datetime64(getattr(window, WINDOW_END)))
        inside[positions[in_window]] = True

    return inside


def build_labelled_counts(
    daily: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    """Calendar-fill *daily*, then label it. The whole stage in one call."""
    return attach_labels(calendar_fill(daily), windows)


def _require_unique_keys(frame: pd.DataFrame, what: str) -> None:
    duplicates = int(frame.duplicated(list(KEY_COLUMNS)).sum())
    if duplicates:
        raise LabellingError(
            f"{what} contains {duplicates} duplicate (user, day) row(s); "
            f"Stage 1 must emit exactly one row per user-day"
        )


def verify(labelled: pd.DataFrame, before: pd.DataFrame) -> None:
    """Raise on anything that cannot be true of a correctly labelled table.

    These four are logical impossibilities rather than surprising measurements,
    so they stop the run instead of printing a warning. Everything that is
    merely *suspicious* -- an insider with no positive days, a positive rate
    outside the expected band -- is reported by :func:`format_report`, because a
    surprising number is something to look at, not necessarily something wrong.

    Raises:
        LabellingError: An invariant was violated.
    """
    if len(labelled) < len(before):
        raise LabellingError(
            f"the fill lost rows: {len(before):,} in, {len(labelled):,} out"
        )

    _require_unique_keys(labelled, "labelled counts")

    nan_columns = [c for c in labelled.columns if labelled[c].isna().any()]
    if nan_columns:
        raise LabellingError(f"NaNs present in column(s) {nan_columns}")

    orphans = labelled.loc[
        (labelled[LABEL_COLUMN] == 1) & (labelled[SCENARIO_COLUMN] == BENIGN_SCENARIO)
    ]
    if not orphans.empty:
        users = sorted(orphans[USER_COLUMN].unique())[:5]
        raise LabellingError(
            f"{len(orphans)} row(s) are label=1 but user_scenario=0, e.g. users "
            f"{users}. A positive day must belong to a known insider."
        )

    shrunk = _users_that_lost_rows(labelled, before)
    if shrunk:
        raise LabellingError(f"user(s) {shrunk[:5]} have fewer rows after the fill")


def _users_that_lost_rows(after: pd.DataFrame, before: pd.DataFrame) -> list[str]:
    counts_before = before.groupby(USER_COLUMN).size()
    counts_after = after.groupby(USER_COLUMN).size().reindex(counts_before.index).fillna(0)
    return sorted(counts_before.index[counts_after < counts_before])


@dataclass(frozen=True, slots=True)
class LabelStats:
    """The numbers the report prints and the checks are computed from."""

    rows_before: int
    rows_after: int
    users: int
    positives: int
    insiders_flagged: int
    insiders_expected: int
    insiders_without_positives: tuple[str, ...]

    @property
    def negatives(self) -> int:
        return self.rows_after - self.positives

    @property
    def positive_rate(self) -> float:
        return self.positives / self.rows_after if self.rows_after else 0.0

    @property
    def pos_weight(self) -> float:
        """``n_neg / n_pos`` -- reported for orientation only.

        Stage 5 must recompute this from its own training split. Freezing the
        whole-table value here would leak the test set's class balance into the
        loss function.
        """
        return self.negatives / self.positives if self.positives else float("nan")


def compute_stats(
    labelled: pd.DataFrame, before: pd.DataFrame, windows: pd.DataFrame
) -> LabelStats:
    """Collect everything the report and the checks need in one pass."""
    positives_by_user = labelled.groupby(USER_COLUMN)[LABEL_COLUMN].sum()
    named = set(windows[USER_COLUMN]) & set(labelled[USER_COLUMN])
    silent = sorted(u for u in named if positives_by_user.get(u, 0) == 0)

    return LabelStats(
        rows_before=len(before),
        rows_after=len(labelled),
        users=labelled[USER_COLUMN].nunique(),
        positives=int(labelled[LABEL_COLUMN].sum()),
        insiders_flagged=int(
            (labelled.groupby(USER_COLUMN)[SCENARIO_COLUMN].first() != BENIGN_SCENARIO).sum()
        ),
        insiders_expected=len(named),
        insiders_without_positives=tuple(silent),
    )


def format_report(
    labelled: pd.DataFrame,
    before: pd.DataFrame,
    windows: pd.DataFrame,
    out: Path | None = None,
) -> str:
    """Render the stage's shape, class balance, per-scenario split and checks."""
    stats = compute_stats(labelled, before, windows)
    lines = [
        _RULE,
        "InnerSight Stage 2 - labelled daily counts",
        _RULE,
        f"Rows before fill   : {stats.rows_before:,}",
        f"Rows after fill    : {stats.rows_after:,} "
        f"(+{stats.rows_after - stats.rows_before:,} zero-filled)",
        f"Distinct users     : {stats.users:,}",
    ]
    if not labelled.empty:
        lines.append(
            f"Date range         : {labelled[DAY_COLUMN].min():%Y-%m-%d} -> "
            f"{labelled[DAY_COLUMN].max():%Y-%m-%d}"
        )
    lines += [
        f"Positive rows      : {stats.positives:,} "
        f"({stats.positive_rate:.3%} of all rows)",
        f"pos_weight (neg/pos): {stats.pos_weight:,.1f}   "
        f"[orientation only - Stage 5 recomputes it on the train split]",
        "",
        "Per scenario:",
        *_scenario_lines(labelled, windows),
        "",
        "Sanity checks:",
        *_check_lines(labelled, stats),
        _RULE,
    ]
    if out is not None:
        lines.append(f"Written to: {out}")
    return "\n".join(lines)


def _scenario_lines(labelled: pd.DataFrame, windows: pd.DataFrame) -> list[str]:
    """One row per scenario: insiders, positive days, and days per insider.

    Worth reading rather than skipping. On both releases scenario 3 has 20
    positive days in total across the whole dataset -- ten insiders, two days
    each -- so once Stage 4 splits by user the test set may hold four of them,
    which is the explanation for scenario 3's Stage 7 numbers, stated before
    they are measured rather than after.

    The scenario that dominates is not the same one in both releases, which is
    why this block is printed rather than remembered: on r4.2 scenario 2 is
    ~89% of all positive days, so any overall metric is very nearly a
    scenario-2 metric; on r5.2 scenario 4 takes ~56% and scenario 2 ~40%.
    """
    header = (
        f"  {'scenario':<10}{'insiders':>10}{'positive days':>16}"
        f"{'mean days/insider':>20}"
    )
    lines = [header]
    positives = labelled.loc[labelled[LABEL_COLUMN] == 1]

    for scenario in SCENARIOS:
        users = set(windows.loc[windows["scenario"] == scenario, USER_COLUMN])
        present = users & set(labelled[USER_COLUMN])
        days = int(positives[USER_COLUMN].isin(present).sum())
        mean = days / len(present) if present else 0.0
        lines.append(
            f"  {scenario:<10}{len(present):>10}{days:>16,}{mean:>20.1f}"
        )
    return lines


def _check_lines(labelled: pd.DataFrame, stats: LabelStats) -> list[str]:
    """One line per check, each marked ok or CHECK. Nothing here raises."""
    scenario_users = labelled.groupby(USER_COLUMN)[SCENARIO_COLUMN].nunique()
    inconsistent = int((scenario_users > 1).sum())
    nan_columns = [c for c in labelled.columns if labelled[c].isna().any()]
    silent = stats.insiders_without_positives

    return [
        _check(
            f"{stats.insiders_expected} users flagged as insiders",
            stats.insiders_flagged == stats.insiders_expected,
            f"{stats.insiders_flagged} flagged",
        ),
        _check(
            "every insider has >= 1 positive row",
            not silent,
            "none missing" if not silent else f"no positives for {list(silent[:5])}",
        ),
        _check(
            "no label=1 row belongs to a benign user",
            not (
                (labelled[LABEL_COLUMN] == 1)
                & (labelled[SCENARIO_COLUMN] == BENIGN_SCENARIO)
            ).any(),
            "none",
        ),
        _check(
            "user_scenario constant per user",
            inconsistent == 0,
            f"{inconsistent} user(s) vary",
        ),
        _check("no NaNs anywhere", not nan_columns, nan_columns or "none"),
        _check(
            "no duplicate (user, day)",
            not labelled.duplicated(list(KEY_COLUMNS)).any(),
            f"{int(labelled.duplicated(list(KEY_COLUMNS)).sum())} duplicates",
        ),
        _check(
            "row count did not decrease",
            stats.rows_after >= stats.rows_before,
            f"{stats.rows_before:,} -> {stats.rows_after:,}",
        ),
        _check(
            "labels are 0/1 only",
            set(pd.unique(labelled[LABEL_COLUMN])) <= {0, 1},
            sorted(pd.unique(labelled[LABEL_COLUMN]).tolist()),
        ),
    ]


def _check(label: str, passed: bool, detail: object) -> str:
    return f"  {'ok   ' if passed else 'CHECK'} {label:<40} {detail}"


def format_user_report(
    labelled: pd.DataFrame,
    user: str,
    windows: pd.DataFrame | None = None,
    rows: int = 30,
) -> str:
    """Print one user's most notable days with their labels, in date order.

    The point is to make the label BOUNDARY visible. Showing the first N days is
    useless -- an attack is rarely in January -- so days are ranked by the
    features that carry signal and then displayed chronologically, which puts
    the days on either side of the window edge next to each other where a wrong
    date comparison is obvious.
    """
    preview_features = (
        "after_hours_logons", "weekend_logons", "usb_connect_count",
        "after_hours_usb", "file_count", "email_to_external_count",
        "job_search_visits", "cloud_upload_visits",
    )
    signal_features = (
        "usb_connect_count", "after_hours_usb", "after_hours_logons",
        "weekend_logons", "file_count", "job_search_visits",
        "cloud_upload_visits",
    )

    subset = labelled.loc[labelled[USER_COLUMN] == user]
    if subset.empty:
        sample = ", ".join(sorted(labelled[USER_COLUMN].unique())[:5])
        return f"No rows for user {user!r}. Users look like: {sample}"

    signal = subset[list(signal_features)].sum(axis=1)
    ranked = signal.add(signal.max() + 1 if len(signal) else 1).where(
        subset[LABEL_COLUMN] == 1, signal
    )
    notable = subset.loc[ranked.sort_values(ascending=False).index[:rows]]
    shown = notable.sort_values(DAY_COLUMN).copy()
    shown["in_window"] = np.where(shown[LABEL_COLUMN] == 1, "<<<", "")
    shown[DAY_COLUMN] = shown[DAY_COLUMN].dt.strftime("%Y-%m-%d")

    scenario = int(subset[SCENARIO_COLUMN].iloc[0])
    positives = int(subset[LABEL_COLUMN].sum())
    heading = (
        f"{user} - scenario {scenario}, {len(subset):,} days, "
        f"{positives} labelled 1, showing the {len(shown)} most notable"
    )
    body = shown[[DAY_COLUMN, *preview_features, LABEL_COLUMN, "in_window"]]

    lines = [_RULE, heading]
    if windows is not None:
        lines += _window_lines(windows, user)
    lines += [_RULE, body.to_string(index=False), _RULE]
    return "\n".join(lines)


def _window_lines(windows: pd.DataFrame, user: str) -> list[str]:
    mine = windows.loc[windows[USER_COLUMN] == user]
    return [
        f"  window: {row[WINDOW_START]:%Y-%m-%d} -> {row[WINDOW_END]:%Y-%m-%d} "
        f"(inclusive, {(row[WINDOW_END] - row[WINDOW_START]).days + 1} days)"
        for _, row in mine.iterrows()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Label the daily count table, write it, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample", action="store_true",
        help="Read the 30-user sample table and write a sample-named output.",
    )
    parser.add_argument(
        "--counts", type=Path, default=None,
        help="Daily counts parquet to read. Defaults to the full or sample path.",
    )
    parser.add_argument(
        "--answers", type=Path, default=None,
        help="insiders.csv to read. Defaults to the resolved answers directory.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Parquet to write. Defaults to the full or sample labelled path.",
    )
    parser.add_argument(
        "--user", default=None,
        help="After labelling, print this user's days and window.",
    )
    args = parser.parse_args(argv)

    counts_path = args.counts or (
        DAILY_COUNTS_SAMPLE_PATH if args.sample else DAILY_COUNTS_PATH
    )
    out_path = args.out or (
        LABELLED_COUNTS_SAMPLE_PATH if args.sample else LABELLED_COUNTS_PATH
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not counts_path.is_file():
        logger.error(
            "daily counts not found: %s. Run `python -m extraction.extract%s` first.",
            counts_path, "" if args.sample else " --full",
        )
        return 1

    logger.info("reading %s", counts_path)
    daily = pd.read_parquet(counts_path)
    windows = load_insiders(args.answers or insiders_csv_path())

    labelled = build_labelled_counts(daily, windows)
    verify(labelled, daily)

    ensure_directories()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(out_path, index=False)
    logger.info("wrote %d rows to %s", len(labelled), out_path)

    print(format_report(labelled, daily, windows, out=out_path))
    if args.user:
        print()
        print(format_user_report(labelled, args.user, windows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
