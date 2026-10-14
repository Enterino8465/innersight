"""Build the r5.2 sample release, because CERT does not ship one.

    python -m tools.build_r52_sample --dry-run    # choose the users, write nothing
    python -m tools.build_r52_sample              # build it

Without a sample, every test of r5.2 code costs a 39.7 GB run: http.csv is
30.1 GB, email.csv 9.4 GB, file.csv 408 MB, logon.csv 124 MB, device.csv 76 MB
(measured 2026-08-21). This produces the same shape of thing that already sits
next to it as ``r4.2_sample/`` -- five logs cut down to a chosen set of users,
with everything else copied through untouched.

**The sample's numbers are meaningless as results, and every stage says so when
run with ``--sample``.** Its only job is to prove the code runs end to end.


Why these users
---------------

**Twelve insiders, three from each of the four scenarios.** All four is
non-negotiable: scenario 4 is what r5.2 adds and what brief R3 exists to
handle, so a sample without it cannot smoke-test the thing being built. Three
per scenario rather than one, because one insider whose attack window happens
to sit at the very start of the calendar would leave a scenario untestable by
Stage 3, which needs ``MIN_HISTORY_DAYS`` of prior days before it trusts a
z-score.

Within a scenario the three are the **earliest, median and latest** attack
window by start time. That spreads them across the ~17-month calendar instead
of clustering them, and it is a rule rather than a preference -- rerun this
script and you get the same twelve people.

**Eighteen benign users, present in both the first and the last LDAP month.**
An employee who joined late or left early has a truncated history, and a
truncated history is exactly what Stage 3's per-user baseline handles worst;
the sample should not smoke-test the pipeline against its own hardest edge
case by accident. Of those present throughout, the eighteen are taken at an
even stride through the user ids sorted alphabetically, which is deterministic
and spreads them across the organisation rather than taking eighteen people
from one team.

Thirty users, twelve of them insiders. ``r4.2_sample/`` is thirty users, nine
of them insiders (3 / 2 / 4 across its three scenarios) -- so this is the same
size at a slightly higher insider rate, which costs nothing because the rate is
not a result either.


Why it streams rather than parses
---------------------------------

Measured on the shipped r5.2 files on 2026-08-21, over 300 MB prefixes of
``email.csv`` and ``http.csv``, 100 MB of ``logon.csv``, 60 MB of ``device.csv``
and the whole of ``file.csv``:

* every CSV record occupies exactly one physical line -- no field contains an
  embedded newline, so line count and ``csv.reader`` row count agree exactly;
* ``line.split(",", 3)[2]`` is the ``user`` field in every one of the five
  logs, with zero disagreements against ``csv.reader``. The first three columns
  are ``id,date,user`` and none of the three can contain a comma.

So the filter is a text scan: read a line, look at one field, keep or drop.
Running ``csv.reader`` over 39.7 GB would be several times slower for no gain.
The assumption is not taken on trust -- :func:`verify_output` re-reads every
file this script writes with ``csv.reader`` and fails if the row count, the
column count or the user set disagrees with what the text scan believed.


What is copied through untouched
--------------------------------

``answers/insiders.csv``, ``LDAP/`` and ``psychometric.csv``, matching
``r4.2_sample/`` exactly -- inspected on 2026-08-21, and none of the three is
filtered there either. insiders.csv in particular holds all ten CERT releases
and is filtered by ``RELEASE_ID`` at read time, so cutting it down here would
only create a second, subtly different copy of the ground truth.

``decoy_file.csv`` is NOT copied. It has no ``user`` column (its columns are
``decoy_filename,pc``), nothing in the pipeline reads it, and ``r4.2_sample/``
has no equivalent.

After a successful run this prints the five row counts as a paste-ready snippet
for ``config._LOG_SPECS_R52``, whose ``expected_sample_rows`` are ``None``
until then. They are a checksum, and a checksum that was guessed is worse than
no checksum at all.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from config import (
    ANSWERS_DIR,
    ANSWERS_DIR_MIRROR,
    INSIDERS_COLUMNS,
    INSIDERS_DATE_FORMAT,
    INSIDERS_FILENAME,
    PROFILES,
    RELEASE_R52,
    LogSpec,
)

logger = logging.getLogger("build_r52_sample")

PROFILE: Final = PROFILES[RELEASE_R52]

COPIED_FILES: Final[tuple[str, ...]] = ("psychometric.csv",)
COPIED_DIRS: Final[tuple[str, ...]] = ("LDAP",)
SAMPLE_USERS_FILENAME: Final[str] = "sample_users.txt"
LDAP_DIRNAME: Final[str] = "LDAP"
LDAP_USER_COLUMN: Final[str] = "user_id"
USER_FIELD_INDEX: Final[int] = 2
PROGRESS_EVERY_BYTES: Final[int] = 2 * 1024**3

DEFAULT_INSIDERS_PER_SCENARIO: Final[int] = 3
DEFAULT_BENIGN_USERS: Final[int] = 18


class SampleBuildError(RuntimeError):
    """The sample cannot be built from what is on disk."""


@dataclass(frozen=True, slots=True)
class Insider:
    """One row of ``insiders.csv``, already filtered to r5.2."""

    user: str
    scenario: int
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class Selection:
    """Who ends up in the sample, and enough context to explain why."""

    insiders: tuple[Insider, ...]
    benign: tuple[str, ...]

    @property
    def users(self) -> tuple[str, ...]:
        """Insiders first, then benign -- the order written to the users file."""
        return tuple(i.user for i in self.insiders) + self.benign

    @property
    def user_set(self) -> frozenset[str]:
        return frozenset(self.users)


def read_r52_insiders(answers_dir: Path) -> tuple[Insider, ...]:
    """Every r5.2 row of insiders.csv, parsed.

    The ``dataset`` column is compared as TEXT. Read as a float it would make
    "5.2" and "5.20" the same release and "4.2" indistinguishable from "4.20",
    which is the kind of thing that works right up until it does not.
    """
    path = answers_dir / INSIDERS_FILENAME
    if not path.is_file():
        raise SampleBuildError(f"no {INSIDERS_FILENAME} at {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in INSIDERS_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SampleBuildError(
                f"{path} is missing columns {missing}; expected "
                f"{list(INSIDERS_COLUMNS)} and found {reader.fieldnames}"
            )
        rows = [row for row in reader if row["dataset"] == PROFILE.release_id]

    insiders = tuple(
        Insider(
            user=row["user"],
            scenario=int(row["scenario"]),
            start=datetime.strptime(row["start"], INSIDERS_DATE_FORMAT),
            end=datetime.strptime(row["end"], INSIDERS_DATE_FORMAT),
        )
        for row in rows
    )

    if len(insiders) != PROFILE.expected_insider_count:
        logger.warning(
            "%s holds %d r%s rows; config.EXPECTED_INSIDER_COUNT says %d. "
            "Not fatal here, but one of the two is out of date.",
            path, len(insiders), PROFILE.release_id,
            PROFILE.expected_insider_count,
        )
    return insiders


def pick_insiders(
    insiders: Iterable[Insider], per_scenario: int
) -> tuple[Insider, ...]:
    """Earliest, median and latest attack window from each scenario.

    Deterministic by construction: sorted by ``(start, user)``, then indexed at
    evenly spaced positions. Rerunning this picks the same people.
    """
    chosen: list[Insider] = []
    for scenario in PROFILE.scenarios:
        pool = sorted(
            (i for i in insiders if i.scenario == scenario),
            key=lambda i: (i.start, i.user),
        )
        if not pool:
            raise SampleBuildError(
                f"no r{PROFILE.release_id} insiders for scenario {scenario}; "
                f"a sample without every scenario cannot smoke-test the "
                f"labelling that brief R3 adds"
            )
        take = min(per_scenario, len(pool))
        if take < per_scenario:
            logger.warning(
                "scenario %d has only %d insiders; taking all of them "
                "instead of %d", scenario, len(pool), per_scenario,
            )
        if take == 1:
            positions = [0]
        else:
            positions = [
                round(k * (len(pool) - 1) / (take - 1)) for k in range(take)
            ]
        chosen += [pool[p] for p in dict.fromkeys(positions)]
    return tuple(chosen)


def read_ldap_users(directory: Path, filename: str) -> tuple[str, ...]:
    path = directory / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if LDAP_USER_COLUMN not in (reader.fieldnames or []):
            raise SampleBuildError(
                f"{path} has no {LDAP_USER_COLUMN!r} column; found "
                f"{reader.fieldnames}"
            )
        return tuple(row[LDAP_USER_COLUMN] for row in reader)


def pick_benign(
    source: Path, insiders: Iterable[Insider], count: int
) -> tuple[str, ...]:
    """Users present in both the first and last LDAP month, at an even stride.

    Present-throughout matters: an employee who joined late or left early has a
    truncated history, and Stage 3's per-user baseline is at its least
    trustworthy on exactly those people. The sample should not put the pipeline
    on its own hardest edge case by accident.
    """
    ldap = source / LDAP_DIRNAME
    months = sorted(p.name for p in ldap.glob("*.csv"))
    if len(months) < 2:
        raise SampleBuildError(
            f"expected at least two monthly files in {ldap}, found {months}"
        )

    first = set(read_ldap_users(ldap, months[0]))
    last = set(read_ldap_users(ldap, months[-1]))
    everyone = set(first) | set(last)
    logger.info(
        "LDAP %s .. %s: %d employees in the first month, %d in the last, "
        "%d present in both",
        months[0], months[-1], len(first), len(last), len(first & last),
    )
    if len(everyone) != PROFILE.expected_employee_count:
        logger.warning(
            "LDAP names %d distinct employees; config."
            "EXPECTED_EMPLOYEE_COUNT says %d",
            len(everyone), PROFILE.expected_employee_count,
        )

    insider_ids = {i.user for i in insiders}
    pool = sorted((first & last) - insider_ids)
    if len(pool) < count:
        raise SampleBuildError(
            f"only {len(pool)} benign users are present throughout; "
            f"{count} were asked for"
        )
    stride = len(pool) / count
    return tuple(pool[int(k * stride)] for k in range(count))


def choose(
    source: Path,
    answers_dir: Path,
    *,
    per_scenario: int = DEFAULT_INSIDERS_PER_SCENARIO,
    benign: int = DEFAULT_BENIGN_USERS,
) -> Selection:
    all_insiders = read_r52_insiders(answers_dir)
    chosen = pick_insiders(all_insiders, per_scenario)
    return Selection(
        insiders=chosen,
        benign=pick_benign(source, all_insiders, benign),
    )


def describe(selection: Selection) -> str:
    rate = len(selection.insiders) / len(selection.users)
    lines = [
        f"{len(selection.users)} users: {len(selection.insiders)} insiders, "
        f"{len(selection.benign)} benign ({rate:.0%} insider rate)",
        "",
        "insiders (earliest / median / latest attack window per scenario):",
    ]
    for scenario in PROFILE.scenarios:
        for insider in selection.insiders:
            if insider.scenario == scenario:
                lines.append(
                    f"  scenario {insider.scenario}  {insider.user}  "
                    f"{insider.start:%Y-%m-%d} .. {insider.end:%Y-%m-%d}  "
                    f"({(insider.end - insider.start).days + 1} days)"
                )
    lines += ["", "benign (present in the first and last LDAP month):"]
    for start in range(0, len(selection.benign), 6):
        lines.append("  " + "  ".join(selection.benign[start : start + 6]))
    return "\n".join(lines)


def check_header(spec: LogSpec, header: str, path: Path) -> None:
    """Refuse to filter a file whose columns are not the ones config expects.

    The header is authoritative -- r5.2's own readme.txt describes file.csv
    with r4.2's columns and is simply wrong -- so a mismatch here means either
    the wrong folder or a spec that has drifted, and both are worth stopping
    for.
    """
    found = tuple(header.rstrip("\r\n").split(","))
    if found != spec.columns:
        raise SampleBuildError(
            f"{path} does not have the columns config expects for r"
            f"{PROFILE.release_id}.\n"
            f"  expected: {','.join(spec.columns)}\n"
            f"  found:    {','.join(found)}"
        )
    if spec.columns[USER_FIELD_INDEX] != "user":
        raise SampleBuildError(
            f"{spec.filename}: this script finds the user by splitting on the "
            f"first {USER_FIELD_INDEX + 1} commas, which assumes column "
            f"{USER_FIELD_INDEX} is 'user'. It is "
            f"{spec.columns[USER_FIELD_INDEX]!r}."
        )


def filter_log(spec: LogSpec, source: Path, destination: Path,
               users: frozenset[str]) -> int:
    """Copy the header plus every line belonging to ``users``. Returns rows kept.

    A text scan, not a CSV parse -- see this module's docstring for the
    measurement that makes that safe, and :func:`verify_output` for the check
    that it stayed safe.
    """
    in_path = source / spec.filename
    out_path = destination / spec.filename
    if not in_path.is_file():
        raise SampleBuildError(f"no {spec.filename} at {in_path}")

    total_bytes = in_path.stat().st_size
    logger.info(
        "%s: %.1f GB -> %s", spec.filename, total_bytes / 1024**3, out_path
    )

    kept = 0
    seen = 0
    read_bytes = 0
    next_report = PROGRESS_EVERY_BYTES
    with in_path.open(encoding="utf-8", newline="") as reader, \
            out_path.open("w", encoding="utf-8", newline="") as writer:
        header = reader.readline()
        check_header(spec, header, in_path)
        writer.write(header)
        for line in reader:
            seen += 1
            read_bytes += len(line)
            parts = line.split(",", USER_FIELD_INDEX + 1)
            if len(parts) <= USER_FIELD_INDEX:
                raise SampleBuildError(
                    f"{in_path} line {seen + 1} has fewer than "
                    f"{USER_FIELD_INDEX + 1} comma-separated fields, so it is "
                    f"not one record per line as measured. Refusing to guess: "
                    f"{line[:120]!r}"
                )
            if parts[USER_FIELD_INDEX] in users:
                writer.write(line)
                kept += 1
            if read_bytes >= next_report:
                logger.info(
                    "  %s: %.0f%% (%s rows read, %s kept)",
                    spec.filename, 100 * read_bytes / max(total_bytes, 1),
                    f"{seen:,}", f"{kept:,}",
                )
                next_report += PROGRESS_EVERY_BYTES

    logger.info("  %s: %s of %s rows kept", spec.filename, f"{kept:,}", f"{seen:,}")
    return kept


def verify_output(spec: LogSpec, destination: Path,
                  users: frozenset[str], expected_rows: int) -> None:
    """Re-read what was written, with a real CSV parser this time.

    The filter believed one record per physical line and the user at a fixed
    field index. This is where that belief is checked rather than asserted:
    if any kept record had held an embedded newline, the parsed row count would
    not match the line count and the column count would be wrong.
    """
    path = destination / spec.filename
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if tuple(rows[0]) != spec.columns:
        raise SampleBuildError(f"{path}: header changed during the copy")
    body = rows[1:]
    if len(body) != expected_rows:
        raise SampleBuildError(
            f"{path}: wrote {expected_rows} rows but the CSV parser sees "
            f"{len(body)}. A field must contain an embedded newline, which "
            f"the measured assumption says it cannot -- do not ship this "
            f"sample until that is understood."
        )
    width = len(spec.columns)
    bad_width = [n for n, row in enumerate(body, 2) if len(row) != width]
    if bad_width:
        raise SampleBuildError(
            f"{path}: {len(bad_width)} rows do not have {width} fields, "
            f"first at line {bad_width[0]}"
        )
    strangers = {row[USER_FIELD_INDEX] for row in body} - users
    if strangers:
        raise SampleBuildError(
            f"{path}: kept rows for users not in the sample: "
            f"{sorted(strangers)[:5]}"
        )
    logger.info("  %s verified: %s rows, all users in the sample",
                spec.filename, f"{len(body):,}")


def copy_through(source: Path, destination: Path, answers_dir: Path) -> None:
    """Everything that is carried across unfiltered, matching r4.2_sample/."""
    answers_out = destination / "answers"
    answers_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(answers_dir / INSIDERS_FILENAME, answers_out / INSIDERS_FILENAME)
    logger.info("copied %s (all ten releases, filtered by RELEASE_ID at read "
                "time)", INSIDERS_FILENAME)

    for name in COPIED_DIRS:
        shutil.copytree(source / name, destination / name, dirs_exist_ok=True)
        logger.info("copied %s/", name)

    for name in COPIED_FILES:
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)
            logger.info("copied %s", name)
        else:
            logger.warning("%s is not in %s; skipping", name, source)


def write_users_file(destination: Path, selection: Selection) -> None:
    path = destination / SAMPLE_USERS_FILENAME
    path.write_text("\n".join(selection.users) + "\n", encoding="utf-8")
    logger.info("wrote %s (%d users, insiders first)",
                path, len(selection.users))


def build(source: Path, answers_dir: Path, destination: Path,
          selection: Selection) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    copy_through(source, destination, answers_dir)
    write_users_file(destination, selection)

    users = selection.user_set
    counts: dict[str, int] = {}
    for spec in PROFILE.log_specs:
        counts[spec.name] = filter_log(spec, source, destination, users)
    for spec in PROFILE.log_specs:
        verify_output(spec, destination, users, counts[spec.name])
    return counts


def config_snippet(counts: dict[str, int]) -> str:
    """The paste-ready `expected_sample_rows` values for config._LOG_SPECS_R52."""
    lines = [
        "Paste these into config._LOG_SPECS_R52 -- they are the checksum that",
        "tells verify_setup.py the sample was rebuilt:",
        "",
    ]
    lines += [
        f"    {name:<7}  expected_sample_rows={count:_},"
        for name, count in counts.items()
    ]
    return "\n".join(lines)


def default_source() -> Path | None:
    for candidate in PROFILE.full_candidates:
        if candidate.is_dir():
            return candidate
    return None


def default_answers() -> Path | None:
    for candidate in (ANSWERS_DIR, ANSWERS_DIR_MIRROR):
        if candidate.is_dir():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="The full r5.2 release. Defaults to the first of "
             + ", ".join(str(p) for p in PROFILE.full_candidates) + " that exists.",
    )
    parser.add_argument(
        "--answers", type=Path, default=None,
        help="The answers directory holding insiders.csv.",
    )
    parser.add_argument(
        "--out", type=Path, default=PROFILE.sample_dir,
        help=f"Where to write the sample (default: {PROFILE.sample_dir}).",
    )
    parser.add_argument(
        "--insiders-per-scenario", type=int,
        default=DEFAULT_INSIDERS_PER_SCENARIO,
        help=f"Default {DEFAULT_INSIDERS_PER_SCENARIO}, one from each end of "
             f"the calendar and one from the middle.",
    )
    parser.add_argument(
        "--benign", type=int, default=DEFAULT_BENIGN_USERS,
        help=f"How many non-insiders to include (default "
             f"{DEFAULT_BENIGN_USERS}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Choose the users, print them, and write nothing. Seconds, not "
             "minutes -- it never opens a log file.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing sample directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    source = args.source or default_source()
    if source is None or not source.is_dir():
        logger.error(
            "no r%s release found. Tried: %s. Plug the drive in, or pass "
            "--source.", PROFILE.release_id,
            ", ".join(str(p) for p in PROFILE.full_candidates),
        )
        return 2

    answers_dir = args.answers or default_answers()
    if answers_dir is None or not answers_dir.is_dir():
        logger.error(
            "no answers directory found. Tried: %s, %s. Pass --answers.",
            ANSWERS_DIR, ANSWERS_DIR_MIRROR,
        )
        return 2

    logger.info("source  %s", source)
    logger.info("answers %s", answers_dir)
    logger.info("out     %s", args.out)

    try:
        selection = choose(
            source, answers_dir,
            per_scenario=args.insiders_per_scenario,
            benign=args.benign,
        )
    except SampleBuildError as error:
        logger.error("%s", error)
        return 1

    print()
    print(describe(selection))
    print()

    if args.dry_run:
        logger.info("--dry-run: nothing written")
        return 0

    if args.out.exists() and not args.force:
        logger.error(
            "%s already exists. Re-run with --force to overwrite it, after "
            "checking that nothing depends on the copy that is there.",
            args.out,
        )
        return 2

    try:
        counts = build(source, answers_dir, args.out, selection)
    except SampleBuildError as error:
        logger.error("%s", error)
        return 1

    print()
    print(config_snippet(counts))
    print()
    logger.info("sample built at %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
