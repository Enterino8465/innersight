"""Which CERT release a folder holds, decided by five header lines.

The whole of this module is one idea: **the header is authoritative**. Not the
folder's name, not the release's own ``readme.txt`` — which describes r5.2's
``file.csv`` with r4.2's column list and is simply wrong — and not what the
person clicking the button believes they pointed at.

Only the first line of each CSV is read. That is not an optimisation for its
own sake: r5.2's ``http.csv`` is 30 GB, and a detector that has to read a body
is a detector nobody will wait for.

**Why refusing a release matters more than crashing on it.** r5.2's
``email.csv`` carries an ``activity`` column with values ``Send`` and
``Receive`` that r4.2 has no equivalent of. Run *unaligned* r4.2 extraction
over it and ``email_sent_count`` silently becomes the count of *all* mail —
measured over all 17,361,575 rows, 68.00% Receive against 32.00% Send, so
3.1253x too large and counting inboxes as outboxes. Nothing raises. The
z-scores still look reasonable, the model still trains, and the number at the
end is confident and wrong. A crash would be the kind outcome; this module is
what produces one instead.

That particular hazard is gone: ``extraction/extract.py``'s ``_is_sent()``
filters the column, and **r5.2 became SUPPORTED on 2026-08-22** under brief R8.
The paragraph stays because the shape of the failure is what this module
exists for, and the next release will arrive with its own version of it.

**This is a gate, not a wall.** ``config.RELEASE_CAPABILITY`` is a table with
one row per release, read at call time, and it is the only place in
``pipeline/`` that decides what a release string means. A release is enabled by
changing one entry from ``NOT_YET_ALIGNED`` to ``SUPPORTED``;
``ui/test_ui.py::test_the_gate_opens`` — which monkeypatches exactly that entry
— is the standing proof, and it passed unedited on both sides of r5.2's flip.
Two tests beside it prove the other direction, so the mechanism does not depend
on any release being refused.

There is deliberately no ``--force-release``. An override's only function would
be to produce silently wrong numbers, and the honest way to enable a release is
to do the alignment work.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from config import (
    LOG_NAMES,
    LOG_SPECS,
    RELEASE_CAPABILITY,
    RELEASE_LOG_COLUMNS,
    RELEASE_NOT_YET_ALIGNED,
    RELEASE_SUPPORTED,
    UI_DATASET_CHOICES,
    resolve_release_folder,
)
from pipeline.commands import Reporter

logger = logging.getLogger(__name__)


class DetectionError(RuntimeError):
    """A folder that cannot be identified, stated with the file that failed."""


_MAX_HEADER_BYTES: Final[int] = 8_192

_NOT_YET_ALIGNED_REASON: Final[dict[str, str]] = {
    "5.2": (
        "Recognised as CERT r5.2 — 2,000 employees, four attack scenarios. "
        "Not yet supported. The extraction stage is aligned to r4.2's "
        "columns. On r5.2, email.csv carries a Send/Receive activity column "
        "that r4.2 does not have, so email_sent_count would count received "
        "mail as sent — about 3.2x too large — and the run would finish "
        "without any error at all. Running it anyway would produce a "
        "confident wrong number rather than a crash. Feature alignment for "
        "r5.2 is specified and queued; until it lands this button stops here."
    ),
}


@dataclass(frozen=True, slots=True)
class LogHeader:
    """One raw CSV's first line, or the reason there isn't one."""

    log: str
    path: Path
    columns: tuple[str, ...] | None
    problem: str | None

    @property
    def readable(self) -> bool:
        return self.columns is not None

    def describe(self) -> str:
        if self.columns is None:
            return f"{self.path.name}: {self.problem}"
        return f"{self.path.name}: {','.join(self.columns)}"


@dataclass(frozen=True, slots=True)
class Detection:
    """What a folder is, and whether the pipeline may be pointed at it."""

    folder: Path
    headers: tuple[LogHeader, ...]
    release: str | None
    capability: str | None
    refusal: str | None

    @property
    def allowed(self) -> bool:
        """True only when a release was identified *and* the table permits it."""
        return self.refusal is None

    def evidence(self) -> tuple[str, ...]:
        """The five header lines, which are the entire basis for the verdict."""
        return tuple(header.describe() for header in self.headers)

    def summary(self) -> str:
        if self.release is None:
            return f"unrecognised: {self.refusal}"
        if self.allowed:
            return f"CERT r{self.release} — supported"
        return f"CERT r{self.release} — {self.capability}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "folder": str(self.folder),
            "release": self.release,
            "capability": self.capability,
            "allowed": self.allowed,
            "refusal": self.refusal,
            "evidence": list(self.evidence()),
        }


def identify(folder: Path | str) -> Detection:
    """Identify ``folder``'s release and decide whether it may be used.

    Never raises for an unusable folder: a missing file, an unreadable header
    and an unrecognised release are all *answers*, carried back in
    :attr:`Detection.refusal` so the page can print them. The only exception
    this raises is for a broken capability table, which is a bug in this
    project rather than a fact about the folder.
    """
    folder = Path(folder).expanduser()
    verify_tables()

    if not folder.is_dir():
        return _refuse(folder, (), f"{folder} is not a directory that exists.")

    headers = tuple(_read_log_header(folder, name) for name in LOG_NAMES)

    unreadable = [header for header in headers if not header.readable]
    if unreadable:
        return _refuse(folder, headers, _describe_unreadable(folder, unreadable))

    release = _match_release(headers)
    if release is None:
        return _refuse(folder, headers, _describe_mismatch(headers))

    capability = RELEASE_CAPABILITY[release]
    if capability != RELEASE_SUPPORTED:
        reason = _NOT_YET_ALIGNED_REASON.get(
            release,
            f"Recognised as CERT r{release}, which this pipeline lists as "
            f"{capability}. This button stops here.",
        )
        logger.warning("refusing %s: r%s is %s", folder, release, capability)
        return Detection(folder, headers, release, capability, reason)

    logger.info("identified %s as CERT r%s", folder, release)
    return Detection(folder, headers, release, capability, None)


def _read_log_header(folder: Path, log: str) -> LogHeader:
    """The first line of one raw CSV, parsed as CSV, or why it could not be."""
    path = folder / LOG_SPECS[log].filename
    if not path.is_file():
        return LogHeader(log, path, None, "missing from this folder")
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            first = handle.readline(_MAX_HEADER_BYTES)
    except OSError as error:
        return LogHeader(log, path, None, f"could not be read ({error.strerror})")

    first = first.lstrip("﻿").rstrip("\r\n")
    if not first:
        return LogHeader(log, path, None, "is empty")
    columns = tuple(field.strip() for field in next(csv.reader([first])))
    return LogHeader(log, path, columns, None)


def _match_release(headers: tuple[LogHeader, ...]) -> str | None:
    """The release whose table every one of the five headers matches, if any."""
    found = {header.log: header.columns for header in headers}
    for release, expected in RELEASE_LOG_COLUMNS.items():
        if all(found[log] == expected[log] for log in LOG_NAMES):
            return release
    return None


def _refuse(
    folder: Path, headers: tuple[LogHeader, ...], reason: str
) -> Detection:
    logger.warning("refusing %s: %s", folder, reason)
    return Detection(folder, headers, None, None, reason)


def _describe_unreadable(folder: Path, unreadable: list[LogHeader]) -> str:
    """Name every file that failed, and what the folder should have held."""
    listing = "; ".join(
        f"{header.path.name} {header.problem}" for header in unreadable
    )
    wanted = ", ".join(LOG_SPECS[name].filename for name in LOG_NAMES)
    return (
        f"{folder} is not a CERT release folder: {listing}. A release folder "
        f"holds all five of {wanted} at its top level."
    )


def _describe_mismatch(headers: tuple[LogHeader, ...]) -> str:
    """Say, per release, which file first disagreed and what was expected.

    Naming the *first* mismatch per release rather than all of them keeps the
    message short while still being actionable: a folder that is neither
    release is usually one file away from being recognisable, and that file is
    the one worth printing.
    """
    found = {header.log: header.columns for header in headers}
    parts: list[str] = []
    for release, expected in RELEASE_LOG_COLUMNS.items():
        for log in LOG_NAMES:
            if found[log] != expected[log]:
                parts.append(
                    f"not r{release}: {LOG_SPECS[log].filename} has "
                    f"[{','.join(found[log] or ())}], r{release} has "
                    f"[{','.join(expected[log])}]"
                )
                break
    return (
        "The five headers match no release this project knows. "
        + " — ".join(parts)
        + ". The header is what identifies a release; a folder name or a "
        "readme is not evidence."
    )


def verify_tables() -> None:
    """Assert the two module-level tables are consistent with each other.

    Cheap, and it catches the one edit most likely to be made carelessly:
    adding a release to ``RELEASE_LOG_COLUMNS`` and forgetting to give it a row
    in ``RELEASE_CAPABILITY``, which would make :func:`identify` raise a
    ``KeyError`` on a folder rather than refuse it in words.
    """
    for release, table in RELEASE_LOG_COLUMNS.items():
        missing = set(LOG_NAMES) - set(table)
        if missing:
            raise DetectionError(
                f"config.RELEASE_LOG_COLUMNS[{release!r}] is missing header "
                f"rows for {', '.join(sorted(missing))}"
            )
        if release not in RELEASE_CAPABILITY:
            raise DetectionError(
                f"release {release!r} has a header table but no row in "
                f"config.RELEASE_CAPABILITY. Add one, "
                f"{RELEASE_SUPPORTED} or {RELEASE_NOT_YET_ALIGNED}."
            )
    unknown = set(RELEASE_CAPABILITY) - set(RELEASE_LOG_COLUMNS)
    if unknown:
        raise DetectionError(
            f"config.RELEASE_CAPABILITY names {', '.join(sorted(unknown))}, "
            f"which have no header table to be detected by"
        )


def format_report(detection: Detection, *, include_reason: bool = True) -> str:
    """The block the page and the command line both show.

    ``include_reason=False`` leaves the refusal paragraph off, for callers that
    are about to print it themselves — a running task marks its own refusal,
    and saying the same paragraph twice makes the page look like it stuttered.
    """
    lines = [f"Folder: {detection.folder}", f"Verdict: {detection.summary()}"]
    lines.append("Evidence — the header line of each file:")
    lines.extend(f"  {line}" for line in detection.evidence())
    if detection.refusal and include_reason:
        lines.append("")
        lines.append(detection.refusal)
    return "\n".join(lines)


def locate(release: str) -> tuple[Path | None, str]:
    """Turn a menu choice into a folder, or into a sentence about why not.

    The menu is the only way to name a folder, so an unknown release string
    means the page is out of step with the config rather than that the user
    typed something odd — say that, rather than raising.
    """
    if release not in UI_DATASET_CHOICES:
        return None, (
            f"{release!r} is not one of the releases this page offers "
            f"({', '.join(UI_DATASET_CHOICES)}). Reload the page."
        )
    folder = resolve_release_folder(release)
    if folder is None:
        looked = "\n".join(f"  - {p}" for p in UI_DATASET_CHOICES[release])
        return None, (
            f"CERT r{release} is not on this machine right now. Looked in:\n"
            f"{looked}\nPlug the drive in, or set INNERSIGHT_DATASET_FULL."
        )
    return folder, str(folder)


def identify_or_refuse(report: Reporter, release: str) -> Detection | None:
    """Detection, printed with its evidence. None means the run is over.

    The menu said which release this is meant to be; that claim is not used
    for anything except finding the folder. What happens next is a header
    read, and if the two disagree the header wins and the run stops.
    """
    folder, where = locate(release)
    if folder is None:
        report.refuse(where)
        return None
    report.say(f"menu chose CERT r{release}; looking at {folder}")
    report.say("the folder's own headers decide what it is — reading five lines")
    detection = identify(folder)
    for line in format_report(detection, include_reason=False).splitlines():
        report.say(line)
    if not detection.allowed:
        report.refuse(detection.refusal or "the folder was refused")
        return None
    return detection


def main(argv: list[str] | None = None) -> int:
    """Identify a folder from the command line. Exit 0 usable, 4 refused."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.detect",
        description="Identify which CERT release a folder holds.",
    )
    parser.add_argument("folder", type=Path, help="the folder to identify.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    detection = identify(args.folder)
    print(format_report(detection))
    return 0 if detection.allowed else 4


if __name__ == "__main__":
    sys.exit(main())
