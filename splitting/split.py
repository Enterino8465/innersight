"""Stage 4: assign users — never days — to train, validation and test.

This is the smallest stage in the pipeline and the one whose failure mode is
the most expensive, because it produces no error. Every one of a user's ~465
rows describes the same person, scored against the same personal baseline,
reflecting the same habits. Put that person's Monday in training and their
Tuesday in test and the model has already met them: it knows their baseline,
their shift pattern, their email volume. The test score then measures
memorisation and it looks *excellent*. So this module assigns **users**, and
every row belonging to a user follows them.

Three decisions carry the stage:

**Stratify by ``user_scenario``, not by insider/benign.** Every group the data
contains — benign, and one per attack scenario — is split 60/20/20 and
concatenated. There are four such groups on r4.2 and five on r5.2, and this
module never needs to be told which: it reads the scenario values out of the
table. On r4.2, scenario 3 has ten insiders in the entire release at two attack
days each, so a test set holding two of them holds roughly *four* positive rows
of that attack type. A plain insider/benign stratification could hand test zero
of them, at which point scenario-3 recall is undefined and a quarter of the
results table silently disappears. That number is not a problem to engineer
around; it is a fact about the release that belongs in the write-up before any
result is shown.

**Sort before shuffling.** The classic silent bug here is shuffling users in
whatever order they fell out of the parquet file or a Python ``set``. That
order is not a guarantee, so the same seed produces a different split on a
different machine or after an unrelated change upstream, and nothing warns
you — while Stages 5, 6 and 7 all read this file and quietly stop being
comparable. Every group is sorted lexicographically first and shuffled with an
explicitly seeded generator, never the global one. :func:`split_users` is a
pure function of (user set, scenario, seed).

**Three seeds, and all three are reported.** Attack windows differ enormously
in length — on r4.2 scenario 2 runs 45-59 days and scenario 3 exactly 2 — so
*which* insiders land in test moves the positive count substantially. One split
is one draw of that lottery. Generating several and keeping the tidiest is selection
on the test set; the spread across ``config.SEEDS`` is itself a result.

What this stage does **not** do: no z-scores, no windowing, no standardisation,
no models, and no copy of ``dataset.parquet`` — the split is an index into it.
``pos_weight`` is deliberately not computed here either; Stage 5 recomputes it
from its own training split, and freezing a whole-table value would leak the
test set's class balance into the loss function.

Run it from the project root::

    python -m splitting.split              # the 1,000-user table
    python -m splitting.split --sample     # the 30-user sample
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from config import (
    BENIGN_SCENARIO,
    DATASET_PATH,
    DATASET_SAMPLE_PATH,
    LABEL_COLUMN,
    SCENARIO_COLUMN,
    SCENARIOS,
    SEEDS,
    SPLIT,
    SPLITS_PATH,
    SPLITS_SAMPLE_PATH,
    USER_COLUMN,
    SplitFractions,
    ensure_directories,
)

logger = logging.getLogger(__name__)

SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")

REQUIRED_COLUMNS: Final[tuple[str, str, str]] = (
    USER_COLUMN, SCENARIO_COLUMN, LABEL_COLUMN,
)

ROWS_COLUMN: Final[str] = "rows"
POSITIVES_COLUMN: Final[str] = "positive_rows"

JSON_INDENT: Final[int] = 2

MIN_PLAUSIBLE_POSITIVE_RATE: Final[float] = 0.002
MAX_PLAUSIBLE_POSITIVE_RATE: Final[float] = 0.007

_RULE: Final[str] = "=" * 78


class SplitError(AssertionError):
    """A split violates something that must be true of any valid assignment."""


def read_user_table(path: Path) -> pd.DataFrame:
    """One row per user: their scenario, their row count, their positives.

    Reads three columns out of the parquet and collapses ~465,000 rows to
    1,000 immediately. Everything downstream in this module works on users.

    The frame is indexed by user and sorted lexicographically, which is where
    determinism starts: no ordering that came out of the file survives past
    this function.
    """
    frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    return user_table(frame)


def user_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a row-per-day table to a row-per-user table."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SplitError(
            f"dataset is missing required column(s) {missing}; "
            f"expected {list(REQUIRED_COLUMNS)}"
        )
    if not pd.api.types.is_string_dtype(frame[USER_COLUMN]):
        raise SplitError(
            f"{USER_COLUMN!r} must be a string column, got "
            f"{frame[USER_COLUMN].dtype!r}"
        )

    grouped = frame.groupby(USER_COLUMN, sort=True)
    scenarios = grouped[SCENARIO_COLUMN].nunique()
    if len(scenarios) and int(scenarios.max()) > 1:
        offenders = sorted(scenarios[scenarios > 1].index)[:5]
        raise SplitError(
            f"{SCENARIO_COLUMN!r} must be constant per user; it is not for "
            f"{len(scenarios[scenarios > 1])} user(s), e.g. {offenders}. "
            f"Stage 2 writes it per user -- something merged at the wrong grain."
        )

    table = grouped.agg(
        **{
            SCENARIO_COLUMN: pd.NamedAgg(column=SCENARIO_COLUMN, aggfunc="first"),
            ROWS_COLUMN: pd.NamedAgg(column=LABEL_COLUMN, aggfunc="size"),
            POSITIVES_COLUMN: pd.NamedAgg(column=LABEL_COLUMN, aggfunc="sum"),
        }
    )
    table[SCENARIO_COLUMN] = table[SCENARIO_COLUMN].astype("int64")
    table[ROWS_COLUMN] = table[ROWS_COLUMN].astype("int64")
    table[POSITIVES_COLUMN] = table[POSITIVES_COLUMN].astype("int64")
    return table.sort_index()


def allocate(n: int, fractions: SplitFractions = SPLIT) -> dict[str, int]:
    """Split *n* items into three whole numbers that sum to exactly *n*.

    Largest-remainder allocation. Rounding each fraction independently would
    lose or invent a user whenever the remainders do not happen to cancel --
    and losing six users out of a scenario-1 group of thirty is exactly the
    kind of thing nobody notices.

    On the real group sizes the fractions are already whole (930 -> 558/186/186,
    30 -> 18/6/6, 10 -> 6/2/2) so the remainder step never fires; it exists for
    the groups that are not, including groups smaller than the number of splits.
    """
    if n < 0:
        raise ValueError(f"cannot allocate a negative count, got {n}")
    exact = {name: n * getattr(fractions, name) for name in SPLIT_NAMES}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remaining = n - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != n:
        raise SplitError(f"allocation of {n} summed to {sum(counts.values())}")
    return counts


def split_users(
    scenarios: Mapping[str, int] | pd.Series,
    seed: int,
    fractions: SplitFractions = SPLIT,
) -> dict[str, list[str]]:
    """Assign each user to one split, stratified by scenario.

    *scenarios* maps user -> ``user_scenario``; a Series indexed by user works.
    The return value is a dict of split name -> sorted user list.

    Determinism, in three parts, because all three are needed:

    1. each scenario group is **sorted lexicographically** before it is
       touched, so the caller's iteration order cannot reach the shuffle;
    2. each group gets its **own** generator, seeded from ``(seed, scenario)``.
       Drawing every group from one stream would make the benign group's
       assignment depend on how many scenario-1 users happened to exist;
    3. the output lists are sorted, so the JSON is stable and a diff between
       two runs means something.
    """
    pairs = [(str(user), int(value)) for user, value in scenarios.items()]
    duplicates = len(pairs) - len({user for user, _ in pairs})
    if duplicates:
        raise SplitError(f"{duplicates} user(s) appear more than once in the input")

    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for scenario in sorted({value for _, value in pairs}):
        members = sorted(user for user, value in pairs if value == scenario)
        rng = np.random.default_rng([seed, scenario])
        shuffled = [members[i] for i in rng.permutation(len(members))]
        counts = allocate(len(members), fractions)
        start = 0
        for name in SPLIT_NAMES:
            assigned[name].extend(shuffled[start : start + counts[name]])
            start += counts[name]
    return {name: sorted(users) for name, users in assigned.items()}


def split_all_seeds(
    table: pd.DataFrame,
    seeds: Sequence[int] = SEEDS,
    fractions: SplitFractions = SPLIT,
) -> dict[int, dict[str, list[str]]]:
    """:func:`split_users` for every seed, keyed by seed."""
    scenarios = table[SCENARIO_COLUMN]
    return {int(seed): split_users(scenarios, int(seed), fractions) for seed in seeds}


def summarise_split(
    users: Sequence[str], table: pd.DataFrame
) -> dict[str, object]:
    """Users, rows and positives for one split, broken down by scenario.

    ``positives_by_scenario`` is not in the brief's sketch of the format and is
    here anyway: it is the number Stage 7 needs to say anything honest about
    per-scenario recall, and re-deriving it later means reopening the 16 MB
    table to answer a question this stage already knew the answer to.
    """
    rows = table.loc[list(users)]
    by_scenario = rows[SCENARIO_COLUMN].value_counts().sort_index()
    positives_by_scenario = (
        rows.groupby(SCENARIO_COLUMN)[POSITIVES_COLUMN].sum().sort_index()
    )
    total_rows = int(rows[ROWS_COLUMN].sum())
    total_positives = int(rows[POSITIVES_COLUMN].sum())
    return {
        "users": len(rows),
        "by_scenario": {str(k): int(v) for k, v in by_scenario.items()},
        "rows": total_rows,
        "positive_rows": total_positives,
        "positive_rate": (total_positives / total_rows) if total_rows else 0.0,
        "positives_by_scenario": {
            str(k): int(v) for k, v in positives_by_scenario.items()
        },
    }


def summarise(
    splits: Mapping[int, Mapping[str, Sequence[str]]], table: pd.DataFrame
) -> dict[str, dict[str, dict[str, object]]]:
    """The summary block: every seed, every split, counted."""
    return {
        str(seed): {
            name: summarise_split(assignment[name], table)
            for name in SPLIT_NAMES
        }
        for seed, assignment in splits.items()
    }


def build_payload(
    splits: Mapping[int, Mapping[str, Sequence[str]]],
    table: pd.DataFrame,
    fractions: SplitFractions = SPLIT,
) -> dict[str, object]:
    """The whole ``splits.json`` document, ready to serialise."""
    return {
        "fractions": {name: getattr(fractions, name) for name in SPLIT_NAMES},
        "seeds": sorted(int(seed) for seed in splits),
        "n_users": int(len(table)),
        "splits": {
            str(seed): {name: list(assignment[name]) for name in SPLIT_NAMES}
            for seed, assignment in splits.items()
        },
        "summary": summarise(splits, table),
    }


def dumps(payload: Mapping[str, object]) -> str:
    """Serialise with sorted keys and a stable indent, so diffs are readable."""
    return json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class Check:
    """One assertion about the split, and whether it held."""

    ok: bool
    text: str
    fatal: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        return "FAIL" if self.fatal else "warn"


def check_splits(
    splits: Mapping[int, Mapping[str, Sequence[str]]],
    table: pd.DataFrame,
    fractions: SplitFractions = SPLIT,
) -> list[Check]:
    """Every invariant this stage can be wrong about, tested on the real output.

    The last one is not fatal: on the 30-user sample, scenario 2 has two
    insiders and cannot put one in all three splits. That is a fact about the
    sample, not a bug in the split, so it is reported rather than raised.
    """
    all_users = set(table.index.astype(str))
    checks: list[Check] = []

    disjoint_failures: list[str] = []
    complete_failures: list[str] = []
    for seed, assignment in sorted(splits.items()):
        parts = [set(assignment[name]) for name in SPLIT_NAMES]
        total = sum(len(part) for part in parts)
        union: set[str] = set().union(*parts)
        if len(union) != total:
            disjoint_failures.append(f"seed {seed}")
        if union != all_users:
            complete_failures.append(
                f"seed {seed} (missing {len(all_users - union)}, "
                f"invented {len(union - all_users)})"
            )
    checks.append(
        Check(not disjoint_failures, "no user appears in two splits, in any seed")
    )
    checks.append(
        Check(
            not complete_failures,
            f"union of the three splits = all {len(all_users):,} users, in any seed",
        )
    )

    proportion_failures: list[str] = []
    for seed, assignment in sorted(splits.items()):
        for scenario, group in table.groupby(SCENARIO_COLUMN):
            expected = allocate(len(group), fractions)
            members = set(group.index.astype(str))
            for name in SPLIT_NAMES:
                got = len(members & set(assignment[name]))
                if got != expected[name]:
                    proportion_failures.append(
                        f"seed {seed} scenario {scenario} {name}: "
                        f"{got} != {expected[name]}"
                    )
    checks.append(
        Check(
            not proportion_failures,
            "every scenario is divided in the right proportion, in every seed",
        )
    )

    scenarios = sorted(int(s) for s in table[SCENARIO_COLUMN].unique())
    repeated = {
        seed: split_users(table[SCENARIO_COLUMN], int(seed), fractions)
        for seed in splits
    }
    checks.append(
        Check(
            all(
                {k: list(v) for k, v in repeated[seed].items()}
                == {k: list(v) for k, v in splits[seed].items()}
                for seed in splits
            ),
            "the same seed run twice produces an identical split",
        )
    )

    distinct = {
        seed: tuple(sorted(assignment["test"])) for seed, assignment in splits.items()
    }
    checks.append(
        Check(
            len(set(distinct.values())) == len(distinct),
            "different seeds produce different splits",
        )
    )

    thin: list[str] = []
    for seed, assignment in sorted(splits.items()):
        for scenario in scenarios:
            if scenario == 0:
                continue
            members = set(
                table.index[table[SCENARIO_COLUMN] == scenario].astype(str)
            )
            for name in SPLIT_NAMES:
                if not members & set(assignment[name]):
                    thin.append(f"seed {seed} scenario {scenario} {name}")
    checks.append(
        Check(
            not thin,
            "every split holds at least one insider of every scenario"
            + (f"   [{', '.join(thin[:4])}]" if thin else ""),
            fatal=False,
        )
    )
    return checks


def verify(
    splits: Mapping[int, Mapping[str, Sequence[str]]],
    table: pd.DataFrame,
    fractions: SplitFractions = SPLIT,
) -> list[Check]:
    """Run :func:`check_splits` and raise on any fatal failure."""
    checks = check_splits(splits, table, fractions)
    broken = [check.text for check in checks if not check.ok and check.fatal]
    if broken:
        raise SplitError("the split violates: " + "; ".join(broken))
    return checks


def scenario_columns(
    table: pd.DataFrame, scenarios: Sequence[int] = SCENARIOS
) -> tuple[int, ...]:
    """Every scenario the report must show a column for, ascending.

    The union of three things, and each is load-bearing:

    * **the values present in the table**, so no group of users can be counted
      into the totals and left out of the breakdown. This is the one that makes
      "the columns sum to the user count" a property rather than a hope, and it
      is the check that would have caught the bug this function exists to fix;
    * **``config.SCENARIOS``**, so a scenario with zero users in this
      particular table prints as a ``0`` instead of vanishing from the header;
    * **benign**, which is scenario 0 and is never in ``SCENARIOS``.

    Reading only ``SCENARIOS`` would have reinstated the original bug in a new
    costume: on the day the data holds a value config has not been told about,
    a column of users would disappear again, silently, exactly as before.
    """
    observed = {int(value) for value in table[SCENARIO_COLUMN].unique()}
    return tuple(sorted(observed | {BENIGN_SCENARIO} | {int(s) for s in scenarios}))


_BENIGN_COLUMN_WIDTH: Final[int] = 9
_SCENARIO_COLUMN_WIDTH: Final[int] = 5


def _column_heading(scenario: int) -> tuple[str, int]:
    """The heading and width for one scenario column, as one decision.

    Header and row are formatted from this single function so that they cannot
    drift apart -- which is precisely how the previous version failed: the
    header listed three scenarios and the row printed three lookups, and adding
    a fourth to the data changed neither.
    """
    if scenario == BENIGN_SCENARIO:
        return "benign", _BENIGN_COLUMN_WIDTH
    return f"s{scenario}", _SCENARIO_COLUMN_WIDTH


def _seed_lines(
    seed: int,
    summary: Mapping[str, Mapping[str, object]],
    columns: Sequence[int],
) -> list[str]:
    headings = [_column_heading(scenario) for scenario in columns]
    header = (
        f"{'':>7}{'users':>8}"
        + "".join(f"{title:>{width}}" for title, width in headings)
        + f"{'rows':>11}{'positives':>12}{'rate':>9}"
    )
    lines = [f"Seed {seed}", header]
    for name in SPLIT_NAMES:
        block = summary[name]
        by_scenario = block["by_scenario"]
        lines.append(
            f"  {name:<5}{block['users']:>7,}"
            + "".join(
                f"{by_scenario.get(str(scenario), 0):>{width},}"
                for scenario, (_, width) in zip(columns, headings)
            )
            + f"{block['rows']:>11,}"
            f"{block['positive_rows']:>12,}"
            f"{block['positive_rate']:>8.3%}"
        )
    test_positives = summary["test"]["positives_by_scenario"]
    lines.append(
        "  positive rows by scenario in TEST:   "
        + "   ".join(
            f"s{scenario} = {test_positives.get(str(scenario), 0):,}"
            for scenario in columns
            if scenario != BENIGN_SCENARIO
        )
    )
    off_band = [
        f"{name} {summary[name]['positive_rate']:.3%}"
        for name in SPLIT_NAMES
        if not (
            MIN_PLAUSIBLE_POSITIVE_RATE
            <= float(summary[name]["positive_rate"])
            <= MAX_PLAUSIBLE_POSITIVE_RATE
        )
    ]
    if off_band:
        lines.append(
            "  note: positive rate outside the "
            f"{MIN_PLAUSIBLE_POSITIVE_RATE:.1%}-{MAX_PLAUSIBLE_POSITIVE_RATE:.1%} "
            f"band for {', '.join(off_band)} -- report it, do not reshuffle"
        )
    return lines


def format_report(
    payload: Mapping[str, object],
    table: pd.DataFrame,
    checks: Sequence[Check],
    out: Path | None = None,
) -> str:
    """The stage's diagnostics, in one block, printed on every run."""
    by_scenario = table[SCENARIO_COLUMN].value_counts().sort_index()
    census = ", ".join(
        f"{int(count)} " + ("benign" if scenario == 0 else f"s{scenario}")
        for scenario, count in by_scenario.items()
    )
    summary = payload["summary"]
    seeds = payload["seeds"]
    columns = scenario_columns(table)

    lines = [
        _RULE,
        "InnerSight Stage 4 - train / validation / test split (by USER)",
        _RULE,
        f"Users                    : {len(table):,}  ({census})",
        f"Rows behind them         : {int(table[ROWS_COLUMN].sum()):,}",
        f"Positive rows            : {int(table[POSITIVES_COLUMN].sum()):,}"
        f"  ({int(table[POSITIVES_COLUMN].sum()) / max(int(table[ROWS_COLUMN].sum()), 1):.3%})",
        "Fractions                : "
        + "  ".join(
            f"{name} {float(payload['fractions'][name]):.0%}" for name in SPLIT_NAMES
        ),
        f"Seeds                    : {', '.join(str(s) for s in seeds)}",
        "",
    ]
    for seed in seeds:
        lines.extend(_seed_lines(int(seed), summary[str(seed)], columns))
        lines.append("")

    spread = _spread_lines(summary, seeds, smallest_scenario(table))
    if spread:
        lines.extend(["Across seeds:", *spread, ""])

    lines.append("Checks:")
    lines.extend(f"  {check.status:<4} {check.text}" for check in checks)
    lines.append(_RULE)
    if out is not None:
        lines.append(f"Written to: {out}")
    return "\n".join(lines)


def _spread_lines(
    summary: Mapping[str, Mapping[str, Mapping[str, object]]],
    seeds: Sequence[int],
    smallest: int | None,
) -> list[str]:
    """How much the positive counts move between seeds.

    This is the number that decides how the final table is allowed to be
    phrased. If test positives swing by a third across three seeds, a single
    reported recall is a claim about luck.

    The last line singles out the *smallest* attack scenario, whichever one
    that is. It used to say "scenario 3" because on r4.2 scenario 3 is the
    smallest -- true, and true by accident. The rule is about the group whose
    recall rests on a handful of days, not about the number 3.
    """
    if len(seeds) < 2:
        return []
    lines = []
    for name in SPLIT_NAMES:
        values = [int(summary[str(seed)][name]["positive_rows"]) for seed in seeds]
        lines.append(
            f"  {name:<5} positives {min(values):,} .. {max(values):,}   "
            f"mean {np.mean(values):,.1f}  sd {np.std(values, ddof=1):,.1f}  "
            f"spread {(max(values) - min(values)) / max(np.mean(values), 1):.0%} of mean"
        )
    if smallest is None:
        return lines
    counts = [
        int(summary[str(seed)]["test"]["positives_by_scenario"].get(str(smallest), 0))
        for seed in seeds
    ]
    lines.append(
        f"  scenario-{smallest} positive rows in TEST: "
        + ", ".join(f"seed {seed} -> {value}" for seed, value in zip(seeds, counts))
        + "   <- state this before any per-scenario recall"
    )
    return lines


def smallest_scenario(
    table: pd.DataFrame, scenarios: Sequence[int] = SCENARIOS
) -> int | None:
    """The attack scenario with the fewest positive rows in the whole table.

    **Fewest positive rows, not fewest insiders**, because the fragile quantity
    is the denominator of a recall: scenario 3 on r4.2 has ten insiders and
    only twenty attack days between them, and it is the twenty that makes a
    single flagged day worth five percentage points. Ties break towards the
    scenario with fewer insiders, then towards the lower id, so the answer is
    a function of the table and not of dictionary order.

    Benign is excluded -- it has no positives by construction. A scenario the
    release declares but the table does not contain scores zero and therefore
    wins, which is deliberate: "scenario 4 contributes 0 positive test rows" is
    the single most useful line the report could print in that situation.
    Returns ``None`` only when there is no attack scenario at all, on either
    side; the caller then drops the line rather than inventing one to name.
    """
    attack = {int(s) for s in scenarios} | {
        int(value) for value in table[SCENARIO_COLUMN].unique()
    }
    attack.discard(BENIGN_SCENARIO)
    if not attack:
        return None
    present = table[table[SCENARIO_COLUMN].isin(sorted(attack))]
    positives = (
        present.groupby(SCENARIO_COLUMN)[POSITIVES_COLUMN].sum().to_dict()
        if not present.empty
        else {}
    )
    insiders = (
        present[SCENARIO_COLUMN].value_counts().to_dict() if not present.empty else {}
    )
    return min(
        sorted(attack),
        key=lambda s: (int(positives.get(s, 0)), int(insiders.get(s, 0)), s),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Read the dataset's user column, split it, write splits.json, report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample", action="store_true",
        help="Read the 30-user sample dataset and write a sample-named output.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Dataset parquet to read. Defaults to the full/sample path.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="JSON to write. Defaults to the full/sample splits path.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "Override config.SEEDS. Provided for experiments only -- generating "
            "extra seeds and keeping the nicest is selection on the test set."
        ),
    )
    args = parser.parse_args(argv)

    dataset_path = args.dataset or (
        DATASET_SAMPLE_PATH if args.sample else DATASET_PATH
    )
    out_path = args.out or (SPLITS_SAMPLE_PATH if args.sample else SPLITS_PATH)
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not dataset_path.is_file():
        logger.error(
            "dataset not found: %s. Run `python -m baseline.baseline%s` first.",
            dataset_path, " --sample" if args.sample else "",
        )
        return 1

    logger.info("reading %s (3 columns of it)", dataset_path)
    table = read_user_table(dataset_path)
    logger.info("%d users, %d rows behind them", len(table), int(table[ROWS_COLUMN].sum()))

    splits = split_all_seeds(table, seeds)
    checks = verify(splits, table)
    payload = build_payload(splits, table)

    ensure_directories()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps(payload), encoding="utf-8")
    logger.info("wrote %s", out_path)

    print(format_report(payload, table, checks, out=out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
