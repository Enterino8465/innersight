"""Tests for Stage 4, chosen by the Stage 1/2/3 criterion.

Cover the failures that are **silent** — no exception, no NaN, just an
assignment that is subtly wrong and survives into the final numbers. A crash
here would be a gift. What this stage actually risks is a split that drifts
between runs (and quietly decouples Model A's numbers from Model B's), a split
that leaks a user across the boundary (and reports memorisation as recall), or
a stratification that loses a scenario-3 insider nobody counted.

Run from the project root::

    python -m pytest splitting/test_split.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import (
    LABEL_COLUMN,
    SCENARIO_COLUMN,
    SEEDS,
    SPLIT,
    USER_COLUMN,
    SplitFractions,
)
from splitting.split import (
    POSITIVES_COLUMN,
    ROWS_COLUMN,
    SPLIT_NAMES,
    SplitError,
    allocate,
    build_payload,
    check_splits,
    dumps,
    split_all_seeds,
    split_users,
    user_table,
    verify,
)


def make_population(
    counts: dict[int, int], days: int = 5, positives: dict[int, int] | None = None
) -> pd.DataFrame:
    """A day-grained frame for a synthetic population.

    *counts* maps scenario -> number of users. User ids are deliberately built
    so that lexicographic order is NOT the order they appear in the frame:
    the whole point of several tests below is that the input's order must not
    reach the shuffle.
    """
    positives = positives or {}
    rows = []
    for scenario, n_users in sorted(counts.items()):
        for index in range(n_users):
            user = f"S{scenario}U{index:04d}"
            n_positive = positives.get(scenario, 0)
            for day in range(days):
                rows.append(
                    {
                        USER_COLUMN: user,
                        SCENARIO_COLUMN: np.int8(scenario),
                        LABEL_COLUMN: np.int8(1 if day < n_positive else 0),
                    }
                )
    frame = pd.DataFrame(rows)
    frame[USER_COLUMN] = frame[USER_COLUMN].astype("string")
    return frame


def realistic_table() -> pd.DataFrame:
    """r4.2's own census: 930 benign and 30/30/10 insiders."""
    return user_table(make_population({0: 930, 1: 30, 2: 30, 3: 10}))


# --------------------------------------------------------------------------
# 1. Disjointness -- the leak that turns memorisation into a good score
# --------------------------------------------------------------------------


def test_no_user_lands_in_two_splits() -> None:
    table = realistic_table()
    for seed in SEEDS:
        assignment = split_users(table[SCENARIO_COLUMN], seed)
        parts = [set(assignment[name]) for name in SPLIT_NAMES]
        assert sum(len(part) for part in parts) == len(set().union(*parts))


# --------------------------------------------------------------------------
# 2. Completeness -- nobody dropped, nobody invented
# --------------------------------------------------------------------------


def test_every_user_is_assigned_exactly_once() -> None:
    table = realistic_table()
    everyone = set(table.index)
    for seed in SEEDS:
        assignment = split_users(table[SCENARIO_COLUMN], seed)
        union = set().union(*(set(assignment[name]) for name in SPLIT_NAMES))
        assert union == everyone
        assert sum(len(assignment[name]) for name in SPLIT_NAMES) == len(everyone)


# --------------------------------------------------------------------------
# 3. Stratification -- and specifically that scenario 3 survives it
# --------------------------------------------------------------------------


def test_each_scenario_is_divided_in_the_right_proportion() -> None:
    table = realistic_table()
    expected = {0: (558, 186, 186), 1: (18, 6, 6), 2: (18, 6, 6), 3: (6, 2, 2)}
    for seed in SEEDS:
        assignment = split_users(table[SCENARIO_COLUMN], seed)
        for scenario, wanted in expected.items():
            members = set(table.index[table[SCENARIO_COLUMN] == scenario])
            got = tuple(
                len(members & set(assignment[name])) for name in SPLIT_NAMES
            )
            assert got == wanted, f"seed {seed} scenario {scenario}: {got}"


def test_scenario_three_puts_an_insider_in_every_split() -> None:
    """Ten insiders, two attack days each. A stratification that groups all
    insiders together can hand test zero of them and make scenario-3 recall
    undefined without raising anything."""
    table = realistic_table()
    members = set(table.index[table[SCENARIO_COLUMN] == 3])
    for seed in SEEDS:
        assignment = split_users(table[SCENARIO_COLUMN], seed)
        for name in SPLIT_NAMES:
            assert members & set(assignment[name]), f"seed {seed}, {name} has no s3"


# --------------------------------------------------------------------------
# 4. Determinism -- same seed, same split, byte for byte
# --------------------------------------------------------------------------


def test_the_same_seed_twice_gives_a_byte_identical_document() -> None:
    table = realistic_table()
    first = dumps(build_payload(split_all_seeds(table), table))
    second = dumps(build_payload(split_all_seeds(table), table))
    assert first == second


# --------------------------------------------------------------------------
# 5. Order independence -- the sort-before-shuffle bug, and why it is here
# --------------------------------------------------------------------------


def test_the_input_order_cannot_reach_the_shuffle() -> None:
    """The classic silent bug: shuffling users in whatever order they came out
    of the parquet file. That order is not guaranteed, so the same seed gives a
    different split on a different machine and nothing warns you — while
    Stages 5-7 all read this file and stop being comparable."""
    frame = make_population({0: 100, 1: 10, 2: 10, 3: 10})
    forward = user_table(frame)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    shuffled_frame = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    for other in (user_table(reversed_frame), user_table(shuffled_frame)):
        for seed in SEEDS:
            assert split_users(forward[SCENARIO_COLUMN], seed) == split_users(
                other[SCENARIO_COLUMN], seed
            )


def test_a_dict_input_in_a_different_insertion_order_gives_the_same_split() -> None:
    table = realistic_table()
    forward = {str(user): int(s) for user, s in table[SCENARIO_COLUMN].items()}
    backward = dict(reversed(list(forward.items())))
    assert split_users(forward, 42) == split_users(backward, 42)


def test_one_scenario_group_does_not_depend_on_the_others() -> None:
    """Each group gets its own generator. Drawing every group from one stream
    would make the benign assignment depend on how many scenario-1 users
    happened to exist, so an upstream change to one scenario would silently
    reshuffle all the others."""
    full = realistic_table()
    without_s1 = full[full[SCENARIO_COLUMN] != 1]
    a = split_users(full[SCENARIO_COLUMN], 42)
    b = split_users(without_s1[SCENARIO_COLUMN], 42)
    benign = set(full.index[full[SCENARIO_COLUMN] == 0])
    for name in SPLIT_NAMES:
        assert benign & set(a[name]) == benign & set(b[name])


# --------------------------------------------------------------------------
# 6. Seeds actually differ
# --------------------------------------------------------------------------


def test_different_seeds_produce_different_splits() -> None:
    table = realistic_table()
    assignments = {
        seed: tuple(split_users(table[SCENARIO_COLUMN], seed)["test"])
        for seed in SEEDS
    }
    assert len(set(assignments.values())) == len(SEEDS)


# --------------------------------------------------------------------------
# 7. Tiny inputs -- no crash, and above all no silent drop
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", list(range(0, 13)))
def test_allocation_of_a_tiny_group_sums_to_the_group(n: int) -> None:
    counts = allocate(n)
    assert sum(counts.values()) == n
    assert all(value >= 0 for value in counts.values())


def test_a_scenario_smaller_than_the_number_of_splits_keeps_everyone() -> None:
    table = user_table(make_population({0: 5, 1: 1, 2: 2, 3: 3}))
    assignment = split_users(table[SCENARIO_COLUMN], 42)
    union = set().union(*(set(assignment[name]) for name in SPLIT_NAMES))
    assert union == set(table.index)
    assert sum(len(assignment[name]) for name in SPLIT_NAMES) == len(table)


def test_an_empty_population_does_not_crash() -> None:
    empty = pd.Series(dtype="int64")
    assignment = split_users(empty, 42)
    assert assignment == {name: [] for name in SPLIT_NAMES}


# --------------------------------------------------------------------------
# The summary block, and the guards around the input
# --------------------------------------------------------------------------


def test_the_summary_counts_rows_and_positives_not_users() -> None:
    """Two users of the same scenario can carry wildly different positive
    counts -- scenario 2 averages 56 attack days and scenario 3 exactly 2 --
    so a summary that counted users would say nothing about class balance."""
    table = user_table(
        make_population({0: 6, 3: 6}, days=10, positives={3: 2})
    )
    payload = build_payload(split_all_seeds(table, (42,)), table)
    summary = payload["summary"]["42"]
    assert sum(summary[name]["users"] for name in SPLIT_NAMES) == 12
    assert sum(summary[name]["rows"] for name in SPLIT_NAMES) == 120
    assert sum(summary[name]["positive_rows"] for name in SPLIT_NAMES) == 12
    # allocate(6) is 4/1/1 -- 0.6*6 = 3.6, and the largest remainder goes to
    # train -- so train holds four s3 users at two attack days each.
    assert summary["train"]["by_scenario"]["3"] == 4
    assert summary["train"]["positives_by_scenario"]["3"] == 8
    assert summary["train"]["rows"] == 80  # 8 users x 10 days
    assert summary["train"]["positive_rate"] == pytest.approx(8 / 80)


def test_the_document_carries_every_seed_and_every_user() -> None:
    table = realistic_table()
    payload = build_payload(split_all_seeds(table), table)
    assert payload["n_users"] == 1000
    assert payload["seeds"] == sorted(SEEDS)
    for seed in SEEDS:
        assert sum(
            len(payload["splits"][str(seed)][name]) for name in SPLIT_NAMES
        ) == 1000


def test_a_scenario_that_changes_within_a_user_is_a_hard_failure() -> None:
    """user_scenario is per user by construction. If it is not, something
    merged insiders.csv at the wrong grain and every stratified count below is
    meaningless -- so this must raise rather than pick a value."""
    frame = make_population({0: 3, 1: 1})
    frame.loc[0, SCENARIO_COLUMN] = np.int8(2)
    with pytest.raises(SplitError):
        user_table(frame)


def test_verify_raises_on_a_leaked_user() -> None:
    table = realistic_table()
    splits = split_all_seeds(table)
    leaked = {
        seed: {name: list(users) for name, users in assignment.items()}
        for seed, assignment in splits.items()
    }
    leaked[42]["test"].append(leaked[42]["train"][0])
    with pytest.raises(SplitError):
        verify(leaked, table)


def test_a_thin_scenario_is_reported_but_not_fatal() -> None:
    """The 30-user sample has two scenario-2 insiders and cannot put one in
    all three splits. That is a fact about the sample, not a broken split."""
    table = user_table(make_population({0: 21, 1: 3, 2: 2, 3: 4}))
    splits = split_all_seeds(table)
    checks = check_splits(splits, table)
    coverage = [c for c in checks if "at least one insider" in c.text][0]
    assert not coverage.ok and not coverage.fatal
    verify(splits, table)  # must not raise


def test_fractions_other_than_60_20_20_still_partition_the_population() -> None:
    table = realistic_table()
    fractions = SplitFractions(train=0.7, val=0.15, test=0.15)
    assignment = split_users(table[SCENARIO_COLUMN], 42, fractions)
    union = set().union(*(set(assignment[name]) for name in SPLIT_NAMES))
    assert union == set(table.index)
    assert len(assignment["train"]) == 700


def test_the_json_is_sorted_and_therefore_diffable() -> None:
    table = user_table(make_population({0: 10, 1: 5}))
    payload = build_payload(split_all_seeds(table), table)
    text = dumps(payload)
    assert text.endswith("\n")
    top_level_keys = [line for line in text.splitlines() if line.startswith('  "')]
    assert top_level_keys == sorted(top_level_keys)
    for seed in SEEDS:
        for name in SPLIT_NAMES:
            users = payload["splits"][str(seed)][name]
            assert users == sorted(users)
