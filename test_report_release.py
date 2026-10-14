"""What ``evaluation/report.py`` must say differently on a different release.

Brief R8 §6. Until 2026-08-22 this module wrote r4.2's name, r4.2's population
and r4.2's findings whatever release it was rendering, so
``releases/r52/outputs/final_results.json`` -- the machine-readable deliverable
-- carried r5.2's numbers under the string *"insider threat detection on CERT
r4.2"*.

**Why this file is here and not in ``evaluation/``.** ``python -m pytest
evaluation/ modeling/ -q`` must keep reporting exactly 101. That number is the
fingerprint of the frozen results and it is a hard stop across every brief in
this project, so a new test about ``evaluation/report.py`` goes beside the
other root-level tests rather than inside the package it is about.

**Why subprocesses.** ``config`` reads ``INNERSIGHT_RELEASE`` once, at import,
and ``report`` binds the constants it needs at *its* import. Monkeypatching
``report.ACTIVE_RELEASE`` would test the branch and skip the wiring, which is
the half that was actually broken. Each rendering below is a fresh interpreter
with the variable set or removed -- the same argument, and the same helper
shape, as ``test_config.py``.

Two renderings, three questions:

1. Do the release-varying facts follow the release? (They must.)
2. Do the r4.2-only measurements stay out of the r5.2 rendering? (They must --
   nobody has measured their equivalents, and inventing one would be the worst
   thing anyone could ship into this project.)
3. Does everything else agree? (A release-aware report that quietly rewrote
   its argument would be harder to trust than one that never learned.)

Run it with::

    python -m pytest test_report_release.py -q
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import pytest

ROOT = Path(__file__).resolve().parent

#: The two percentages this brief took out of the report's prose: r4.2's
#: per-day positive rate and its complement. Narrow on purpose -- the file
#: still types percentages that are not positive rates (the Stage 3 floor's
#: 0.1%, scenario 2's 88% of positives) and those are r4.2 measurements that
#: are labelled rather than derived.
_TYPED_RATE: Final[re.Pattern[str]] = re.compile(r"\b(?:0\.4|99\.6)%")

#: Rendered by a fresh interpreter under one release, and printed as JSON so
#: the parent process never imports ``config`` and therefore never has an
#: opinion about which release is active.
_RENDER: Final[str] = """
import json
import config
from evaluation import report

sources = report.Sources.load()
print(json.dumps({
    "release": config.ACTIVE_RELEASE,
    "insiders": config.EXPECTED_INSIDER_COUNT,
    "employees": config.EXPECTED_EMPLOYEE_COUNT,
    "markdown": report.build(sources),
    "project": report.final_results(sources)["project"],
    "test_baseline": sources.phase2["headline"]["model_a"]["random_baseline"]["mean"],
    "ablation_baseline":
        sources.ablation_a["aggregate"]["random_baseline"]["mean"],
    "per_scenario": sources.phase2["per_scenario_pooled"],
    "profiles": {
        str(key): [profile.kind, profile.min_days, profile.max_days]
        for key, profile in config.SCENARIO_PROFILES.items()
    },
}))
"""

#: A day range written into the prose, and a scenario's shape written beside a
#: length. Deliberately narrow, in the same spirit as ``_TYPED_RATE``: the
#: report legitimately says "28-day window" (a model's scope, not a scenario's
#: length) and "days 1-7" (a latency bucket), and neither is a claim about a
#: release. What must never be typed again is "45–59 day" or "2-day burst".
_TYPED_SCENARIO: Final[re.Pattern[str]] = re.compile(
    r"\d+\s*[–-]\s*\d+[\s-]*days?"
    r"|\d+[\s-]*days?[\s-]*(?:burst|drift|escalation)"
)

#: Phrases that are r4.2 measurements and nothing else. Each was typed into
#: this report as a fact about "the dataset"; each is a fact about ONE dataset.
#: A failure here does not mean the wording drifted -- it means an r5.2 report
#: is claiming something nobody measured on r5.2.
_R42_ONLY: Final[tuple[str, ...]] = (
    "15 of the 18 features",   # Stage 1's coverage pass, r4.2 only
    "0.9369",                  # LAN (Cai et al., 2024), published on r4.2
    "Cai et al.",
    "BIH0745",                 # an r4.2 user; no counterpart in r5.2
    "KPC0073",
)


def _render(release: str | None) -> dict[str, Any]:
    """Render the whole report under one release, in a fresh interpreter.

    ``release=None`` removes the variable rather than setting it empty, so the
    default path is exercised the way a clean shell exercises it.
    """
    environment = dict(os.environ)
    environment.pop("INNERSIGHT_RELEASE", None)
    if release is not None:
        environment["INNERSIGHT_RELEASE"] = release

    completed = subprocess.run(
        [sys.executable, "-c", _RENDER],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"rendering the report under INNERSIGHT_RELEASE={release!r} exited "
            f"{completed.returncode}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def r42() -> dict[str, Any]:
    return _render(None)


@pytest.fixture(scope="module")
def r52() -> dict[str, Any]:
    return _render("5.2")


# --------------------------------------------------------------------------
# 1. The release-varying facts follow the release
# --------------------------------------------------------------------------


def test_the_deliverable_names_the_release_that_produced_it(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """The defect this brief opened with, asserted as data.

    ``final_results.json`` is the machine-readable deliverable. An examiner who
    opens r5.2's copy must not find r5.2's numbers under r4.2's name.
    """
    assert r42["project"] == "InnerSight — insider threat detection on CERT r4.2"
    assert r52["project"] == "InnerSight — insider threat detection on CERT r5.2"


def test_the_population_is_the_active_releases(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """70 among 1,000 and 99 among 2,000 -- read off config, not typed.

    The percentage is asserted too, because it is the number most likely to be
    left behind: it was typed as "7%" and is now insiders / employees, which on
    r5.2 is 4.95% rather than a rounder-looking 5%.
    """
    for snapshot, insiders, employees, rate in (
        (r42, "70", "1,000", "7.00%"),
        (r52, "99", "2,000", "4.95%"),
    ):
        limitations = _flat(_section(snapshot["markdown"], "## 8. Limitations"))
        assert f"carries {insiders} insiders among {employees} users" in limitations
        assert f"a {rate} rate" in limitations
        assert snapshot["insiders"] / snapshot["employees"] == pytest.approx(
            float(rate.rstrip("%")) / 100, abs=5e-5
        ), "the rate in the prose must be insiders / employees, not a typed number"


def test_the_other_releases_population_never_appears(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """The failure mode is a stale number, not a missing one.

    Asserting "99 is present" would pass on a report that also still said 70.
    """
    assert "carries 99 insiders" not in _flat(r42["markdown"])
    assert "carries 70 insiders" not in _flat(r52["markdown"])


def test_every_quoted_positive_rate_matches_this_release_s_artifacts(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """It was typed as "0.4%" in five places, and it is a measured quantity.

    ``random_baseline`` *is* the positive rate -- the average precision of a
    model that ranks at random equals the fraction of rows that are positive --
    so the report has always carried the number it was quoting a constant
    beside.

    **This test cannot tell a derived rate from a typed one on today's data,
    and that is worth writing down.** Measured 2026-08-22: r4.2's test split is
    0.00428 positive and r5.2's is 0.00443, which are 0.4% and 0.4%. The
    parameterisation is real and currently invisible. What makes it checkable
    is the test below, which reads the source.
    """
    for snapshot in (r42, r52):
        release = snapshot["release"]
        quoted = set(_rates_in(snapshot["markdown"]))
        assert quoted, f"r{release} quotes no positive rate at all"
        allowed = {
            f"{snapshot['test_baseline']:.1%}",
            f"{snapshot['ablation_baseline']:.1%}",
        }
        assert quoted <= allowed, (
            f"r{release} quotes {sorted(quoted - allowed)} as a positive rate; "
            f"its artifacts measured {sorted(allowed)}"
        )


def test_no_positive_rate_is_typed_into_the_module() -> None:
    """The check the one above cannot make, made against the source instead.

    Two releases whose positive rates agree to one decimal place cannot
    distinguish a derived number from a surviving literal. The source can:
    every rate in this report is now formatted from ``random_baseline``, so
    either of the two literals this brief removed -- ``0.4%`` and its
    complement ``99.6%`` -- reappearing in the prose means one site was missed.

    Deliberately narrow. A regex for *any* percentage would fire on the Stage 3
    floor's 0.1% and on "88% of positives", which are r4.2 measurements that
    §6.1 says to label rather than to derive. This is a regression guard for
    the five sites that moved, not a rule about percentages.

    Comments and docstrings are exempt -- this module's comments name the
    literals it *used* to carry, and punishing that would be punishing the
    record of the fix.
    """
    offenders = [
        literal
        for literal in _prose_literals(ROOT / "evaluation" / "report.py")
        if _TYPED_RATE.search(literal)
    ]
    assert offenders == [], (
        f"a positive rate is typed into the report's prose: {offenders}. It "
        f"comes off random_baseline now, in every place it appears."
    )


# --------------------------------------------------------------------------
# 2. The r4.2 measurements stay on r4.2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", _R42_ONLY)
def test_no_r42_measurement_appears_in_an_r52_report(
    phrase: str, r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """The rule this brief cared about most, one phrase at a time.

    Both directions are asserted. "absent from r5.2" alone would also pass if
    the phrase had been deleted outright, and deleting r4.2's measurements to
    make an r5.2 report tidy would lose the finding rather than scope it.
    """
    assert phrase in _flat(r42["markdown"]), (
        f"{phrase!r} is an r4.2 measurement and belongs in the r4.2 report"
    )
    assert phrase not in _flat(r52["markdown"]), (
        f"{phrase!r} was measured on r4.2 and appears in the r5.2 report. "
        f"Nobody has measured its equivalent on r5.2; saying so is the honest "
        f"output, and swapping in a number would be the dishonest one."
    )


def test_r52_says_what_has_not_been_measured(r52: dict[str, Any]) -> None:
    """A hole a reader can see is worth more than a report that looks complete.

    That is this module's own rule for a missing input, applied to a missing
    *measurement*. Each gated block is replaced by a sentence, never by
    silence -- a section that simply vanished would read as "nothing to say".
    """
    limitations = _flat(_section(r52["markdown"], "## 8. Limitations"))
    assert "has not been measured" in limitations
    assert "has not been run here" in limitations
    assert "the equivalent inspection has not been done here" in limitations

    gap = _flat(_section(r52["markdown"], "## 6. The AUROC/AUPRC gap"))
    assert "measured on r4.2" in gap, (
        "the absent anchor must say why it is absent, or a reader will read "
        "the omission as an oversight"
    )


def test_future_work_does_not_propose_the_report_it_is_printed_in(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """r4.2 wanted r5.2 as a generalisation test. r5.2 *is* that test.

    Carried over unchanged, §9 of the r5.2 report proposed as future work the
    thing the reader was holding.
    """
    r42_future = _flat(_section(r42["markdown"], "## 9. Future work"))
    r52_future = _flat(_section(r52["markdown"], "## 9. Future work"))
    assert "r5.2 as a" in r42_future
    assert "r5.2 as a" not in r52_future


# --------------------------------------------------------------------------
# 3. Everything else agrees
# --------------------------------------------------------------------------


def test_both_renderings_have_the_same_sections(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """Same argument, same shape. Only the measurements differ.

    A release-aware report that also reorganised itself would be a second
    variable in a comparison whose whole method is changing one at a time.
    """
    assert _headings(r42["markdown"]) == _headings(r52["markdown"])


def test_the_r42_rendering_still_states_its_own_measurements(
    r42: dict[str, Any]
) -> None:
    """Nothing was gated away from the release it was measured on.

    The r4.2 report is a frozen deliverable. Making the module release-aware
    must not have quietly thinned it.
    """
    markdown = _flat(r42["markdown"])
    assert "`large_attachment_count` is identically zero" in markdown
    assert "65-minute labelled window" in markdown
    assert "0.6767 to 0.6022" in markdown


# --------------------------------------------------------------------------
# 4. Every scenario in the results gets a row  (brief R9)
# --------------------------------------------------------------------------


def test_every_scenario_in_the_results_gets_a_row(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """The defect: §2's table was built from a hardcoded three-key dict.

    r4.2 has three attack scenarios and r5.2 has four, so on r5.2 the table
    rendered three rows and dropped scenario 4 -- 1,378 of 2,451 attack days,
    56% of the release, and Model B's strongest result in the project.

    Asserted against the artifact rather than against the number four: the
    rule is "iterate over what was measured", and a test that says "four"
    would pass on a table hardcoded to four.
    """
    for snapshot in (r42, r52):
        measured = sorted(snapshot["per_scenario"]["model_a"], key=int)
        assert _scenario_rows(snapshot["markdown"]) and (
            list(_scenario_rows(snapshot["markdown"])) == measured
        ), (
            f"r{snapshot['release']} measured scenarios {measured} and the "
            f"report shows {list(_scenario_rows(snapshot['markdown']))}"
        )
    assert len(_scenario_rows(r42["markdown"])) == 3
    assert len(_scenario_rows(r52["markdown"])) == 4


def test_every_count_in_the_table_is_the_artifact_s_count(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """The numbers are the byte-identity clause and this brief does not move
    them. Recomputed from the pooled artifact, per model, per scenario."""
    for snapshot in (r42, r52):
        rows = _scenario_rows(snapshot["markdown"])
        for model, column in (("model_a", 0), ("model_b", 1)):
            for scenario, entry in snapshot["per_scenario"][model].items():
                hits, total = entry["hits_summed"], entry["attack_days_summed"]
                assert rows[scenario][column] == (
                    f"{hits} of {total} ({hits / total:.1%})"
                ), f"r{snapshot['release']} scenario {scenario}, {model}"


def test_the_r42_counts_are_exactly_what_they_were(r42: dict[str, Any]) -> None:
    """The frozen deliverable's six numbers, typed out on purpose.

    The test above proves the table agrees with the artifact; this one proves
    the artifact is the one r4.2's report was written from. Labels changed
    under this brief. Counts did not, and this is what says so.
    """
    rows = _scenario_rows(r42["markdown"])
    assert [rows[key][0] for key in ("1", "2", "3")] == [
        "47 of 128 (36.7%)", "211 of 989 (21.3%)", "4 of 12 (33.3%)",
    ]
    assert [rows[key][1] for key in ("1", "2", "3")] == [
        "90 of 128 (70.3%)", "649 of 989 (65.6%)", "6 of 12 (50.0%)",
    ]


def test_scenario_4_is_in_the_r52_report(r52: dict[str, Any]) -> None:
    """The row the report never had, with the numbers that make it the point."""
    row = _scenario_rows(r52["markdown"])["4"]
    assert row == ["166 of 1378 (12.0%)", "1175 of 1378 (85.3%)"]
    assert "escalation" in _scenario_labels(r52["markdown"])["4"]


def test_every_scenario_label_is_the_release_profile(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """The labels are derived, and they are derived per release.

    r4.2's read "sharp 1–12 day burst"; "sharp" and "slow" are editorial words
    that exist nowhere in the data, and the day ranges were r4.2's whichever
    release was rendering. Scenario 1 runs 1–10 days on r5.2 and scenario 2
    runs 44–60, so the old dict was not merely incomplete there -- it was
    wrong about the scenarios it did cover.
    """
    for snapshot in (r42, r52):
        labels = _scenario_labels(snapshot["markdown"])
        for scenario, (kind, low, high) in snapshot["profiles"].items():
            span = f"{low}" if low == high else f"{low}–{high}"
            assert labels[scenario] == f"**{scenario}** — {kind} ({span} days)"
    assert _scenario_labels(r42["markdown"])["1"] == "**1** — burst (1–12 days)"
    assert _scenario_labels(r52["markdown"])["1"] == "**1** — burst (1–10 days)"


def test_the_report_and_the_figure_finally_agree() -> None:
    """One rule, one source, two shapes.

    ``evaluation/figures.py`` has built this label from ``SCENARIO_PROFILES``
    since brief R3; the report typed its own words, and on r4.2 the two
    disagreed about scenarios that exist in both. The helpers stay separate
    because the figure's label is two lines -- right for an x-axis tick, fatal
    to a markdown table row -- so this is what holds them to the same wording.

    In-process, over whichever release the environment selects, so it holds
    for both rather than for the one this file renders first.
    """
    from evaluation import figures, report  # noqa: PLC0415 - see module docstring

    assert report.SCENARIO_PROFILES, "no release has an empty scenario profile"
    for scenario in map(str, report.SCENARIO_PROFILES):
        assert report._scenario_label(scenario).replace("*", "") == (
            figures.scenario_label(scenario).replace("\n", " ")
        )


def test_a_scenario_with_no_profile_still_gets_a_row() -> None:
    """A missing label is cosmetic. A missing row is a lost finding.

    The same rule ``figures.py`` states for its bars: a scenario the release
    profile does not describe still gets its id and its counts. Built by hand
    rather than rendered, because no supported release has a results scenario
    its own profile omits -- scenario 9 exists in neither -- which is exactly
    why the branch needs a test rather than an assumption.
    """
    from types import SimpleNamespace

    from evaluation import report  # noqa: PLC0415 - see module docstring

    pooled = {
        model: {
            "1": {"hits_summed": 1, "attack_days_summed": 2},
            "9": {"hits_summed": 3, "attack_days_summed": 4},
        }
        for model in ("model_a", "model_b")
    }
    sources = SimpleNamespace(
        phase2={"per_scenario_pooled": pooled, "seeds": [42],
                "headline": {"model_a": {"random_baseline": {"mean": 0.004}}}}
    )
    sources.phase2["per_scenario_pooled"]["model_a"]["3"] = {
        "hits_summed": 0, "attack_days_summed": 12,
    }
    sources.phase2["per_scenario_pooled"]["model_b"]["3"] = {
        "hits_summed": 0, "attack_days_summed": 12,
    }
    rendered = report._per_scenario(sources)  # type: ignore[arg-type]
    rows = _scenario_rows(rendered)
    assert list(rows) == ["1", "3", "9"], "scenario 9 has no profile row"
    assert rows["9"] == ["3 of 4 (75.0%)", "3 of 4 (75.0%)"]
    assert _scenario_labels(rendered)["9"] == "**9**", (
        "an undescribed scenario gets its id and its counts, not a guess"
    )


def test_no_scenario_name_or_day_range_is_typed_into_the_module() -> None:
    """The source check, in the same spirit as the positive-rate one above.

    Two releases can agree on a rendered string by coincidence; the source
    cannot. Every day range in this report now comes off
    ``SCENARIO_PROFILES``, so a literal "45–59 days" reappearing in the prose
    means a site was missed -- as one in §3 was, which the §2 fix would have
    left saying r4.2's window on an r5.2 report.

    Comments and docstrings are exempt: this module's comments name the
    literals it used to carry, and punishing that punishes the record.
    """
    offenders = [
        literal
        for literal in _prose_literals(ROOT / "evaluation" / "report.py")
        if _TYPED_SCENARIO.search(literal)
    ]
    assert offenders == [], (
        f"a scenario's shape or window is typed into the report: {offenders}. "
        f"Both come off config.SCENARIO_PROFILES now."
    )


def test_the_latency_paragraph_states_this_release_s_drift_window(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """§3's opening sentence was the fifth typed r4.2 day range in the file."""
    assert "Scenario-2 attacks run 45–59 days." in _flat(
        _section(r42["markdown"], "## 3. Detection latency")
    )
    assert "Scenario-2 attacks run 44–60 days." in _flat(
        _section(r52["markdown"], "## 3. Detection latency")
    )


def test_scenario_2_s_share_of_positives_is_derived(
    r42: dict[str, Any], r52: dict[str, Any]
) -> None:
    """"Scenario 2 is 88% of positives" was typed, and it is an r4.2 fact.

    On r5.2 the fourth scenario takes most of that mass and the same
    expression reads 39%. The sentence around it is still an r4.2 validation
    measurement, so the share now names the release it belongs to.
    """
    for snapshot, expected in ((r42, "88%"), (r52, "39%")):
        pooled = snapshot["per_scenario"]["model_a"]
        total = sum(entry["attack_days_summed"] for entry in pooled.values())
        share = f"{pooled['2']['attack_days_summed'] / total:.0%}"
        assert share == expected, "this test's own arithmetic is out of date"
        assert (
            f"Scenario 2 is {share} of r{snapshot['release']}'s positives"
            in _flat(_section(snapshot["markdown"], "## 8. Limitations"))
        )
    assert "is 88% of" not in _flat(r52["markdown"])


# --------------------------------------------------------------------------
# Reading a rendered report
# --------------------------------------------------------------------------


def _scenario_table(markdown: str) -> list[list[str]]:
    """§2's table body, one list of cells per row."""
    section = _section(markdown, "## 2. Per-scenario recall")
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| Scenario")
    ]


def _scenario_rows(markdown: str) -> dict[str, list[str]]:
    """§2's counts, keyed by scenario id: ``{"1": [model_a, model_b]}``."""
    return {_row_id(row): row[1:] for row in _scenario_table(markdown)}


def _scenario_labels(markdown: str) -> dict[str, str]:
    """§2's first column, keyed by scenario id."""
    return {_row_id(row): row[0] for row in _scenario_table(markdown)}


def _row_id(row: list[str]) -> str:
    return row[0].split("**")[1]


def _prose_literals(path: Path) -> list[str]:
    """Every string literal in a module that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(getattr(node, "body", None), list)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        literal
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(literal := node.value, str)
        and id(node) not in docstrings
    ]


def _flat(markdown: str) -> str:
    """One long line, so a phrase is not missed because it was wrapped.

    Every prose assertion in this file goes through here. The report is
    hand-wrapped at roughly 78 columns and interpolating a value moves the
    wrap points, so ``"carries 70 insiders among 1,000 users" in markdown``
    is otherwise a test of the line breaks rather than of the sentence.
    """
    return " ".join(markdown.split())


def _headings(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("## ")]


def _section(markdown: str, heading: str) -> str:
    """One ``## `` section's text, so a phrase is asserted where it belongs.

    Searching the whole document would let a phrase move from §8 to §2 without
    any test noticing.
    """
    lines = markdown.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(heading)]
    assert starts, f"no section starting {heading!r}; this test is out of date"
    start = starts[0]
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _rates_in(markdown: str) -> list[str]:
    """Every ``N.N%`` on a line that calls it a positive rate."""
    return [
        match.group()
        for line in markdown.splitlines()
        if "positive rate" in line
        for match in re.finditer(r"\d+\.\d+%", line)
    ]
