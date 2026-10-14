"""Shared ground-truth window helper for Brief M1 (§5).

One definition of the window-concentration test, imported by every measurement
script in this directory so that all reported numbers are comparable.

Conventions fixed by the brief:

* Attack windows are compared at DAY granularity, inclusive at both ends.
  A 65-minute window still marks its whole calendar day.
* ``answers/insiders.csv`` covers all ten CERT releases and MUST be filtered
  first.  The ``dataset`` column is read as TEXT and compared to "5.2" / "4.2";
  parsed as a float, release identity starts behaving by luck.
* "Out-of-window" means every other event by the SAME population (the labelled
  insiders of that release) -- not by the whole company.

Days are carried around as the raw 10-character ``MM/DD/YYYY`` prefix of the
CSV ``date`` field.  Every shipped log uses that zero-padded format, so no date
parsing is needed on the multi-gigabyte passes; ``check_date_format`` verifies
the assumption on every chunk rather than trusting it.

This module is READ-ONLY with respect to the rest of the project.  It imports
nothing from the pipeline and writes only into ``measurements/cache``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

DATASET_ROOT = HERE.parent.parent / "innersight_dataset"
ANSWERS = DATASET_ROOT / "answers"

RELEASE_DIR: dict[str, Path] = {
    "4.2": DATASET_ROOT / "r4.2 copy",
    "5.2": DATASET_ROOT / "r5.2",
    "4.2s": DATASET_ROOT / "r4.2_sample",
    "5.2s": DATASET_ROOT / "r5.2_sample",
}

BASE_RELEASE = {"4.2": "4.2", "5.2": "5.2", "4.2s": "4.2", "5.2s": "5.2"}

DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
DAY_LEN = 10


def log_path(release: str, log: str) -> Path:
    return RELEASE_DIR[release] / f"{log}.csv"


def progress(msg: str) -> None:
    """Print immediately -- a long pass must never look like a hang."""
    print(msg, flush=True)


def load_insiders(release: str) -> pd.DataFrame:
    """The labelled attack windows for one release, day-granular.

    Returns columns: user, scenario (int), start_ts, end_ts, start_day, end_day
    (``datetime.date``), n_days (inclusive).
    """
    base = BASE_RELEASE[release]
    df = pd.read_csv(ANSWERS / "insiders.csv", dtype=str)
    df = df[df["dataset"] == base].copy()
    if df.empty:
        raise ValueError(f"no ground-truth rows for dataset={base!r}")
    df["scenario"] = df["scenario"].astype(int)
    df["start_ts"] = pd.to_datetime(df["start"], format=DATE_FORMAT)
    df["end_ts"] = pd.to_datetime(df["end"], format=DATE_FORMAT)
    df["start_day"] = df["start_ts"].dt.normalize()
    df["end_day"] = df["end_ts"].dt.normalize()
    df["n_days"] = (df["end_day"] - df["start_day"]).dt.days + 1
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class GroundTruth:
    """Everything the concentration test needs, precomputed once."""

    release: str
    insiders: pd.DataFrame
    scenario_of: dict[str, int]
    attack_keys: frozenset[str]
    attack_days_per_user: dict[str, int]

    @property
    def users(self) -> frozenset[str]:
        return frozenset(self.scenario_of)

    def scenarios(self) -> list[int]:
        return sorted(set(self.scenario_of.values()))


def ground_truth(release: str) -> GroundTruth:
    ins = load_insiders(release)
    scenario_of: dict[str, int] = {}
    attack_keys: set[str] = set()
    attack_days_per_user: dict[str, int] = {}
    for row in ins.itertuples(index=False):
        scenario_of[row.user] = int(row.scenario)
        days = pd.date_range(row.start_day, row.end_day, freq="D")
        attack_days_per_user[row.user] = len(days)
        for day in days:
            attack_keys.add(f"{row.user}|{day.strftime('%m/%d/%Y')}")
    return GroundTruth(
        release=release,
        insiders=ins,
        scenario_of=scenario_of,
        attack_keys=frozenset(attack_keys),
        attack_days_per_user=attack_days_per_user,
    )


def user_spans(release: str) -> pd.DataFrame:
    """First and last calendar day each user appears in ``logon.csv``.

    logon.csv is the smallest log that covers every employee, so it is the
    honest per-user calendar: it shrinks for users lost to LDAP attrition
    instead of charging everyone the full release span.

    Cached, because several scripts need it.  Columns: user, first_day,
    last_day, n_days (inclusive).
    """
    cache = CACHE / f"user_spans_{release.replace('.', '_')}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["first_day", "last_day"])

    path = log_path(release, "logon")
    progress(f"[spans] scanning {path.name} for per-user calendar spans ...")
    lo: dict[str, str] = {}
    hi: dict[str, str] = {}
    rows = 0
    for chunk in pd.read_csv(
        path, usecols=["date", "user"], dtype=str, chunksize=1_000_000
    ):
        rows += len(chunk)
        check_date_format(chunk["date"], path.name)
        d = chunk["date"]
        sortable = d.str[6:10] + d.str[0:2] + d.str[3:5]
        g = sortable.groupby(chunk["user"])
        for user, v in g.min().items():
            if user not in lo or v < lo[user]:
                lo[user] = v
        for user, v in g.max().items():
            if user not in hi or v > hi[user]:
                hi[user] = v
        progress(f"[spans]   {rows:,} rows")

    out = pd.DataFrame(
        {
            "user": list(lo),
            "first_day": [pd.Timestamp(lo[u]) for u in lo],
            "last_day": [pd.Timestamp(hi[u]) for u in lo],
        }
    ).sort_values("user")
    out["n_days"] = (out["last_day"] - out["first_day"]).dt.days + 1
    out.to_csv(cache, index=False)
    progress(f"[spans] {len(out):,} users, cached -> {cache}")
    return out


def check_date_format(series: pd.Series, where: str) -> None:
    """Fail loudly if a date field is not zero-padded ``MM/DD/YYYY HH:MM:SS``.

    The day-prefix trick used everywhere here is only valid if it is.
    """
    bad = series[(series.str.len() != 19) | (series.str[2] != "/") | (series.str[5] != "/")]
    if len(bad):
        raise ValueError(f"{where}: {len(bad)} rows with unexpected date format, "
                         f"e.g. {bad.iloc[0]!r}")


@dataclass
class Tally:
    """In-window / out-of-window counts for one named event set.

    Counts are kept per scenario so that scenario lengths -- 1-12 days for
    scenario 1, 45-59 for scenario 2, exactly 2 for scenario 3, ~3 months for
    scenario 4 -- never get averaged together.
    """

    name: str
    inside: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    outside: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    users_inside: dict[int, set] = field(default_factory=lambda: defaultdict(set))

    def add(self, scenario: int, in_window: int, out_window: int) -> None:
        self.inside[scenario] += int(in_window)
        self.outside[scenario] += int(out_window)

    def total_in(self) -> int:
        return sum(self.inside.values())

    def total_out(self) -> int:
        return sum(self.outside.values())


class Concentrator:
    """Streams (user, day) event keys and tallies them against ground truth.

    Usage::

        c = Concentrator("5.2")
        c.feed("monster.com", users, days)     # pandas Series, aligned
        c.frame()                              # -> tidy DataFrame

    Only events by the labelled insider population count, per §5.
    """

    def __init__(self, release: str) -> None:
        self.gt = ground_truth(release)
        self.release = release
        self.tallies: dict[str, Tally] = {}
        self.per_user: dict[tuple[str, str], list[int]] = {}
        self._denoms: pd.DataFrame | None = None
        self._midpoint = {
            r.user: r.start_day + (r.end_day - r.start_day) / 2
            for r in self.gt.insiders.itertuples(index=False)
        }

    def insider_mask(self, users: pd.Series) -> pd.Series:
        return users.isin(self.gt.users)

    def feed(self, name: str, users: pd.Series, days: pd.Series) -> None:
        """Tally one batch of events.  ``days`` must be ``MM/DD/YYYY`` strings.

        Rows by users who are not labelled insiders are ignored -- they are
        outside the population the test is defined over.
        """
        if len(users) == 0:
            return
        keep = self.insider_mask(users)
        if not keep.any():
            self.tallies.setdefault(name, Tally(name))
            return
        u = users[keep]
        d = days[keep]
        key = u.str.cat(d, sep="|")
        inside = key.isin(self.gt.attack_keys)
        scen = u.map(self.gt.scenario_of)
        t = self.tallies.setdefault(name, Tally(name))
        for s, grp in inside.groupby(scen):
            n_in = int(grp.sum())
            t.add(int(s), n_in, int(len(grp)) - n_in)
            if n_in:
                t.users_inside[int(s)].update(u[grp.index[grp]].unique().tolist())

        for user, n in u[inside.to_numpy()].value_counts().items():
            self.per_user.setdefault((name, user), [0, 0, 0, 0])[0] += int(n)
        for user, n in u[~inside.to_numpy()].value_counts().items():
            self.per_user.setdefault((name, user), [0, 0, 0, 0])[1] += int(n)
        if inside.any():
            iu = u[inside.to_numpy()]
            idays = pd.to_datetime(d[inside.to_numpy()], format="%m/%d/%Y")
            first = idays.to_numpy() <= iu.map(self._midpoint).to_numpy()
            for user, n in iu[first].value_counts().items():
                self.per_user.setdefault((name, user), [0, 0, 0, 0])[2] += int(n)
            for user, n in iu[~first].value_counts().items():
                self.per_user.setdefault((name, user), [0, 0, 0, 0])[3] += int(n)

    def feed_keyed(self, names: pd.Series, users: pd.Series, days: pd.Series) -> None:
        """Tally MANY event sets at once, one per distinct value of ``names``.

        Used for domain discovery, where feeding several hundred domains
        through :meth:`feed` one at a time would mean several hundred passes
        over the same block.  Aggregate only -- no per-user detail.
        """
        if len(users) == 0:
            return
        keep = self.insider_mask(users)
        if not keep.any():
            return
        u, d, nm = users[keep], days[keep], names[keep]
        inside = u.str.cat(d, sep="|").isin(self.gt.attack_keys).to_numpy()
        scen = u.map(self.gt.scenario_of).to_numpy()
        g = pd.DataFrame({"name": nm.to_numpy(), "scen": scen,
                          "inside": inside, "user": u.to_numpy()})
        for (n, sc, ins), cnt in g.groupby(
                ["name", "scen", "inside"], sort=False).size().items():
            t = self.tallies.setdefault(n, Tally(n))
            if ins:
                t.add(int(sc), int(cnt), 0)
            else:
                t.add(int(sc), 0, int(cnt))
        for (n, sc, user), _ in g[g["inside"]].groupby(
                ["name", "scen", "user"], sort=False).size().items():
            self.tallies[n].users_inside[int(sc)].add(user)

    def denominators(self) -> pd.DataFrame:
        """User-days available inside and outside attack windows, per scenario.

        in_user_days  = sum over that scenario's insiders of their attack days
                        that fall within the user's observed logon span
        out_user_days = sum of (observed span days - in-span attack days)
        """
        if self._denoms is not None:
            return self._denoms
        spans = user_spans(self.release).set_index("user")
        rows = []
        for row in self.gt.insiders.itertuples(index=False):
            if row.user not in spans.index:
                rows.append((int(row.scenario), row.user, 0, 0, row.n_days, False))
                continue
            s = spans.loc[row.user]
            span_days = int(s["n_days"])
            lo = max(pd.Timestamp(s["first_day"]), row.start_day)
            hi = min(pd.Timestamp(s["last_day"]), row.end_day)
            in_days = max(0, (hi - lo).days + 1) if lo <= hi else 0
            rows.append((int(row.scenario), row.user, in_days,
                         span_days - in_days, row.n_days, True))
        df = pd.DataFrame(rows, columns=["scenario", "user", "in_user_days",
                                         "out_user_days", "labelled_days", "in_logon"])
        self._denoms = df
        return df

    def frame(self) -> pd.DataFrame:
        """Tidy per-(event set, scenario) results, plus an ALL row per set."""
        den = self.denominators()
        per_scen = den.groupby("scenario")[["in_user_days", "out_user_days"]].sum()
        n_ins = den.groupby("scenario")["user"].nunique()
        out = []
        for name, t in sorted(self.tallies.items()):
            for scenario in self.gt.scenarios():
                i, o = t.inside.get(scenario, 0), t.outside.get(scenario, 0)
                out.append(_row(name, scenario, i, o,
                                int(per_scen.loc[scenario, "in_user_days"]),
                                int(per_scen.loc[scenario, "out_user_days"]),
                                int(n_ins.loc[scenario]),
                                len(t.users_inside.get(scenario, ()))))
            out.append(_row(name, 0, t.total_in(), t.total_out(),
                            int(per_scen["in_user_days"].sum()),
                            int(per_scen["out_user_days"].sum()),
                            int(n_ins.sum()),
                            len(set().union(*t.users_inside.values()) if t.users_inside else ())))
        cols = ["event_set", "scenario", "in_window", "out_window", "ratio",
                "in_user_days", "out_user_days", "rate_in", "rate_out",
                "rate_ratio", "n_insiders", "n_insiders_with_hits"]
        return pd.DataFrame(out, columns=cols)

    def user_frame(self) -> pd.DataFrame:
        """Per-insider in/out counts, rates, and the window-half split.

        ``rate_ratio`` here is each insider measured against THEIR OWN
        out-of-window behaviour, which is the same question the pipeline's
        per-user z-score asks.  A population-level ratio near 1.0 can still
        hide a strong per-user signal, so both are reported.
        """
        den = self.denominators().set_index("user")
        rows = []
        for (name, user), (i, o, h1, h2) in sorted(self.per_user.items()):
            if user not in den.index:
                continue
            ind = int(den.loc[user, "in_user_days"])
            outd = int(den.loc[user, "out_user_days"])
            ri = i / ind if ind else float("nan")
            ro = o / outd if outd else float("nan")
            rr = (ri / ro) if ro else (float("inf") if ri else float("nan"))
            rows.append([name, int(den.loc[user, "scenario"]), user, i, o,
                         ind, outd, ri, ro, rr, h1, h2])
        return pd.DataFrame(rows, columns=[
            "event_set", "scenario", "user", "in_window", "out_window",
            "in_user_days", "out_user_days", "rate_in", "rate_out",
            "rate_ratio", "in_first_half", "in_second_half"])


def _row(name, scenario, i, o, ind, outd, n_ins, n_hit):
    ratio = (i / o) if o else float("inf") if i else float("nan")
    rate_in = i / ind if ind else float("nan")
    rate_out = o / outd if outd else float("nan")
    rr = (rate_in / rate_out) if rate_out else (float("inf") if rate_in else float("nan"))
    return [name, scenario, i, o, ratio, ind, outd, rate_in, rate_out, rr, n_ins, n_hit]


def write(df: pd.DataFrame, stem: str) -> Path:
    path = OUT / f"{stem}.csv"
    df.to_csv(path, index=False)
    progress(f"[write] {path}  ({len(df):,} rows)")
    return path


if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else "5.2"
    gt = ground_truth(rel)
    print(f"release {rel}: {len(gt.scenario_of)} insiders, "
          f"{len(gt.attack_keys):,} labelled attack user-days")
    ins = gt.insiders
    print(ins.groupby("scenario")["n_days"].describe()[["count", "min", "50%", "max"]])
    c = Concentrator(rel)
    d = c.denominators()
    print(d.groupby("scenario")[["in_user_days", "out_user_days"]].sum())
    print("insiders missing from logon.csv:", (~d["in_logon"]).sum())
