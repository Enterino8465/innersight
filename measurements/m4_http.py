"""Brief M1 §4.4 -- the two domain lists, re-measured on BOTH releases.

The expensive pass: 30.1 GB of r5.2 ``http.csv`` and 14.5 GB of r4.2's.
Resumable -- run repeatedly until it prints COMPLETE:

    python3 m4_http.py 5.2
    python3 m4_http.py 4.2
    python3 m4_http.py report

Three things, in one pass per release:

1. Score every CURRENT entry of JOB_SEARCH_DOMAINS and CLOUD_UPLOAD_DOMAINS
   with the §5 window-concentration test.  Matching replicates the pipeline's
   own matcher exactly -- substring against the lowercased URL -- which is why
   a path entry like ``aol.com/jobs`` works at all.
2. Discover candidates: score EVERY host in the release the same way and rank
   by in-window concentration, so the top unlisted ones can be reported.
3. Report both releases side by side, recomputed by this code rather than
   compared against remembered figures.

Corpus-wide hit counts and distinct-visitor counts are collected too, because
the r4.2 decisions turned on them (mediafire.com was 90% of feature 17).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

import stream as S
import windows as W

JOB_SEARCH_DOMAINS = ("careerbuilder.com", "monster.com", "jobhuntersbible.com",
                      "job-hunt.org", "indeed.com", "simplyhired.com",
                      "aol.com/jobs")
CLOUD_UPLOAD_DOMAINS = ("wikileaks.org", "dropbox.com", "4shared.com")
DROPPED_BY_R42 = ("craigslist.org", "linkedin.com", "mediafire.com")
CONFIGURED = tuple(f"job:{d}" for d in JOB_SEARCH_DOMAINS) + \
             tuple(f"cloud:{d}" for d in CLOUD_UPLOAD_DOMAINS) + \
             tuple(f"dropped:{d}" for d in DROPPED_BY_R42)

EXPECTED_ROWS = {"5.2": 58_960_449, "4.2": None}
PATH_WATCH = frozenset({"aol.com"})


@dataclass
class HttpAcc:
    conc: W.Concentrator
    dom_all: Counter = field(default_factory=Counter)
    dom_users: dict = field(default_factory=dict)
    path_all: Counter = field(default_factory=Counter)
    user_ids: dict = field(default_factory=dict)
    cfg_all: Counter = field(default_factory=Counter)
    scheme_other: int = 0
    uppercase_hosts: int = 0

    def uid(self, user: str) -> int:
        i = self.user_ids.get(user)
        if i is None:
            i = len(self.user_ids)
            self.user_ids[user] = i
        return i


def block(acc: HttpAcc, df: pd.DataFrame) -> None:
    W.check_date_format(df["date"], "http.csv")
    day = df["date"].str[: W.DAY_LEN]
    url = df["url"]
    host = url.str.extract(r"^https?://([^/]+)", expand=False).fillna("(unparsed)")
    acc.scheme_other += int((host == "(unparsed)").sum())
    acc.uppercase_hosts += int(host.str.contains(r"[A-Z]", regex=True).sum())

    acc.dom_all.update(host.value_counts().to_dict())
    pairs = pd.DataFrame({"h": host.to_numpy(), "u": df["user"].to_numpy()}
                         ).drop_duplicates()
    for h, u in zip(pairs["h"], pairs["u"]):
        acc.dom_users.setdefault(h, set()).add(acc.uid(u))

    for entry in CONFIGURED:
        needle = entry.split(":", 1)[1]
        if "/" in needle:
            h0 = needle.split("/", 1)[0]
            cand = host.eq(h0).to_numpy()
            if cand.any():
                m = cand.copy()
                m[cand] = url[cand].str.lower().str.contains(
                    needle, regex=False).to_numpy()
            else:
                m = cand
        else:
            hits = {h for h in acc.dom_all if needle in h}
            m = host.isin(hits).to_numpy() if hits else host.eq("\0").to_numpy()
        n = int(m.sum())
        if n:
            acc.cfg_all[entry] += n
            acc.conc.feed(entry, df["user"][m], day[m])
        else:
            acc.conc.feed(entry, df["user"][:0], day[:0])

    watch = host.isin(PATH_WATCH)
    if watch.any():
        seg = url[watch].str.lower().str.split("/", n=4).str[3].fillna("")
        acc.path_all.update((host[watch] + "/" + seg).value_counts().to_dict())

    acc.conc.feed_keyed(host, df["user"], day)


def job(release: str) -> S.Job:
    return S.Job(
        tag=f"m4_http_{release.replace('.', '_')}",
        path=W.log_path(release, "http"),
        usecols=["date", "user", "url"],
        acc=HttpAcc(conc=W.Concentrator(release)),
        on_block=block,
    ).load()


def summarise(j: S.Job, release: str) -> pd.DataFrame:
    acc: HttpAcc = j.acc
    res = acc.conc.frame()
    res = res[res["scenario"] == 0].copy()
    res["release"] = release
    res["corpus_hits"] = res["event_set"].map(
        lambda n: acc.cfg_all.get(n, acc.dom_all.get(n, 0)))
    res["corpus_visitors"] = res["event_set"].map(
        lambda n: len(acc.dom_users.get(n, ())))
    return res


def report() -> None:
    frames, per_scen, jobs = {}, {}, {}
    for rel in ("5.2", "4.2"):
        j = job(rel)
        if not j.rows:
            continue
        jobs[rel] = j
        frames[rel] = summarise(j, rel)
        f = j.acc.conc.frame()
        f.insert(0, "release", rel)
        per_scen[rel] = f
        W.write(f, f"m4_r{rel.replace('.','')}_domain_window_test_full")
        W.write(pd.DataFrame(
            {"host": list(j.acc.dom_all), "corpus_hits": list(j.acc.dom_all.values()),
             "corpus_visitors": [len(j.acc.dom_users.get(h, ())) for h in j.acc.dom_all]}
        ).sort_values("corpus_hits", ascending=False),
            f"m4_r{rel.replace('.','')}_domain_corpus_counts")

    print("\n" + "=" * 78)
    print("§4.4  DOMAIN LISTS")
    print("=" * 78)
    for rel, j in jobs.items():
        st = "COMPLETE" if j.done else f"PARTIAL {100*j.offset/j.size:,.1f}%"
        print(f"\nr{rel} http.csv -- {st}: {j.rows:,} rows, "
              f"{len(j.acc.dom_all):,} distinct hosts, "
              f"{j.acc.scheme_other:,} unparsed URLs")
        if j.done:
            print(S.assert_rows(f"m4_http_{rel}", j.rows, EXPECTED_ROWS[rel]))

    print("\n--- 1/3  CURRENT CONFIG ENTRIES, both releases, all scenarios ---")
    rows = []
    for entry in CONFIGURED:
        r = {"entry": entry}
        for rel in ("4.2", "5.2"):
            if rel not in frames:
                continue
            m = frames[rel][frames[rel]["event_set"] == entry]
            if len(m):
                x = m.iloc[0]
                r[f"{rel}_in"] = int(x["in_window"])
                r[f"{rel}_out"] = int(x["out_window"])
                r[f"{rel}_rate_ratio"] = round(float(x["rate_ratio"]), 2) \
                    if pd.notna(x["rate_ratio"]) else float("nan")
                r[f"{rel}_insiders_hit"] = int(x["n_insiders_with_hits"])
                r[f"{rel}_corpus_hits"] = int(x["corpus_hits"])
        rows.append(r)
    cur = pd.DataFrame(rows)
    W.write(cur, "m4_current_entries_side_by_side")
    print(cur.to_string(index=False))

    for rel in frames:
        acc: HttpAcc = jobs[rel].acc
        f = per_scen[rel]
        allrows = f[(f["scenario"] == 0) & (~f["event_set"].isin(CONFIGURED))].copy()
        allrows["corpus_hits"] = allrows["event_set"].map(acc.dom_all).fillna(0).astype(int)
        allrows["corpus_visitors"] = allrows["event_set"].map(
            lambda h: len(acc.dom_users.get(h, ()))).astype(int)
        allrows = allrows[allrows["in_window"] >= 5]
        allrows = allrows.sort_values("rate_ratio", ascending=False)
        W.write(allrows, f"m4_r{rel.replace('.','')}_discovery_ranked")
        print(f"\n--- 2/3  r{rel}: top hosts by in-window concentration "
              f"(>=5 in-window hits), NOT currently listed ---")
        cols = ["event_set", "in_window", "out_window", "rate_in", "rate_out",
                "rate_ratio", "n_insiders_with_hits", "corpus_hits", "corpus_visitors"]
        print(allrows[cols].head(20).to_string(index=False,
              formatters={c: "{:,.3f}".format for c in
                          ("rate_in", "rate_out", "rate_ratio")}))

    for rel in per_scen:
        f = per_scen[rel]
        sub = f[f["event_set"].isin(CONFIGURED) & (f["scenario"] != 0)]
        sub = sub[sub["in_window"] + sub["out_window"] > 0]
        print(f"\n--- 3/3  r{rel}: configured entries per scenario ---")
        d = sub.copy()
        for c in ("ratio", "rate_in", "rate_out", "rate_ratio"):
            d[c] = d[c].map(lambda v: f"{v:,.3f}" if pd.notna(v) else "n/a")
        print(d[["event_set", "scenario", "in_window", "out_window",
                 "rate_ratio", "n_insiders", "n_insiders_with_hits"]
                ].to_string(index=False))

    for rel in jobs:
        uf = jobs[rel].acc.conc.user_frame()
        W.write(uf[uf["event_set"].isin(CONFIGURED)],
                f"m4_r{rel.replace('.','')}_configured_per_user")
    for rel in jobs:
        acc = jobs[rel].acc
        if acc.path_all:
            top = pd.Series(acc.path_all).sort_values(ascending=False).head(10)
            print(f"\nr{rel}: first path segment under {sorted(PATH_WATCH)} "
                  f"(corpus-wide), top 10:\n{top.to_string()}")


import __main__ as _main
for _cls in (HttpAcc,):
    if not hasattr(_main, _cls.__name__):
        setattr(_main, _cls.__name__, _cls)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "5.2"
    if which == "report":
        report()
    else:
        job("5.2s" if which == "sample" else which).run()
