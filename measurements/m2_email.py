"""Brief M1 §4.2 + §4.3(activity) -- large_attachment_count, the dead detector.

Resumable.  Run it repeatedly until it prints COMPLETE:

    python3 m2_email.py 5.2      # 9.4 GB, several invocations
    python3 m2_email.py 4.2      # 1.36 GB
    python3 m2_email.py report   # print the results from the checkpoints

One pass over r5.2's email.csv answers:
  §4.2 per-email TOTAL attachment bytes and LARGEST SINGLE attachment
       distributions; counts over the 1,000,000-byte threshold under three
       candidate definitions; whether `size` includes attachment bytes;
       the §5 window-concentration test per scenario.
  §4.3 the full-file `activity` value counts -> the real Send/Receive ratio.

The r4.2 pass confirms its `attachments` column is a plain integer count.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import stream as S
import windows as W
from hist import Corr, Dist

THRESHOLD = 1_000_000
SIZE_RE = re.compile(r"\((\d+)\)")
EXPECTED_ROWS = {"5.2": 17_361_575, "4.2": None}


def parse_attachments(col: pd.Series):
    """(n_attachments, total_bytes, max_single_bytes) for one block.

    r5.2 format: ``name.ext(bytes);name.ext(bytes);...``.  Parsed from the
    parenthesised byte counts, so a filename containing digits or a semicolon
    cannot confuse it.  NaN / empty -> zeros.
    """
    n = len(col)
    counts = np.zeros(n, dtype=np.int32)
    totals = np.zeros(n, dtype=np.int64)
    maxima = np.zeros(n, dtype=np.int64)
    for i, raw in enumerate(col.to_numpy()):
        if not isinstance(raw, str) or not raw:
            continue
        found = SIZE_RE.findall(raw)
        if not found:
            continue
        nums = [int(x) for x in found]
        counts[i] = len(nums)
        totals[i] = sum(nums)
        maxima[i] = max(nums)
    return counts, totals, maxima


@dataclass
class EmailAcc:
    conc: W.Concentrator
    activity: Counter = field(default_factory=Counter)
    n_with_attach: int = 0
    over_total: int = 0
    over_single: int = 0
    over_msgsize: int = 0
    d_total: Dist = field(default_factory=lambda: Dist("per-email TOTAL attachment bytes", 1000))
    d_max: Dist = field(default_factory=lambda: Dist("LARGEST SINGLE attachment bytes", 1000))
    d_cnt: Dist = field(default_factory=lambda: Dist("attachments per email", 1))
    d_size_all: Dist = field(default_factory=lambda: Dist("message size, all emails", 1000))
    d_size_att: Dist = field(default_factory=lambda: Dist("message size, emails with attachments", 1000))
    d_resid: Dist = field(default_factory=lambda: Dist("size minus total attachment bytes", 1000))
    corr: Corr = field(default_factory=Corr)
    bad_dates: int = 0


def block_r52(acc: EmailAcc, df: pd.DataFrame) -> None:
    W.check_date_format(df["date"], "email.csv")
    day = df["date"].str[: W.DAY_LEN]
    acc.activity.update(df["activity"].fillna("(missing)"))

    counts, totals, maxima = parse_attachments(df["attachments"])
    msgsize = pd.to_numeric(df["size"], errors="coerce").fillna(0).to_numpy(np.int64)

    acc.d_size_all.add(msgsize)
    has = counts > 0
    acc.n_with_attach += int(has.sum())
    if has.any():
        acc.d_total.add(totals[has])
        acc.d_max.add(maxima[has])
        acc.d_cnt.add(counts[has])
        acc.d_size_att.add(msgsize[has])
        acc.d_resid.add(msgsize[has] - totals[has])
        acc.corr.add(msgsize[has], totals[has])

    m_total = totals > THRESHOLD
    m_single = maxima > THRESHOLD
    m_msg = msgsize > THRESHOLD
    acc.over_total += int(m_total.sum())
    acc.over_single += int(m_single.sum())
    acc.over_msgsize += int(m_msg.sum())

    u, d = df["user"], day
    sends = df["activity"].eq("Send").to_numpy()
    c = acc.conc
    c.feed("ALL emails", u, d)
    c.feed("Send emails", u[sends], d[sends])
    c.feed("any attachment at all", u[has], d[has])
    c.feed("D1 total attach > 1MB", u[m_total], d[m_total])
    c.feed("D2 largest single > 1MB", u[m_single], d[m_single])
    c.feed("D3 message size > 1MB (current code)", u[m_msg], d[m_msg])
    c.feed("D1 and Send", u[m_total & sends], d[m_total & sends])
    c.feed("D2 and Send", u[m_single & sends], d[m_single & sends])


def job_r52(release: str = "5.2") -> S.Job:
    return S.Job(
        tag=f"m2_email_{release.replace('.', '_')}",
        path=W.log_path(release, "email"),
        usecols=["date", "user", "activity", "size", "attachments"],
        acc=EmailAcc(conc=W.Concentrator(release)),
        on_block=block_r52,
    ).load()


@dataclass
class R42Acc:
    values: Counter = field(default_factory=Counter)
    non_integer: int = 0
    with_paren: int = 0
    max_size: int = 0
    over_msgsize: int = 0
    d_size: Dist = field(default_factory=lambda: Dist("r4.2 message size", 1000))
    examples: list = field(default_factory=list)


def block_r42(acc: R42Acc, df: pd.DataFrame) -> None:
    a = df["attachments"].fillna("")
    acc.values.update(a.value_counts().to_dict())
    bad = ~a.str.fullmatch(r"\d*")
    acc.non_integer += int(bad.sum())
    if bad.any() and len(acc.examples) < 5:
        acc.examples.extend(a[bad].head(5).tolist())
    acc.with_paren += int(a.str.contains("(", regex=False).sum())
    s = pd.to_numeric(df["size"], errors="coerce").fillna(0).to_numpy(np.int64)
    acc.d_size.add(s)
    acc.max_size = max(acc.max_size, int(s.max()) if len(s) else 0)
    acc.over_msgsize += int((s > THRESHOLD).sum())


def job_r42() -> S.Job:
    return S.Job(
        tag="m2_email_4_2",
        path=W.log_path("4.2", "email"),
        usecols=["size", "attachments"],
        acc=R42Acc(),
        on_block=block_r42,
    ).load()


def report() -> None:
    j5, j4 = job_r52(), job_r42()
    print("\n" + "=" * 78)
    print("§4.2 / §4.3  EMAIL")
    print("=" * 78)

    if j5.rows:
        a: EmailAcc = j5.acc
        rows = j5.rows
        st = "COMPLETE" if j5.done else f"PARTIAL ({100*j5.offset/j5.size:,.1f}% of file)"
        print(f"\nr5.2 email.csv -- {st}")
        print(S.assert_rows("m2_email_5_2", rows, EXPECTED_ROWS["5.2"] if j5.done else None))

        act = pd.DataFrame({"activity": list(a.activity),
                            "rows": list(a.activity.values())}
                           ).sort_values("rows", ascending=False)
        act["pct"] = 100 * act["rows"] / rows
        W.write(act, "m2_r52_email_activity")
        print("\n[§4.3] `activity` value counts over the file:")
        print(act.to_string(index=False))
        ns, nr = int(a.activity.get("Send", 0)), int(a.activity.get("Receive", 0))
        other = rows - ns - nr
        if ns:
            print(f"[§4.3] Send {ns:,} / Receive {nr:,} -> Receive/Send = {nr/ns:,.4f}")
            print(f"[§4.3] rows other than Send|Receive: {other:,}")
            print(f"[§4.3] WITHOUT a Send filter email_sent_count would be "
                  f"{rows/ns:,.4f}x too large ({rows:,} rows / {ns:,} Sends)")

        dist = pd.DataFrame([a.d_total.row(), a.d_max.row(), a.d_cnt.row(),
                             a.d_size_att.row(), a.d_size_all.row(), a.d_resid.row()])
        W.write(dist, "m2_r52_attachment_distribution")
        print(f"\n[§4.2] emails with >=1 attachment: {a.n_with_attach:,} of {rows:,}")
        print("[§4.2] distributions (percentiles accurate to the bucket width):")
        print(dist.to_string(index=False))

        over = pd.DataFrame({
            "definition": ["D1 total attachment bytes > 1,000,000",
                           "D2 largest single attachment > 1,000,000",
                           "D3 message `size` > 1,000,000 (what the code does today)"],
            "emails_over": [a.over_total, a.over_single, a.over_msgsize],
            "of_all_rows": [rows] * 3,
            "pct_of_all_rows": [100 * a.over_total / rows, 100 * a.over_single / rows,
                                100 * a.over_msgsize / rows],
            "of_rows_with_attachments": [a.n_with_attach] * 3,
        })
        W.write(over, "m2_r52_over_threshold")
        print(over.to_string(index=False))

        print("\n[§4.2] does `size` include attachment bytes? "
              "(r5.2 readme line 71: 'not including attachments')")
        print(f"   rows compared (emails with attachments): {a.corr.n:,}")
        print(f"   rows where attachment total EXCEEDS size: {a.corr.y_gt_x:,}"
              f"  ({100*a.corr.y_gt_x/max(a.corr.n,1):,.4f}%)")
        print(f"   corr(size, total attachment bytes)      : {a.corr.r():,.6f}")
        print(f"   median size (attachment rows)           : {a.d_size_att.quantile(50):,.0f}")
        print(f"   median total attachment bytes           : {a.d_total.quantile(50):,.0f}")
        print(f"   median residual (size - attach total)   : {a.d_resid.quantile(50):,.0f}")
        pd.DataFrame({"metric": ["rows_compared", "rows_attach_total_gt_size",
                                 "corr_size_vs_attach_total", "median_size_attach_rows",
                                 "median_attach_total", "median_residual"],
                      "value": [a.corr.n, a.corr.y_gt_x, a.corr.r(),
                                a.d_size_att.quantile(50), a.d_total.quantile(50),
                                a.d_resid.quantile(50)]}
                     ).to_csv(W.OUT / "m2_r52_size_vs_attachments.csv", index=False)

        res = a.conc.frame()
        res.insert(0, "release", "5.2")
        W.write(res, "m2_r52_email_window_test")
        print("\n[§4.2/§5] window-concentration test, r5.2:")
        print(fmt(res).to_string(index=False))

        uf = a.conc.user_frame()
        W.write(uf, "m2_r52_email_window_test_per_user")
        for es in ("D1 total attach > 1MB", "D1 and Send"):
            sub = uf[(uf["event_set"] == es) & (uf["scenario"] == 4)]
            if not len(sub):
                continue
            better = int((sub["rate_ratio"] > 1).sum())
            esc = int((sub["in_second_half"] > sub["in_first_half"]).sum())
            print(f"\n[§4.2] scenario-4 insiders under {es!r}: "
                  f"{better} of {len(sub)} have a HIGHER in-window rate than "
                  f"their own out-of-window rate; median per-user rate ratio "
                  f"{sub['rate_ratio'].median():,.3f}")
            print(f"       escalation check: {esc} of {len(sub)} have more "
                  f"in-window events in the SECOND half of their own window "
                  f"than the first")
            print(sub.sort_values("rate_ratio", ascending=False).head(8).to_string(index=False))

    if j4.rows:
        b: R42Acc = j4.acc
        st = "COMPLETE" if j4.done else f"PARTIAL ({100*j4.offset/j4.size:,.1f}%)"
        print(f"\n\nr4.2 email.csv -- {st}: {j4.rows:,} rows")
        vc = pd.DataFrame({"attachments_value": list(b.values),
                           "rows": list(b.values.values())}
                          ).sort_values("rows", ascending=False)
        W.write(vc, "m2_r42_attachments_values")
        print(f"[r4.2] distinct `attachments` values: {len(b.values)} -> "
              f"{sorted(b.values, key=lambda v: int(v) if v.isdigit() else -1)[:12]}")
        print(f"[r4.2] values that are NOT a bare integer: {b.non_integer:,}")
        print(f"[r4.2] values containing '(': {b.with_paren:,}")
        if b.examples:
            print("   examples:", b.examples[:5])
        print(f"[r4.2] max message `size`: {b.max_size:,} bytes")
        print(f"[r4.2] emails with size > {THRESHOLD:,}: {b.over_msgsize:,} of {j4.rows:,}")
        pd.DataFrame([b.d_size.row()]).to_csv(
            W.OUT / "m2_r42_size_distribution.csv", index=False)
        pd.DataFrame({"metric": ["rows", "distinct_attachment_values",
                                 "non_integer_values", "values_with_paren",
                                 "max_message_size", "emails_over_1MB_by_size"],
                      "value": [j4.rows, len(b.values), b.non_integer, b.with_paren,
                                b.max_size, b.over_msgsize]}
                     ).to_csv(W.OUT / "m2_r42_email_summary.csv", index=False)


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("ratio", "rate_in", "rate_out", "rate_ratio"):
        d[c] = d[c].map(lambda v: f"{v:,.4f}" if pd.notna(v) else "n/a")
    return d


import __main__ as _main
for _cls in (EmailAcc, R42Acc):
    if not hasattr(_main, _cls.__name__):
        setattr(_main, _cls.__name__, _cls)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "5.2"
    if which == "report":
        report()
    elif which == "4.2":
        job_r42().run()
    else:
        job_r52("5.2s" if which == "sample" else "5.2").run()
