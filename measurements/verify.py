"""Independent re-derivation of the headline numbers, by different code.

Every number in the report has a script behind it.  This file exists so the
numbers also have a SECOND, unrelated implementation behind them -- awk and
plain-Python line scanning rather than pandas -- because a bug in the shared
helper would otherwise be invisible.

    python3 verify.py
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import windows as W

SIZE_RE = re.compile(r"\((\d+)\)")


def head_rows(path: Path, n: int):
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= n:
                break
            yield row


def check(label: str, got, expected) -> None:
    ok = "OK  " if got == expected else "FAIL"
    print(f"[{ok}] {label}: got {got!r}, expected {expected!r}")


def main(stages=("1","2","3","4","5","6")) -> None:
    print("=" * 78)
    print("INDEPENDENT VERIFICATION -- stdlib csv / wc, not pandas")
    print("=" * 78)

    for rel, log, expected in (("5.2", "file", 887_621),
                               ("5.2", "email", 17_361_575),
                               ("4.2", "file", 445_581),
                               ("4.2", "email", 2_629_979)):
        p = W.log_path(rel, log)
        n = int(subprocess.run(["wc", "-l", str(p)], capture_output=True,
                               text=True).stdout.split()[0]) - 1
        check(f"r{rel} {log}.csv data rows (wc -l minus header)", n, expected)

    c = Counter()
    for row in head_rows(W.log_path("5.2", "file"), 200_000):
        c[(row["activity"], row["to_removable_media"],
           row["from_removable_media"])] += 1
    print("\nr5.2 file.csv cross-tab over the first 200,000 rows (stdlib csv):")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"   {k} -> {v:,}")
    check("to_removable_media==True implies activity=='File Write'",
          {k[0] for k in c if k[1] == "True"}, {"File Write"})

    n_att = over = viol = 0
    resid_max = 0
    for row in head_rows(W.log_path("5.2", "email"), 200_000):
        a = row.get("attachments") or ""
        nums = [int(x) for x in SIZE_RE.findall(a)]
        if not nums:
            continue
        n_att += 1
        tot = sum(nums)
        size = int(row["size"])
        if tot > size:
            viol += 1
        resid_max = max(resid_max, size - tot)
        if tot > 1_000_000:
            over += 1
    print(f"\nr5.2 email, first 200,000 rows (stdlib csv): "
          f"{n_att:,} with attachments, {over:,} over 1 MB total")
    check("rows where attachment total exceeds `size`", viol, 0)
    print(f"   max (size - attachment total) = {resid_max:,} bytes "
          f"-- compare r4.2's max message size, 141,909")

    act = Counter(r["activity"] for r in head_rows(W.log_path("5.2", "email"), 200_000))
    print(f"\nr5.2 email activity over first 200,000 rows: {dict(act)}")
    print("   (the brief quotes 61,693 Send / 138,307 Receive for this prefix)")

    rows = list(csv.DictReader(
        (W.ANSWERS / "insiders.csv").open(newline="", encoding="utf-8")))
    r52 = [r for r in rows if r["dataset"] == "5.2"]
    check("r5.2 insider rows", len(r52), 99)
    check("r5.2 scenario split",
          dict(sorted(Counter(r["scenario"] for r in r52).items())),
          {"1": 29, "2": 30, "3": 10, "4": 30})
    check("r5.2 insiders are 99 distinct users",
          len({r["user"] for r in r52}), 99)

    import m4_http as M
    hosts_hit = Counter()
    literal = Counter()
    n = 0
    with W.log_path("5.2", "http").open("r", newline="", encoding="utf-8",
                                        errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            if n > 400_000:
                break
            u = row["url"]
            low = u.lower()
            m = re.match(r"^https?://([^/]+)", u)
            host = m.group(1) if m else "(unparsed)"
            for entry in M.CONFIGURED:
                needle = entry.split(":", 1)[1]
                if needle in low:
                    literal[entry] += 1
                if "/" in needle:
                    if host == needle.split("/", 1)[0] and needle in low:
                        hosts_hit[entry] += 1
                elif needle in host:
                    hosts_hit[entry] += 1
    print(f"\nmatcher equivalence over the first {n-1:,} http rows:")
    check("host-vocabulary match == literal lowercased substring match",
          dict(hosts_hit), dict(literal))
    print(f"   entries hit in the prefix: {dict(literal)}")


if __name__ == "__main__":
    main()
