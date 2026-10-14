"""Brief M1 §4.5 -- the PARSE cost, on top of the already-measured I/O floor.

The 52-second raw text scan of all 39.7 GB is on record and is NOT redone
here.  That was ``line.split(",", 3)[2]``: no CSV parsing, no date parsing, no
grouping.  What predicts a real Stage 1 run is the cost of what the pipeline
actually does per row, so this times exactly that and nothing else:

    pandas.read_csv(chunked, usecols=<the pipeline's column projection>)
    pandas.to_datetime(date, format=config.DATE_FORMAT)

over a BOUNDED PREFIX of each log, stated in the output, then extrapolates on
rows.  Feature computation is deliberately excluded -- it is measured for real
on r4.2 (16 GB *including* all 18 features in ~90 s) and that calibration is
the comparison the brief asks for.

Usage:  python3 m5_parse_cost.py [prefix_mb]
"""

from __future__ import annotations

import io
import sys
import time

import pandas as pd

import stream as S
import windows as W

DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

PROJECTION = {
    "logon": ["date", "user", "pc", "activity"],
    "device": ["date", "user", "pc", "activity"],
    "file": ["date", "user", "pc", "filename", "activity",
             "to_removable_media", "from_removable_media"],
    "email": ["date", "user", "pc", "to", "cc", "bcc", "from",
              "activity", "size", "attachments"],
    "http": ["date", "user", "pc", "url"],
}

FULL_ROWS = {"logon": 1_810_070, "device": 836_984, "file": 887_621,
             "email": 17_361_575, "http": 58_960_449}


def time_prefix(release: str, log: str, prefix_mb: int) -> dict:
    path = W.log_path(release, log)
    size = path.stat().st_size
    cols = S.header_of(path)
    want = [c for c in PROJECTION[log] if c in cols]
    budget = min(prefix_mb * 1024 * 1024, size)

    with path.open("rb") as f:
        f.readline()
        buf = f.read(budget)
        buf += f.readline()

    t0 = time.time()
    df = pd.read_csv(io.BytesIO(buf), header=None, names=cols, usecols=want,
                     dtype=str, engine="c")
    t_read = time.time() - t0

    t0 = time.time()
    ts = pd.to_datetime(df["date"], format=DATE_FORMAT)
    t_date = time.time() - t0
    assert ts.notna().all(), f"{log}: unparsable dates under {DATE_FORMAT}"

    rows = len(df)
    mb = len(buf) / 1e6
    total = t_read + t_date
    full = FULL_ROWS[log]
    W.progress(f"[{log:6s}] prefix {mb:7,.0f} MB  {rows:>10,} rows  "
               f"read {t_read:5.2f}s  dates {t_date:5.2f}s  "
               f"-> {mb/total:,.0f} MB/s, {rows/total:,.0f} rows/s  "
               f"| full file {full:,} rows -> {full/(rows/total):,.1f}s")
    return {
        "log": log, "prefix_MB": round(mb, 1), "prefix_rows": rows,
        "read_s": round(t_read, 3), "date_parse_s": round(t_date, 3),
        "total_s": round(total, 3),
        "MB_per_s": round(mb / total, 1), "rows_per_s": round(rows / total, 0),
        "file_MB": round(size / 1e6, 1), "file_rows": full,
        "projected_full_s": round(full / (rows / total), 1),
        "columns_projected": len(want),
    }


if __name__ == "__main__":
    prefix_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    W.progress(f"§4.5 parse cost -- bounded prefix of {prefix_mb} MB per log "
               f"(whole file where smaller), r5.2\n")
    rows = [time_prefix("5.2", log, prefix_mb) for log in PROJECTION]
    df = pd.DataFrame(rows)
    W.write(df, "m5_parse_cost")
    print()
    print(df.to_string(index=False))
    tot = df["projected_full_s"].sum()
    print(f"\nprojected r5.2 parse+date total: {tot:,.0f}s "
          f"({tot/60:,.1f} min) for {df['file_MB'].sum()/1000:,.1f} GB")
    print(f"effective rate: {df['file_MB'].sum()/tot:,.0f} MB/s")
    print("r4.2 calibration: ~16 GB *with all 18 features* in ~90 s "
          "= ~180 MB/s effective")
