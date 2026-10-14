"""Brief M1 §4.1 + §4.3(filenames) -- what does a file event MEAN on r5.2?

Full pass over both releases' ``file.csv`` (r5.2 408 MB, r4.2 193 MB).
Produces:

  * the activity x to_removable_media x from_removable_media cross-tab
  * drive-letter distribution in ``filename``, broken down by activity
  * per-user / per-activity event rates on BOTH releases, which is what
    settles the "444 vs 874 events per employee" oddity
  * the §5 window-concentration test under the two candidate definitions of
    ``file_count``: all file events vs removable-media writes only
  * whether scenario 4 touches file.csv at all
  * §4.3: are r4.2 filenames bare (basename normalisation a no-op?), and do
    r5.2 basenames collide once the path is stripped?

Re-runnable, prints progress, writes CSVs into measurements/out/.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict

import pandas as pd

import windows as W

CHUNK = 200_000


def drive_of(filenames: pd.Series) -> pd.Series:
    """'C:' / 'R:' / '(none)' -- the leading drive letter, if there is one."""
    head = filenames.str[:2].str.upper()
    is_drive = filenames.str[1:2].eq(":")
    return head.where(is_drive, "(none)")


def run_r52(release: str = "5.2") -> None:
    path = W.log_path(release, "file")
    W.progress(f"[r5.2] full pass over {path} ({path.stat().st_size/1e6:,.0f} MB)")
    t0 = time.time()

    cross = Counter()
    drives = Counter()
    per_user_activity = Counter()
    full_paths: set[str] = set()
    basenames: set[str] = set()
    basename_owners = defaultdict(set)
    rows = 0

    conc = W.Concentrator(release)

    for chunk in pd.read_csv(
        path,
        usecols=["date", "user", "filename", "activity",
                 "to_removable_media", "from_removable_media"],
        dtype=str,
        chunksize=CHUNK,
    ):
        rows += len(chunk)
        W.check_date_format(chunk["date"], path.name)
        day = chunk["date"].str[: W.DAY_LEN]

        act = chunk["activity"].fillna("(missing)")
        to_rm = chunk["to_removable_media"].fillna("(missing)")
        fr_rm = chunk["from_removable_media"].fillna("(missing)")
        cross.update(zip(act, to_rm, fr_rm))

        fn = chunk["filename"].fillna("")
        drives.update(zip(act, drive_of(fn)))
        per_user_activity.update(zip(chunk["user"], act))

        full_paths.update(fn.unique().tolist())
        base = fn.str.rsplit("\\", n=1).str[-1].str.rsplit("/", n=1).str[-1]
        for b, f in set(zip(base, fn)):
            basename_owners[b].add(f)
        basenames.update(base.unique().tolist())

        is_write = to_rm.eq("True")
        conc.feed("ALL file events", chunk["user"], day)
        conc.feed("removable-media WRITES only",
                  chunk["user"][is_write], day[is_write])
        is_read = fr_rm.eq("True")
        conc.feed("removable-media READS only",
                  chunk["user"][is_read], day[is_read])
        for a in ("File Open", "File Copy", "File Write", "File Delete"):
            m = act.eq(a)
            if m.any():
                conc.feed(f"activity={a}", chunk["user"][m], day[m])

        if rows % 1_000_000 < CHUNK:
            W.progress(f"[r5.2]   {rows:,} rows  ({time.time()-t0:,.1f}s)")

    W.progress(f"[r5.2] {rows:,} rows in {time.time()-t0:,.1f}s")

    ct = pd.DataFrame(
        [(a, t, f, n) for (a, t, f), n in cross.items()],
        columns=["activity", "to_removable_media", "from_removable_media", "rows"],
    ).sort_values("rows", ascending=False)
    ct["pct_of_rows"] = 100 * ct["rows"] / rows
    W.write(ct, "m1_r52_file_crosstab")
    print(ct.to_string(index=False))

    dv = pd.DataFrame([(a, d, n) for (a, d), n in drives.items()],
                      columns=["activity", "drive", "rows"]).sort_values(
                          ["activity", "rows"], ascending=[True, False])
    W.write(dv, "m1_r52_file_drives")
    print(dv.to_string(index=False))

    pua = pd.DataFrame([(u, a, n) for (u, a), n in per_user_activity.items()],
                       columns=["user", "activity", "rows"])
    W.write(pua, "m1_r52_file_per_user_activity")

    n_users = pua["user"].nunique()
    summary = pua.groupby("activity")["rows"].sum().to_frame("rows")
    summary["per_user_in_log"] = summary["rows"] / n_users
    summary["per_employee_2000"] = summary["rows"] / 2000
    print("\n[r5.2] per-activity totals   (users appearing in file.csv: "
          f"{n_users:,})")
    print(summary.to_string())
    W.write(summary.reset_index(), "m1_r52_file_activity_totals")

    coll = {b: v for b, v in basename_owners.items() if len(v) > 1}
    print(f"\n[r5.2 §4.3] distinct full paths : {len(full_paths):,}")
    print(f"[r5.2 §4.3] distinct basenames  : {len(basenames):,}")
    print(f"[r5.2 §4.3] basenames mapping to >1 full path: {len(coll):,}")
    if coll:
        worst = sorted(coll.items(), key=lambda kv: -len(kv[1]))[:5]
        for b, v in worst:
            print(f"   {b!r} <- {len(v)} distinct paths, e.g. {sorted(v)[:3]}")
    pd.DataFrame({"metric": ["distinct_full_paths", "distinct_basenames",
                             "basenames_with_multiple_paths"],
                  "value": [len(full_paths), len(basenames), len(coll)]}
                 ).to_csv(W.OUT / "m1_r52_filename_normalisation.csv", index=False)

    res = conc.frame()
    res.insert(0, "release", release)
    W.write(res, "m1_r52_file_window_test")
    show(res)
    uf = conc.user_frame()
    W.write(uf, "m1_r52_file_window_test_per_user")
    s4u = uf[(uf["scenario"] == 4) & (uf["event_set"].isin(
        ["ALL file events", "activity=File Open", "removable-media WRITES only"]))]
    print("\n[r5.2] scenario-4 insiders, per user "
          "(in_first_half/in_second_half tests the 'escalating' description):")
    print(s4u.to_string(index=False))

    gt = W.ground_truth(release)
    s4 = [u for u, s in gt.scenario_of.items() if s == 4]
    sub = pua[pua["user"].isin(s4)]
    print(f"\n[r5.2] scenario-4 insiders present in file.csv: "
          f"{sub['user'].nunique()} of {len(s4)}")
    print(sub.groupby("activity")["rows"].agg(["sum", "count"]).to_string())


def run_r42(release: str = "4.2") -> None:
    path = W.log_path(release, "file")
    W.progress(f"\n[r4.2] full pass over {path} ({path.stat().st_size/1e6:,.0f} MB)")
    t0 = time.time()
    per_user = Counter()
    sep_hits = 0
    examples: list[str] = []
    full_paths: set[str] = set()
    basenames: set[str] = set()
    rows = 0
    conc = W.Concentrator(release)

    for chunk in pd.read_csv(path, usecols=["date", "user", "filename"],
                             dtype=str, chunksize=CHUNK):
        rows += len(chunk)
        W.check_date_format(chunk["date"], path.name)
        day = chunk["date"].str[: W.DAY_LEN]
        fn = chunk["filename"].fillna("")
        per_user.update(chunk["user"])
        bad = fn.str.contains(r"[\\/]", regex=True)
        sep_hits += int(bad.sum())
        if bad.any() and len(examples) < 5:
            examples.extend(fn[bad].head(5).tolist())
        full_paths.update(fn.unique().tolist())
        base = fn.str.rsplit("\\", n=1).str[-1].str.rsplit("/", n=1).str[-1]
        basenames.update(base.unique().tolist())
        conc.feed("ALL file events", chunk["user"], day)
        if rows % 2_000_000 < CHUNK:
            W.progress(f"[r4.2]   {rows:,} rows  ({time.time()-t0:,.1f}s)")

    W.progress(f"[r4.2] {rows:,} rows in {time.time()-t0:,.1f}s")
    n_users = len(per_user)
    print(f"[r4.2] users in file.csv: {n_users:,}; rows/user in log: "
          f"{rows/n_users:,.1f}; rows/employee(1000): {rows/1000:,.1f}")
    print(f"[r4.2 §4.3] filenames containing a path separator: {sep_hits:,} "
          f"of {rows:,}  -> basename normalisation is "
          f"{'A NO-OP' if sep_hits == 0 else 'NOT a no-op'}")
    if examples:
        print("   examples:", examples[:5])
    print(f"[r4.2 §4.3] distinct full names {len(full_paths):,} / "
          f"distinct basenames {len(basenames):,}")

    pd.DataFrame({"user": list(per_user), "rows": list(per_user.values())}
                 ).to_csv(W.OUT / "m1_r42_file_per_user.csv", index=False)
    pd.DataFrame({"metric": ["rows", "users_in_log", "rows_per_user_in_log",
                             "filenames_with_separator", "distinct_full_names",
                             "distinct_basenames"],
                  "value": [rows, n_users, rows / n_users, sep_hits,
                            len(full_paths), len(basenames)]}
                 ).to_csv(W.OUT / "m1_r42_file_summary.csv", index=False)

    res = conc.frame()
    res.insert(0, "release", release)
    W.write(res, "m1_r42_file_window_test")
    show(res)
    W.write(conc.user_frame(), "m1_r42_file_window_test_per_user")


def show(df: pd.DataFrame) -> None:
    d = df.copy()
    for c in ("ratio", "rate_in", "rate_out", "rate_ratio"):
        d[c] = d[c].map(lambda v: f"{v:,.4f}" if pd.notna(v) else "n/a")
    print(d.to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "5.2"):
        run_r52("5.2s" if which == "sample" else "5.2")
    if which in ("both", "4.2"):
        run_r42()
