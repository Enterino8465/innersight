"""Turns the raw measurement CSVs into the exact figures quoted in the report.

Nothing here reads the dataset -- it only reduces ``measurements/out/*.csv``,
so every number in the write-up can be re-derived by running this after the
four measurement scripts.  Kept separate from them so a reporting change never
means re-reading 40 GB.
"""

from __future__ import annotations

import pandas as pd

import windows as W

pd.set_option("display.width", 200)
OUT = W.OUT


def per_user_summary(path: str, label: str, sets: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(OUT / path)
    if sets:
        df = df[df["event_set"].isin(sets)]
    rows = []
    for (es, sc), g in df.groupby(["event_set", "scenario"]):
        hit = g[g["in_window"] > 0]
        clean = int((g["out_window"] == 0).sum())
        clean_hit = int(((g["out_window"] == 0) & (g["in_window"] > 0)).sum())
        finite = g[(g["out_window"] > 0) & (g["in_window"] > 0)]
        esc = g[g["in_window"] > 0]
        rows.append({
            "source": label, "event_set": es, "scenario": sc,
            "insiders": len(g),
            "with_in_window_hits": len(hit),
            "with_ZERO_out_of_window": clean,
            "clean_detections (in>0 and out==0)": clean_hit,
            "median_per_user_rate_ratio (finite)":
                round(float(finite["rate_ratio"].median()), 3) if len(finite) else None,
            "n_finite": len(finite),
            "escalating (2nd half > 1st half)":
                int((esc["in_second_half"] > esc["in_first_half"]).sum()),
            "of_hitters": len(esc),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 100)
    print("A.  file.csv -- per-insider view of the two candidate file_count definitions")
    print("=" * 100)
    a = per_user_summary(
        "m1_r52_file_window_test_per_user.csv", "r5.2 file",
        ["ALL file events", "activity=File Open", "activity=File Write",
         "removable-media WRITES only", "removable-media READS only"])
    print(a.to_string(index=False))
    print("\nr4.2 (no activity column -- every row counts under either definition):")
    print(per_user_summary("m1_r42_file_window_test_per_user.csv", "r4.2 file"
                           ).to_string(index=False))

    print("\n" + "=" * 100)
    print("B.  email -- per-insider view of the three candidate large_attachment definitions")
    print("=" * 100)
    b = per_user_summary("m2_r52_email_window_test_per_user.csv", "r5.2 email")
    print(b.to_string(index=False))

    print("\n" + "=" * 100)
    print("C.  domains -- top hosts by in-window concentration WITHIN each scenario")
    print("=" * 100)
    for rel in ("52", "42"):
        f = OUT / f"m4_r{rel}_domain_window_test_full.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[~df["event_set"].str.contains(":")]
        for sc in sorted(x for x in df["scenario"].unique() if x != 0):
            s = df[(df["scenario"] == sc) & (df["in_window"] >= 5)].copy()
            s = s.sort_values(["rate_ratio", "in_window"], ascending=False).head(12)
            print(f"\n--- r{rel[0]}.{rel[1]} scenario {sc}: top hosts "
                  f"(>=5 in-window hits by that scenario's insiders) ---")
            print(s[["event_set", "in_window", "out_window", "rate_ratio",
                     "n_insiders", "n_insiders_with_hits"]].to_string(
                         index=False, formatters={"rate_ratio": "{:,.2f}".format}))

    print("\n" + "=" * 100)
    print("D.  file.csv volume, r4.2 vs r5.2 -- the '444 vs 874 per person' claim")
    print("=" * 100)
    r42 = pd.read_csv(OUT / "m1_r42_file_summary.csv").set_index("metric")["value"]
    tot = pd.read_csv(OUT / "m1_r52_file_activity_totals.csv")
    r52_rows = int(tot["rows"].sum())
    r52_users = int(pd.read_csv(OUT / "m1_r52_file_per_user_activity.csv")["user"].nunique())
    write_rows = int(tot.loc[tot["activity"] == "File Write", "rows"].iloc[0])
    tbl = pd.DataFrame([
        {"release": "4.2", "employees": 1000, "file.csv rows": int(r42["rows"]),
         "users appearing": int(r42["users_in_log"]),
         "rows / employee": int(r42["rows"]) / 1000,
         "rows / appearing user": int(r42["rows"]) / int(r42["users_in_log"]),
         "removable-write rows": int(r42["rows"]),
         "removable-writes / employee": int(r42["rows"]) / 1000},
        {"release": "5.2", "employees": 2000, "file.csv rows": r52_rows,
         "users appearing": r52_users,
         "rows / employee": r52_rows / 2000,
         "rows / appearing user": r52_rows / r52_users,
         "removable-write rows": write_rows,
         "removable-writes / employee": write_rows / 2000},
    ])
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print("\n" + "=" * 100)
    print("E.  corpus volume of every configured domain entry, both releases")
    print("=" * 100)
    print(pd.read_csv(OUT / "m4_current_entries_side_by_side.csv").to_string(index=False))


if __name__ == "__main__":
    main()
