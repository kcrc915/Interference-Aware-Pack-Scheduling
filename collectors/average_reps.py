"""Average the 4 reps per (app, core_count) into one row per config.

Run from the repo root:
    python collectors/average_reps.py
Outputs:
    data/<app>/manifest_avg.csv   (24 rows per app)
    data/all_apps_avg.csv         (all apps combined, 12 x 24 rows)
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
KEYS = ["app", "core_count"]
# columns that are identifiers/metadata rather than measurements
DROP = ["rep", "timestamp", "idle_threshold_met", "multiplex_status"]
# columns to also report the spread of (rep-to-rep noise)
STD_COLS = ["wall_time_s", "ipc", "dram_read_gbps", "dram_write_gbps"]


def average_app(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)

    # sanity: every config should have exactly 4 reps
    counts = df.groupby(KEYS).size()
    if (counts != 4).any():
        print(f"[warn] {csv.parent.name}: configs without 4 reps:\n"
              f"{counts[counts != 4]}")

    # keep only rows that are valid measurements
    ok = df[(df["multiplex_status"] == "OK") & (df["idle_threshold_met"])]
    if len(ok) != len(df):
        print(f"[warn] {csv.parent.name}: dropped {len(df) - len(ok)} bad rows")

    num_cols = [c for c in ok.select_dtypes("number").columns
                if c not in KEYS + DROP]

    g = ok.groupby(KEYS)
    avg = g[num_cols].mean()  # all-NaN columns (e.g. rss_*) stay NaN
    std = g[STD_COLS].std().add_suffix("_std")
    n = g.size().rename("n_reps")

    return pd.concat([avg, std, n], axis=1).reset_index()


def main():
    frames = []
    for csv in sorted(DATA.glob("*/manifest.csv")):
        out = average_app(csv)
        out.to_csv(csv.parent / "manifest_avg.csv", index=False)
        frames.append(out)
        print(f"{csv.parent.name}: {len(out)} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(DATA / "all_apps_avg.csv", index=False)
    print(f"combined: {len(combined)} rows -> {DATA / 'all_apps_avg.csv'}")


if __name__ == "__main__":
    main()