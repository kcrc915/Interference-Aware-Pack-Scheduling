"""Full pipeline: raw manifests -> one averaged, fully-derived table.

Run from the repo root:
    python collectors/build_dataset.py

Outputs:
    data/<app>/manifest_avg.csv   per-app, one row per core count
    data/all_apps_avg.csv         all apps combined, with derived columns
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
KEYS = ["app", "core_count"]
DROP = ["rep", "timestamp", "idle_threshold_met", "multiplex_status"]
STD_COLS = ["wall_time_s", "ipc", "dram_read_gbps", "dram_write_gbps"]
EXPECTED_REPS = 4

REQUIRED = [
    "app", "core_count", "n_reps",
    "wall_time_s", "wall_time_s_std",
    "instructions", "cycles", "ipc",
    "mpki", "llc_loads", "llc_load_misses", "llc_miss_rate",
    "dram_read_gbps", "dram_write_gbps", "dram_total_gbps",
    "speedup", "efficiency", "efficiency_pct", "step_speedup",
]


def average_app(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    app = csv.parent.name

    counts = df.groupby(KEYS).size()
    if (counts != EXPECTED_REPS).any():
        print(f"[warn] {app}: configs without {EXPECTED_REPS} reps:\n"
              f"{counts[counts != EXPECTED_REPS]}")

    ok = df[(df["multiplex_status"] == "OK") & (df["idle_threshold_met"])]
    if len(ok) != len(df):
        print(f"[warn] {app}: dropped {len(df) - len(ok)} bad rows")

    num_cols = [c for c in ok.select_dtypes("number").columns
                if c not in KEYS + DROP]
    g = ok.groupby(KEYS)
    out = pd.concat(
        [g[num_cols].mean(),
         g[STD_COLS].std().add_suffix("_std"),
         g.size().rename("n_reps")],
        axis=1,
    ).reset_index()
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    # recompute from raw counters so the metrics are guaranteed consistent
    # with the averaged counters (mean of ratios != ratio of means)
    df["ipc"] = df.instructions / df.cycles
    df["mpki"] = df.llc_load_misses / df.instructions * 1000
    df["llc_miss_rate"] = df.llc_load_misses / df.llc_loads

    df["dram_total_gbps"] = df.dram_read_gbps + df.dram_write_gbps
    df["dram_write_frac"] = df.dram_write_gbps / df.dram_total_gbps
    df["dram_gbps_per_core"] = df.dram_total_gbps / df.core_count

    base = df[df.core_count == 1].set_index("app").wall_time_s
    df["speedup"] = df.app.map(base) / df.wall_time_s
    df["efficiency"] = df.speedup / df.core_count
    df["efficiency_pct"] = df.efficiency * 100

    df = df.sort_values(KEYS).reset_index(drop=True)
    df["step_speedup"] = df.groupby("app").wall_time_s.shift(1) / df.wall_time_s
    return df


def validate(df: pd.DataFrame) -> bool:
    ok = True
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"[FAIL] missing columns: {missing}")
        return False

    # NaNs are expected only in step_speedup at 1 core
    nan_cols = df[REQUIRED].drop(columns="step_speedup").isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols):
        print(f"[FAIL] NaNs in required columns:\n{nan_cols}")
        ok = False

    if not np.isfinite(df[["speedup", "efficiency"]].to_numpy()).all():
        print("[FAIL] non-finite speedup/efficiency")
        ok = False

    # every app must have a 1-core row and 24 core counts
    per_app = df.groupby("app").core_count.agg(["min", "max", "nunique"])
    bad = per_app[(per_app["min"] != 1) | (per_app["nunique"] != 24)]
    if len(bad):
        print(f"[FAIL] apps with incomplete core sweeps:\n{bad}")
        ok = False

    # speedup at 1 core must be exactly 1
    if not np.allclose(df[df.core_count == 1].speedup, 1.0):
        print("[FAIL] speedup at 1 core != 1")
        ok = False

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        print(f"[note] entirely empty columns (not collected): {empty}")
    return ok


def main():
    frames = []
    for csv in sorted(DATA.glob("*/manifest.csv")):
        avg = average_app(csv)
        avg.to_csv(csv.parent / "manifest_avg.csv", index=False)
        frames.append(avg)
        print(f"{csv.parent.name}: {len(avg)} rows")

    df = add_derived(pd.concat(frames, ignore_index=True))
    df.to_csv(DATA / "all_apps_avg.csv", index=False)
    print(f"\ncombined: {df.shape[0]} rows x {df.shape[1]} cols "
          f"-> {DATA / 'all_apps_avg.csv'}")

    if validate(df):
        print("[PASS] all required columns present and valid")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()