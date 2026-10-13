#!/usr/bin/env python3
"""
Scaling plots for the banco whole-run solo-profiling dataset (data/<app>/manifest.csv).

For each app, produces a 2x2 figure:
  - wall-clock time vs core count (mean +/- std across reps)
  - speedup vs core count, with an ideal-linear-speedup reference line
  - IPC vs core count
  - DRAM bandwidth (read+write) vs core count

Also produces one cross-app comparison figure (speedup curves overlaid),
useful for spotting which apps scale well vs which saturate early.

Usage:
    python plotting/plot_scaling.py              # all apps found under data/
    python plotting/plot_scaling.py XSBench HPCG # just these apps

Output: figures/<app>_scaling.png, figures/all_apps_speedup.png
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
FIG_DIR = REPO / "figures"


def load_manifest(app: str):
    path = DATA_DIR / app / "manifest.csv"
    rows = list(csv.DictReader(open(path)))
    by_core = {}
    for r in rows:
        c = int(r["core_count"])
        by_core.setdefault(c, []).append(r)
    return by_core


def mean_std(values):
    values = list(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def app_stats(by_core):
    cores = sorted(by_core.keys())
    wall_mean, wall_std = [], []
    ipc_mean, ipc_std = [], []
    dram_mean, dram_std = [], []
    for c in cores:
        rows = by_core[c]
        wt = [float(r["wall_time_s"]) for r in rows]
        ipc = [float(r["ipc"]) for r in rows]
        dram = [float(r["dram_read_gbps"]) + float(r["dram_write_gbps"]) for r in rows]
        m, s = mean_std(wt); wall_mean.append(m); wall_std.append(s)
        m, s = mean_std(ipc); ipc_mean.append(m); ipc_std.append(s)
        m, s = mean_std(dram); dram_mean.append(m); dram_std.append(s)
    speedup = [wall_mean[0] / w for w in wall_mean]
    return {
        "cores": cores,
        "wall_mean": wall_mean, "wall_std": wall_std,
        "ipc_mean": ipc_mean, "ipc_std": ipc_std,
        "dram_mean": dram_mean, "dram_std": dram_std,
        "speedup": speedup,
    }


def plot_app(app: str, stats: dict):
    cores = stats["cores"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{app} -- scaling (banco, 1-{max(cores)} cores, 4 reps/point)", fontsize=13)

    ax = axes[0, 0]
    ax.errorbar(cores, stats["wall_mean"], yerr=stats["wall_std"], marker="o", capsize=3)
    ax.set_xlabel("core count"); ax.set_ylabel("wall time (s)")
    ax.set_title("Wall-clock time"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ideal = [c / cores[0] for c in cores]
    ax.plot(cores, stats["speedup"], marker="o", label="measured")
    ax.plot(cores, ideal, linestyle="--", color="gray", label="ideal linear")
    ax.set_xlabel("core count"); ax.set_ylabel("speedup vs 1 core")
    ax.set_title("Speedup"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(cores, stats["ipc_mean"], yerr=stats["ipc_std"], marker="o", color="tab:green", capsize=3)
    ax.set_xlabel("core count"); ax.set_ylabel("IPC")
    ax.set_title("Instructions per cycle"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(cores, stats["dram_mean"], yerr=stats["dram_std"], marker="o", color="tab:red", capsize=3)
    ax.set_xlabel("core count"); ax.set_ylabel("DRAM bandwidth (GB/s, read+write)")
    ax.set_title("DRAM bandwidth"); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{app}_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_all_speedup(all_stats: dict):
    fig, ax = plt.subplots(figsize=(9, 7))
    for app, stats in all_stats.items():
        ax.plot(stats["cores"], stats["speedup"], marker="o", markersize=3, label=app)
    max_cores = max(max(s["cores"]) for s in all_stats.values())
    ax.plot([1, max_cores], [1, max_cores], linestyle="--", color="black", linewidth=1, label="ideal linear")
    ax.set_xlabel("core count"); ax.set_ylabel("speedup vs 1 core")
    ax.set_title("Speedup comparison across apps (banco)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "all_apps_speedup.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("apps", nargs="*", help="subset of app names, default all under data/")
    args = ap.parse_args()

    apps = args.apps or sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())

    all_stats = {}
    for app in apps:
        by_core = load_manifest(app)
        stats = app_stats(by_core)
        all_stats[app] = stats
        plot_app(app, stats)

    if len(all_stats) > 1:
        plot_all_speedup(all_stats)


if __name__ == "__main__":
    main()
