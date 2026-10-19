#!/usr/bin/env python3
"""Defensibility check for the archetype-clustering signature: computes the
correlation matrix across the 12 apps' candidate solo-scaling features, so
feature selection for k-means is backed by evidence of non-redundancy
rather than physical intuition alone.

This was NOT run before the original 7-feature signature was built -- those
7 features were picked by reasoning about what *should* be a distinct,
physically-meaningful facet of memory/bandwidth-boundedness, with no check
that they actually turned out to be statistically independent across the
12 apps we have. Running it after the fact found two near-duplicate pairs
(see REDUNDANT_PAIRS below) that the physical reasoning alone missed.

Two more candidates were proposed and tested after the original 7:
dram_bw_mean and dram_bw_max (mean/peak bandwidth across the whole sweep,
rather than just the value at core=24) turned out to be near-perfectly
redundant with dram_bw_24core AND WITH EACH OTHER (r=0.97-0.997 across all
four of dram_bw_24core/growth/mean/max) -- they all measure the same "how
bandwidth-hungry is this app" LEVEL, just sampled differently. With 4
mutually-redundant features instead of 2, a simple pairwise drop rule
breaks (dropping "the loser of each pair" can disagree with itself across
overlapping pairs), so redundant features are grouped into connected
components (see `redundant_groups`) and exactly one survivor is kept per
group. bw_scaling_shape (= dram_bw[24] / dram_bw_max -- near 1.0 if still
rising at 24 cores, well below 1.0 if the app already peaked earlier and
declined from self-contention among its own threads) measures the curve's
SHAPE instead of its level, and tested genuinely independent (|r| < 0.6
against everything, below the redundancy threshold) -- it survives into
the final signature alongside the one surviving level feature.

Unit of analysis: ONE ROW PER APP (12 rows). Each column is a per-app
summary of one candidate feature, computed from that app's quality-filtered
1-24 core solo sweep. Correlation is computed ACROSS THE 12 APPS -- i.e.,
"do apps that score high on feature X also tend to score high on feature
Y?" -- which is the right question for judging whether two features would
make k-means double-count the same signal, not whether they're correlated
within a single app's sweep.

Caveat that should travel with every number this script prints: n=12 is a
small sample for a correlation estimate (SE on the order of +-0.3), so
the exact magnitudes are directional, not precise. The qualitative pattern
(two near-perfect pairs vs. everything else well below 0.8) is consistent
enough to act on; the third decimal place of any single r is not.
"""
import csv
import statistics
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

SOLO_APPS = {
    "ExaMiniMD": "data/solo_wholerun_banco_full_examinimd/manifest.csv",
    "Quicksilver": "data/solo_wholerun_banco_full_quicksilver/manifest.csv",
    "XSBench": "data/solo_wholerun_banco_full_xsbench/manifest.csv",
    "CG": "data/solo_wholerun_banco_full_cg/manifest.csv",
    "LULESH": "data/solo_wholerun_banco_full_lulesh/manifest.csv",
    "HPCG": "data/solo_wholerun_banco_full_hpcg/manifest.csv",
    "MG": "data/solo_wholerun_banco_full_mg/manifest.csv",
    "SP": "data/solo_wholerun_banco_full_sp/manifest.csv",
    "FT": "data/solo_wholerun_banco_full_ft/manifest.csv",
    "LU": "data/solo_wholerun_banco_full_lu/manifest.csv",
    "AMG": "data/solo_wholerun_banco_full_amg/manifest.csv",
    "miniFE": "data/solo_wholerun_banco_full_minife/manifest.csv",
}

# Candidate pool considered for the signature, before any pruning.
CANDIDATE_FEATURES = [
    "dram_bw_24core", "dram_bw_growth", "dram_bw_mean", "dram_bw_max",
    "bw_scaling_shape", "tma_backend_mean", "mpki_mean", "llc_miss_mean",
    "ipc_degradation", "speedup_24",
]

# Within each group of mutually-redundant features, which one to keep --
# a stated, auditable human judgment call, kept separate from the
# automatic redundancy detection below. Preference: the most directly
# interpretable, least-derived value (a single measured point over a
# mean/max/difference computed from several points).
PREFERRED_SURVIVOR = {
    "dram_bw_24core": 0, "dram_bw_mean": 1, "dram_bw_max": 2, "dram_bw_growth": 3,
    "speedup_24": 0, "ipc_degradation": 1,
}

# Collinearity threshold: conventional in feature-selection practice,
# flags a pair as carrying materially overlapping information rather than
# two independent facets. Pairs below this are kept as-is even if moderately
# correlated (some overlap between physically-related quantities is
# expected and acceptable; near-duplication is not).
REDUNDANCY_THRESHOLD = 0.85


def build_app_signature(manifest_path):
    """One app's 8 candidate features from its quality-filtered 1-24 core
    solo sweep. Returns None if core=1 or core=24 data is missing."""
    rows = list(csv.DictReader(open(manifest_path)))
    rows = [r for r in rows if r.get("idle_threshold_met") == "True"
            and r.get("multiplex_status") == "OK"]
    by_cc = {}
    for r in rows:
        by_cc.setdefault(int(r["core_count"]), []).append(r)

    curve = {}
    for cc, reps in by_cc.items():
        curve[cc] = dict(
            dram_bw=statistics.mean(float(r["dram_read_gbps"]) + float(r["dram_write_gbps"]) for r in reps),
            ipc=statistics.mean(float(r["ipc"]) for r in reps),
            tma_backend=statistics.mean(float(r["tma_backend"]) for r in reps),
            mpki=statistics.mean(float(r["mpki"]) for r in reps),
            llc_miss=statistics.mean(float(r["llc_miss_rate"]) for r in reps),
            wall_time=statistics.mean(float(r["wall_time_s"]) for r in reps),
        )
    if 1 not in curve or 24 not in curve:
        return None

    peak_cc = max(curve, key=lambda cc: curve[cc]["dram_bw"])

    return {
        "dram_bw_24core": curve[24]["dram_bw"],
        "dram_bw_growth": curve[24]["dram_bw"] - curve[1]["dram_bw"],
        "dram_bw_mean": statistics.mean(c["dram_bw"] for c in curve.values()),
        "dram_bw_max": curve[peak_cc]["dram_bw"],
        "bw_scaling_shape": curve[24]["dram_bw"] / curve[peak_cc]["dram_bw"],
        "tma_backend_mean": statistics.mean(c["tma_backend"] for c in curve.values()),
        "mpki_mean": statistics.mean(c["mpki"] for c in curve.values()),
        "llc_miss_mean": statistics.mean(c["llc_miss"] for c in curve.values()),
        "ipc_degradation": 1 - curve[24]["ipc"] / curve[1]["ipc"],
        "speedup_24": curve[1]["wall_time"] / curve[24]["wall_time"],
    }


def correlation_matrix(app_signatures, features):
    apps = list(app_signatures.keys())
    X = np.array([[app_signatures[a][f] for f in features] for a in apps])
    return np.corrcoef(X.T), apps, X


def find_redundant_pairs(corr, features, threshold):
    flagged = []
    for i, a in enumerate(features):
        for j, b in enumerate(features):
            if i < j and abs(corr[i, j]) >= threshold:
                flagged.append((a, b, corr[i, j]))
    return sorted(flagged, key=lambda x: -abs(x[2]))


def redundant_groups(flagged_pairs, all_features):
    """Connected components over the redundancy graph: if A-B and B-C are
    both flagged, A/B/C form ONE group to pick a single survivor from, even
    if A-C itself wasn't directly flagged (transitively redundant via B).
    A simple independent per-pair drop rule breaks once more than 2
    features are mutually redundant, which is exactly what happened once
    dram_bw_mean/dram_bw_max joined dram_bw_24core/dram_bw_growth."""
    parent = {f: f for f in all_features}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b, _ in flagged_pairs:
        union(a, b)

    groups = {}
    for f in all_features:
        groups.setdefault(find(f), []).append(f)
    return [g for g in groups.values() if len(g) > 1]


def main():
    app_sig = {}
    for app, rel_path in SOLO_APPS.items():
        sig = build_app_signature(REPO / rel_path)
        if sig is None:
            print(f"WARNING: {app} missing core=1 or core=24 data, skipped")
            continue
        app_sig[app] = sig

    corr, apps, X = correlation_matrix(app_sig, CANDIDATE_FEATURES)

    print(f"Correlation matrix across {len(apps)} apps, "
          f"{len(CANDIDATE_FEATURES)} candidate signature features\n")
    header = " " * 24 + " ".join(f"{f[:10]:>11s}" for f in CANDIDATE_FEATURES)
    print(header)
    for i, f in enumerate(CANDIDATE_FEATURES):
        row = " ".join(f"{corr[i, j]:11.2f}" for j in range(len(CANDIDATE_FEATURES)))
        print(f"{f[:23]:23s} {row}")

    flagged = find_redundant_pairs(corr, CANDIDATE_FEATURES, REDUNDANCY_THRESHOLD)
    print(f"\nPairs at or above |r|={REDUNDANCY_THRESHOLD} (treated as redundant):")
    if not flagged:
        print("  (none)")
    for a, b, v in flagged:
        print(f"  corr({a}, {b}) = {v:.3f}")

    groups = redundant_groups(flagged, CANDIDATE_FEATURES)
    print(f"\nRedundancy groups (connected components -- one survivor kept per group):")
    dropped = set()
    for group in groups:
        survivor = min(group, key=lambda f: PREFERRED_SURVIVOR.get(f, 99))
        losers = [f for f in group if f != survivor]
        dropped.update(losers)
        print(f"  {sorted(group)} -> keep '{survivor}', drop {sorted(losers)}")

    final_features = [f for f in CANDIDATE_FEATURES if f not in dropped]
    print(f"\nDropped (redundant with a kept feature): {sorted(dropped)}")
    print(f"Final signature ({len(final_features)} features): {final_features}")

    print(f"\n(n={len(apps)} apps -- correlation estimates from this few points "
          f"are directional, not precise; re-check if the app set grows materially)")


if __name__ == "__main__":
    main()
