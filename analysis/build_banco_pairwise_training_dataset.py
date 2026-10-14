#!/usr/bin/env python3
"""
Builds the pairwise co-location training dataset for banco (24-core, dual
socket). One row per (app, co-run instance) -- each co-run manifest row
produces 2 rows (app-A's perspective and app-B's perspective), since the
label (slowdown) and the own/partner feature split are not symmetric.

Feature philosophy (revised 2026-10, per explicit decision to stop
hand-picking a small "physically motivated" subset and instead include
every usable raw solo-profile metric, own + partner + sum + product
interaction, then let systematic LOPO search decide what actually
generalizes -- see analysis/search_feature_combinations.py):

RAW_METRICS pulled from the solo whole-run manifest, for both own and
partner side, at each side's allocated core count in this co-run:
  ipc, dram_bw (read+write combined), tma_frontend, tma_backend,
  tma_bad_spec, tma_retiring, mpki, llc_miss_rate, llc_loads,
  llc_load_misses, task_clock_ms, page_faults, context_switches,
  cpu_migrations.
Deliberately EXCLUDED: rss_peak_bytes/rss_mean_bytes (empty for all banco
data -- psutil still not installed there), wall_time_s/instructions/cycles
(these ARE core_count-dependent outcome measures, not independent demand
signals -- ipc/task_clock_ms already summarize them), timestamp/rep
(identifiers, not features), idle_pct/min_enabled_pct/multiplex_status
(data-quality/collection metadata, not a performance signal).

For each raw metric X, four columns are emitted: own_X, partner_X,
X_sum = own_X + partner_X, X_prod = own_X * partner_X -- same interaction
pattern already validated for bandwidth (bw_prod) and IPC (ipc_sum) on the
heliodor model, now applied uniformly to every metric so search can find
which ones (if any) carry real signal beyond bandwidth/IPC.

Plus the NUMA structural features (own perspective only, not own/partner
since they're inherently relative):
  - exposed_frac: fraction of this app's own cores on a NUMA node the
    partner also has cores on. 0 at split=12 (confirmed zero interference).
  - node_ratio: partner's core count on the shared node(s) / this app's own
    core count there. Empirically mixed (helps extreme cases, hurts overall
    LOPO R^2 when forced into every pair) -- kept for search to evaluate.

Label: slowdown = t_co / t_solo, both measured on the identical physical
core set (see collect_colocation_banco.py).

Output: analysis/banco_pairwise_training_dataset.csv
"""
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "colocation_banco" / "manifest.csv"
OUT = REPO / "analysis" / "banco_pairwise_training_dataset.csv"

# NOTE: path layout here matches THIS repo's data/<App>/manifest.csv
# convention -- differs from the main project repo (phase-aware-scheduling),
# which uses data/solo_wholerun_banco_full_<app>/manifest.csv for the same
# underlying data. Same schema either way, just a different directory name.
SOLO_APPS = {
    "ExaMiniMD": "data/ExaMiniMD/manifest.csv",
    "Quicksilver": "data/Quicksilver/manifest.csv",
    "XSBench": "data/XSBench/manifest.csv",
    "CG": "data/CG/manifest.csv",
    "LULESH": "data/LULESH/manifest.csv",
    "HPCG": "data/HPCG/manifest.csv",
    "MG": "data/MG/manifest.csv",
    "SP": "data/SP/manifest.csv",
    "FT": "data/FT/manifest.csv",
    "LU": "data/LU/manifest.csv",
    "AMG": "data/AMG/manifest.csv",
    "miniFE": "data/miniFE/manifest.csv",
}

NODE0 = set(range(0, 24, 2))  # evens -- confirmed via numactl --hardware on banco
NODE1 = set(range(1, 24, 2))  # odds

# raw manifest column -> feature name. dram_bw is special-cased (sum of two
# columns) below rather than listed here.
RAW_METRICS = [
    "ipc", "tma_frontend", "tma_backend", "tma_bad_spec", "tma_retiring",
    "mpki", "llc_miss_rate", "llc_loads", "llc_load_misses",
    "task_clock_ms", "page_faults", "context_switches", "cpu_migrations",
]


def load_solo_features():
    """Returns {(app, core_count): {metric: value}} for core_count in
    {8,12,16} -- the only allocations used in this corun experiment design."""
    features = {}
    for app, rel_path in SOLO_APPS.items():
        rows = list(csv.DictReader(open(REPO / rel_path)))
        for cc in (8, 12, 16):
            matching = [r for r in rows if int(r["core_count"]) == cc]
            vals = {m: statistics.mean(float(r[m]) for r in matching) for m in RAW_METRICS}
            vals["dram_bw"] = statistics.mean(
                float(r["dram_read_gbps"]) + float(r["dram_write_gbps"]) for r in matching
            )
            features[(app, cc)] = vals
    return features


def exposure_feats(cores_mine, cores_partner):
    """(exposed_frac, node_ratio) -- see module docstring."""
    mine, theirs = set(cores_mine), set(cores_partner)
    my_nodes = set(0 if c in NODE0 else 1 for c in mine)
    shared_nodes = set(
        n for n in my_nodes if theirs & (NODE0 if n == 0 else NODE1)
    )
    my_on_shared = sum(1 for c in mine if (0 if c in NODE0 else 1) in shared_nodes)
    partner_on_shared = sum(1 for c in theirs if (0 if c in NODE0 else 1) in shared_nodes)
    exposed_frac = my_on_shared / len(mine)
    node_ratio = (partner_on_shared / my_on_shared) if my_on_shared else 0.0
    return exposed_frac, node_ratio


ALL_METRICS = RAW_METRICS + ["dram_bw"]


def build_row(app, partner, core_count, partner_core_count, cores_mine, cores_partner,
              split_label, pair_id, rep, slowdown, feats):
    f_own = feats[(app, core_count)]
    f_partner = feats[(partner, partner_core_count)]
    exposed_frac, node_ratio = exposure_feats(cores_mine, cores_partner)

    row = {
        "pair_id": "+".join(pair_id),
        "app": app,
        "partner": partner,
        "core_split_label": split_label,
        "own_core_count": core_count,
        "partner_core_count": partner_core_count,
        "rep": rep,
    }
    for m in ALL_METRICS:
        ov, pv = f_own[m], f_partner[m]
        row[f"own_{m}"] = ov
        row[f"partner_{m}"] = pv
        row[f"{m}_sum"] = ov + pv
        row[f"{m}_prod"] = ov * pv
    row["exposed_frac"] = exposed_frac
    row["node_ratio"] = node_ratio
    row["slowdown"] = slowdown
    return row


def main():
    feats = load_solo_features()
    manifest_rows = list(csv.DictReader(open(MANIFEST)))

    out_rows = []
    for r in manifest_rows:
        a, b = r["app_a"], r["app_b"]
        ca, cb = int(r["core_split_a"]), int(r["core_split_b"])
        cores_a = [int(x) for x in r["cores_a"].split(";")]
        cores_b = [int(x) for x in r["cores_b"].split(";")]
        pair_id = tuple(sorted([a, b]))
        split_label = f"{ca}_{cb}"

        out_rows.append(build_row(
            a, b, ca, cb, cores_a, cores_b, split_label, pair_id,
            r["rep"], float(r["slowdown_a"]), feats,
        ))
        out_rows.append(build_row(
            b, a, cb, ca, cores_b, cores_a, split_label, pair_id,
            r["rep"], float(r["slowdown_b"]), feats,
        ))

    fieldnames = list(out_rows[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"Wrote {len(out_rows)} rows ({len(manifest_rows)} co-run measurements x 2 "
          f"perspectives), {len(fieldnames)} columns ({len(ALL_METRICS)} raw metrics x "
          f"4 own/partner/sum/prod + 2 NUMA features + 7 metadata/label) to {OUT}")


if __name__ == "__main__":
    main()
