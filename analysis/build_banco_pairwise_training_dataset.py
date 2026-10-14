#!/usr/bin/env python3
"""
Builds the pairwise co-location training dataset for banco (24-core, dual
socket). One row per (app, co-run instance) -- each co-run manifest row
produces 2 rows (app-A's perspective and app-B's perspective), since the
label (slowdown) and the own/partner feature split are not symmetric.

Features, all derived from data already collected:
  - own_bw, own_ipc: this app's solo-profile DRAM bandwidth (read+write) and
    IPC at its allocated core count in this co-run.
  - partner_bw, partner_ipc: the co-runner's solo-profile features at ITS
    allocated core count.
  - bw_prod = own_bw * partner_bw, ipc_sum = own_ipc + partner_ipc: the
    interaction-term features already validated on the heliodor pairwise
    model (~55% R^2, ~4.88% MAPE there).
  - exposed_frac: fraction of this app's own cores that sit on a NUMA node
    the partner ALSO has cores on. 0 at split=12 (clean one-socket-each,
    confirmed zero interference); nonzero at splits 8/16 where cores
    genuinely share a socket. Captures the "is contention even physically
    possible" structural fact the raw bw/ipc features are blind to.
  - node_ratio: partner's core count on the shared node(s) divided by this
    app's own core count there -- a relative-crowding measure. Empirically
    found (LOPO test, 2026-10) to IMPROVE fit for extreme cases like
    CG+SP@split16 but HURT overall LOPO R^2 when forced into the model for
    every pair -- kept in this dataset as a column for experimentation, not
    because it's already proven useful as a global feature.

Label: slowdown = t_co / t_solo, both measured on the identical physical
core set (see collect_colocation_banco.py).

Output: analysis/banco_pairwise_training_dataset.csv
"""
import csv
import json
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


def load_solo_features():
    """Returns {(app, core_count): {dram_bw, ipc}} for core_count in {8,12,16}
    -- the only allocations used in this corun experiment design."""
    features = {}
    for app, rel_path in SOLO_APPS.items():
        rows = list(csv.DictReader(open(REPO / rel_path)))
        for cc in (8, 12, 16):
            matching = [r for r in rows if int(r["core_count"]) == cc]
            dram = statistics.mean(
                float(r["dram_read_gbps"]) + float(r["dram_write_gbps"]) for r in matching
            )
            ipc = statistics.mean(float(r["ipc"]) for r in matching)
            features[(app, cc)] = {"dram_bw": dram, "ipc": ipc}
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


def build_row(app, partner, core_count, partner_core_count, cores_mine, cores_partner,
              split_label, pair_id, rep, slowdown, feats):
    f_own = feats[(app, core_count)]
    f_partner = feats[(partner, partner_core_count)]
    exposed_frac, node_ratio = exposure_feats(cores_mine, cores_partner)
    return {
        "pair_id": "+".join(pair_id),
        "app": app,
        "partner": partner,
        "core_split_label": split_label,
        "own_core_count": core_count,
        "partner_core_count": partner_core_count,
        "rep": rep,
        "own_bw": f_own["dram_bw"],
        "own_ipc": f_own["ipc"],
        "partner_bw": f_partner["dram_bw"],
        "partner_ipc": f_partner["ipc"],
        "bw_prod": f_own["dram_bw"] * f_partner["dram_bw"],
        "ipc_sum": f_own["ipc"] + f_partner["ipc"],
        "exposed_frac": exposed_frac,
        "node_ratio": node_ratio,
        "slowdown": slowdown,
    }


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
          f"perspectives) to {OUT}")


if __name__ == "__main__":
    main()
