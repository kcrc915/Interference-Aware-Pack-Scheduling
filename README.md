# Interference-Aware Pack Scheduling -- Solo Profiling Data

Whole-run solo-profiling results collected on `banco` (dual-socket Xeon
E5-2680 v3, 24 cores, NUMA node0=even cores / node1=odd cores), under the
`performance` CPU governor, no `perf -I` interval sampling (whole-run counters
only, per the ML prediction document's stated scope).

## Layout

- `data/<app>/manifest.csv` -- one row per (core_count, rep): 1-24 cores,
  4 reps each, for 12 apps. Columns include instructions/cycles/IPC,
  top-down (TMA) breakdown, MPKI/LLC miss rate, DRAM read+write bandwidth,
  task-clock, idle-check status, and PMU multiplexing-coverage status
  (`multiplex_status` -- all rows here are `OK`, meaning no counter
  multiplexing occurred).
- `collectors/` -- the actual collection scripts that produced this data:
  - `collect_solo.py` -- base config/NUMA-pinning/command-building/idle-check
    logic, reused by the whole-run collector.
  - `collect_solo_wholerun.py` -- the whole-run (no interval sampling)
    collection driver actually used for this dataset.
- `plotting/plot_scaling.py` -- generates scaling plots per app (wall time,
  speedup vs ideal-linear, IPC, DRAM bandwidth, all vs core count) plus one
  cross-app speedup comparison plot.
- `figures/` -- plot output (generated, not checked in until you run the script).

## Apps included

XSBench, LULESH, HPCG, Quicksilver, ExaMiniMD, MG, miniFE, AMG, FT, CG, SP, LU.

**EP is intentionally excluded from this push.** Its wall-clock timing showed
a discrete bimodal pattern across reps (two distinct speeds ~1.3-1.5x apart)
most likely caused by Intel Turbo Boost not being disabled on `banco` (only
the CPU governor was confirmed set to `performance`; turbo is a separate
setting). EP is pure compute with negligible memory stalls, making it the
most sensitive "canary" for clock-frequency variability -- none of the other
12 apps show this issue, so they're unaffected. EP will be re-collected after
turbo is confirmed disabled.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python plotting/plot_scaling.py            # all apps
python plotting/plot_scaling.py HPCG CG    # just these
```

## Known findings from this dataset (see project notes for full writeup)

- **HPCG barely scales** (~1.1x speedup at 24 cores vs 1 core) -- it
  saturates DRAM bandwidth already at 1 core (~10.5 GB/s), so additional
  cores mostly add synchronization overhead rather than throughput.
- **Quicksilver shows a sharp NUMA penalty** crossing from 12 to 13 cores
  (the socket boundary): wall time jumps ~36% at that exact transition.
  Most other apps are mildly *helped* by the 13th core despite the
  cross-socket cost.
- **CG, FT, LU, ExaMiniMD scale best** (15-20x speedup at 24 cores),
  consistent with low serial/communication overhead.
