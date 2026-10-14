# Interference-Aware Pack Scheduling

Solo-profiling and pairwise co-location (interference) data collected on
`banco` (dual-socket Xeon E5-2680 v3, 24 cores, NUMA node0=even cores /
node1=odd cores), under the `performance` CPU governor, no `perf -I`
interval sampling (whole-run counters only, per the ML prediction document's
stated scope).

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
- `data/colocation_banco/manifest.csv` -- pairwise co-location (interference)
  results: 16 app pairs (chosen via k-means archetype clustering on the solo
  profiles, not the full C(12,2)=66 combinations), each run at 3 core splits
  (12/12 symmetric -- one NUMA node per app; 8/16 and 16/8 asymmetric --
  one app's cores spill onto the other's socket), 1 rep each. Columns:
  `app_a`/`app_b`, `core_split_a`/`core_split_b`, exact `cores_a`/`cores_b`
  used (NUMA fill order, `;`-separated), `t_solo_a`/`t_solo_b` (freshly
  measured on the SAME exact cores as the co-run, no perf overhead),
  `t_co_a`/`t_co_b`, and `slowdown_a`/`slowdown_b` = t_co/t_solo.
  `data/colocation_banco/solo_baselines.csv` has the raw baseline
  measurements; `data/colocation_banco/pair_<A>_<B>_s<split>/rep_<n>.csv`
  has one file per (pair, split, rep).
- `config/colocation_pairs_banco.yaml` -- the 16 chosen pairs with rationale
  comments explaining the archetype-clustering selection.
- `collectors/collect_colocation_banco.py` -- the co-location collector:
  disjoint NUMA-aware core split (see script docstring for why a plain
  contiguous `taskset` range doesn't work on a dual-socket machine), no perf
  instrumentation during timing, sustained-pressure relaunching so neither
  app gets a head start, solo baselines always freshly re-measured on the
  exact co-run core set (never reused from the whole-run solo sweep, since
  the "B" side's cores are usually the complement of "A"'s, not the same
  physical cores the solo sweep would have used).

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

### Co-location (pairwise interference) findings

- **split=12 (one NUMA node per app) is a clean negative control** -- every
  one of the 16 pairs shows slowdown within 0.97-1.02x, essentially zero
  measurable interference. Expected: at this split neither app shares any
  local hardware resource (separate memory controller, separate L3) with
  the other.
- **splits 8/16 (one app's cores spill onto the other's socket) show real,
  concentrated interference** -- not spread evenly across all 16 pairs.
  Standouts: **CG+SP at split=16, CG takes a 46.6% slowdown** (the largest
  effect in the dataset); **MG+miniFE** shows a clear aggressor/victim
  pattern (miniFE takes a 25.7% hit at split=16 while MG is barely
  touched, 1.1%).
- Apps picked as "good scaler, moderate demand" archetype representatives
  (ExaMiniMD, Quicksilver) show consistently mild interference (1-4%)
  across nearly every pairing; the large effects concentrate in the
  "backend-stalled" and "high-bandwidth" archetypes (CG, SP, MG, miniFE,
  HPCG) -- a reasonable cross-validation that the archetype clustering is
  separating apps that matter for contention from ones that don't.
- **First-pass linear model (own_bw, partner_bw, bw_prod, ipc_sum features,
  same recipe as the earlier validated pairwise model) does not generalize
  well**: leave-one-pair-out R²=0.083, and it badly underestimates the
  single largest case (CG+SP split=16: true 1.466, predicted 1.054). Likely
  causes under investigation: the feature set has no signal for whether the
  two apps' cores actually share a socket (split=12 vs 8/16 look identical
  to the model otherwise), the small sample size (16 pairs), and a
  known saturation nonlinearity near peak DRAM bandwidth that a pure
  multiplicative interaction term doesn't capture (see literature survey
  notes -- Guo et al. and Zhao et al. independently found the same issue
  and fixed it with piecewise/banded regression on the aggregate-demand
  feature).
