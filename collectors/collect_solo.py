#!/usr/bin/env python3
"""
Phase A solo metric collection driver (experiment_plan.md §2).

For each (app, core_count, rep):
  1. Idle-check the target cores (mpstat) before running — abort/retry if busy.
  2. Run 4 sequential `perf stat -a -C <cores> -I <ms>` passes, each a metric
     group verified empirically to run at 100% counter duty cycle on this
     hardware (see config/platform.yaml -> multiplexing_resolution). A single
     combined pass multiplexes badly on this CPU, so re-running the app once
     per group is the tradeoff made to get clean per-interval numbers.
     The first pass also launches a *concurrent companion* perf process that
     reads uncore_imc (DRAM bandwidth) pinned to a single fixed CPU (-C 0),
     never to the app's actual core_list. This matters: uncore counters are
     chip-wide, and `perf stat -a -C <list>` silently re-reads (and sums) the
     same uncore counter once per CPU named in the list — with core_count>1
     that inflates DRAM bandwidth by roughly core_count x. Confirmed
     empirically (identical workload: -C 0 -> correct value, -C 0,1,2 ->
     ~2.9x, -C 0..5 -> ~6.9x). A single fixed-CPU reader avoids it entirely
     since the whole chip's traffic is visible from any one CPU.
  3. Sample RSS (VmRSS, summed over the process tree) during pass 1 only —
     RSS doesn't depend on which perf events are active.
  4. Merge the passes' interval rows (by interval index) into one CSV
     following the schema in experiment_plan.md §2.6.

Output: data/solo/<app>/cores_<N>/rep_<K>.csv, plus data/solo/manifest.csv.
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None  # RSS sampling becomes a no-op (see RssSampler.run) rather
    # than a hard blocker -- lets collection proceed on a machine where psutil
    # isn't installed yet, at the cost of rss_bytes being empty in the output.
    # RSS isn't currently used as a model feature anyway (flagged separately),
    # so this is a safe, explicit degradation, not a silent one: every run
    # with psutil unavailable prints a one-time warning (see main()).
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "config"
DATA_DIR = REPO / "data" / "solo"
# Secondary comparison dataset, aligned with Method C (instructions-quantile
# fraction) instead of the primary Method F (banded DTW on IPC). Same raw
# per-pass counter data, just a different post-hoc alignment -- computed
# alongside the primary dataset at negligible extra cost (no extra benchmark
# executions), specifically to let later analysis quantify how much the
# alignment-method choice affects downstream results. See merge_and_write.
DATA_DIR_ALT_C = REPO / "data" / "solo_altC"
_SHM = Path("/dev/shm")
TMP_ROOT = (_SHM / "phase_a_collect_tmp") if _SHM.is_dir() else (REPO / ".tmp_perf_fallback")
SAMPLE_MS = 100
# 90% was the original target but this machine's true idle floor is ~75-85% (persistent
# kworker I/O), so 90% never passes and the retry loop was pure overhead. 50% was picked
# post-hoc by inspecting the one confirmed-contaminated trial (idle_pct=0.0) against the
# normal 75-99% range everything else falls in — it's calibrated to catch real contention
# episodes without false-positiving on this machine's normal background load.
IDLE_THRESHOLD_PCT = 50.0
IDLE_MAX_RETRIES = 5
IDLE_RETRY_WAIT_S = 5
ENFORCE_IDLE = True  # 2026-08-20: enabled for the real collection run (previously False
# during pilot iteration per user request). Safe now: threshold is realistically
# calibrated to 50%, so this should rarely retry in practice, and it closes off the
# CG-cores1-rep1-style contamination case instead of just flagging it after the fact.
UNCORE_READER_CPU = "0"  # fixed single CPU for chip-wide uncore reads, see module docstring
# The uncore companion starts ~0.1s before the main pass, so its interval k
# corresponds to roughly the main pass's interval k-1. Empirically determined
# across all 460 usable trials (see merge_and_write docstring): +1.
DRAM_COMPANION_SHIFT = 1

_UNCORE_IMC_EVENTS = None  # cache for detect_uncore_imc_events()


def detect_uncore_imc_events():
    """The uncore_imc DRAM-bandwidth event names differ by CPU generation:
    client chips (heliodor, i7-8700) expose uncore_imc/data_reads|data_writes/,
    server chips (banco, Xeon E5-2680 v3) only expose per-channel PMU instances
    (uncore_imc_0, uncore_imc_1, ...) and the generic aliases are named
    cas_count_read/cas_count_write instead. Both report already-scaled MiB in
    perf's -x csv output, so only the event names need to vary -- probe
    `perf list` once and cache the result rather than hardcoding one name."""
    global _UNCORE_IMC_EVENTS
    if _UNCORE_IMC_EVENTS is not None:
        return _UNCORE_IMC_EVENTS
    try:
        out = subprocess.run(["perf", "list"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        out = ""
    if "uncore_imc/data_reads/" in out:
        _UNCORE_IMC_EVENTS = ("uncore_imc/data_reads/", "uncore_imc/data_writes/")
    elif "uncore_imc/cas_count_read/" in out:
        _UNCORE_IMC_EVENTS = ("uncore_imc/cas_count_read/", "uncore_imc/cas_count_write/")
    else:
        _UNCORE_IMC_EVENTS = (None, None)
    return _UNCORE_IMC_EVENTS

PASSES = [
    {
        "name": "core_cache",
        # page-faults/context-switches/cpu-migrations are kernel software events,
        # not hardware PMU counters -- they don't compete with the core-events'
        # counter budget (verified empirically: identical 100% duty cycle with
        # or without them), so they ride along on this pass for free. Useful for
        # telling "this interval was slow from OS scheduling noise" apart from
        # "this interval was slow from genuine microarchitectural contention",
        # which matters once Phase B co-location starts.
        "events": "instructions,cycles,task-clock,LLC-loads,LLC-load-misses,"
                   "page-faults,context-switches,cpu-migrations",
        "extra_events": None,
        "metrics": None,
    },
    {
        "name": "tma_fe_be",
        "events": "instructions,cycles,task-clock",
        "extra_events": None,
        "metrics": "tma_frontend_bound,tma_backend_bound",
    },
    {
        "name": "tma_bad_spec",
        "events": "instructions,cycles,task-clock",
        "extra_events": None,
        "metrics": "tma_bad_speculation",
    },
    {
        "name": "tma_retiring",
        "events": "instructions,cycles,task-clock",
        "extra_events": None,
        "metrics": "tma_retiring",
    },
]


def expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def load_config():
    with open(CONFIG_DIR / "applications.yaml") as f:
        return yaml.safe_load(f)


_NUMA_TOPOLOGY = None  # cached: list of [cpu_ids] per node, in node order


def discover_numa_topology():
    """Parses `numactl --hardware` at runtime rather than hardcoding core IDs
    -- on banco.ittc.ku.edu (dual-socket Xeon E5-2680 v3) node 0 is the EVEN
    cores (0,2,4,...22) and node 1 is the ODD cores (1,3,5,...23), confirmed
    via direct check -- NOT contiguous blocks (0-11 / 12-23) as a naive
    assumption would have it. A plain `taskset -c 0-N` would therefore split
    across both sockets starting at core_count=2, not core_count=13,
    silently contaminating every multi-core solo measurement with
    cross-socket interconnect traffic. Querying at runtime (instead of
    hardcoding this machine's specific layout) means the same code stays
    correct if ever run on a machine with a different NUMA layout, or a
    single-socket machine where this just returns one node covering every
    core (the i7-8700 case -- single NUMA node, no behavior change there)."""
    global _NUMA_TOPOLOGY
    if _NUMA_TOPOLOGY is not None:
        return _NUMA_TOPOLOGY
    out = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True).stdout
    nodes = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("node") and "cpus:" in line:
            parts = line.split("cpus:")
            node_id = int(parts[0].split()[1])
            cpus = [int(c) for c in parts[1].split()]
            nodes[node_id] = cpus
    _NUMA_TOPOLOGY = [nodes[k] for k in sorted(nodes)]
    return _NUMA_TOPOLOGY


def core_list(core_count: int):
    """Returns the actual list of core IDs to use for `core_count` cores,
    filling one NUMA node completely before spilling into the next -- so the
    scaling curve stays single-socket (NUMA-local) for as long as possible,
    with the cross-socket transition happening at one clean, known,
    analyzable point (node-size+1) rather than being smeared across every
    even core count from the start."""
    topology = discover_numa_topology()
    if not topology:  # numactl unavailable/parse failure -- fall back to naive range
        return list(range(core_count))
    cores = []
    for node_cpus in topology:
        if len(cores) >= core_count:
            break
        cores.extend(node_cpus[:max(0, core_count - len(cores))])
    return cores[:core_count]


def core_list_str(core_count: int) -> str:
    return ",".join(str(c) for c in core_list(core_count))


def numa_nodes_for(core_count: int):
    """Which NUMA node IDs are actually touched by core_list(core_count) --
    used for --membind so memory allocation stays local to whichever
    socket(s) the pinned cores are actually on."""
    topology = discover_numa_topology()
    cores = set(core_list(core_count))
    touched = [i for i, node_cpus in enumerate(topology) if cores & set(node_cpus)]
    return touched or [0]


def measure_idle(core_count: int):
    """Single 1s mpstat sample, no retries/waiting. Returns avg idle% or None."""
    cpus = core_list_str(core_count)
    out = subprocess.run(
        ["mpstat", "-P", cpus, "1", "1"], capture_output=True, text=True
    ).stdout
    idle_vals = []
    header_idx = None
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if "%idle" in parts:
            header_idx = parts.index("%idle")
            continue
        if header_idx is not None and (parts[0] == "Average:"):
            try:
                idle_vals.append(float(parts[header_idx]))
            except (ValueError, IndexError):
                pass
    return sum(idle_vals) / len(idle_vals) if idle_vals else None


def check_idle(core_count: int, enforce: bool = False):
    """Returns (idle_pct, threshold_met). When enforce=False (current default,
    per user request 2026-08-19), this only measures and records idle% — it
    never blocks/retries. Pass enforce=True to restore the blocking
    retry-until-quiet behavior for a later, stricter collection pass."""
    if not enforce:
        idle_pct = measure_idle(core_count)
        met = idle_pct is not None and idle_pct >= IDLE_THRESHOLD_PCT
        return idle_pct, met

    idle_pct = None
    for attempt in range(1, IDLE_MAX_RETRIES + 1):
        idle_pct = measure_idle(core_count)
        if idle_pct is not None and idle_pct >= IDLE_THRESHOLD_PCT:
            return idle_pct, True
        print(
            f"  [idle-check] attempt {attempt}/{IDLE_MAX_RETRIES}: "
            f"avg idle {idle_pct}% < {IDLE_THRESHOLD_PCT}%, waiting...",
            file=sys.stderr,
        )
        time.sleep(IDLE_RETRY_WAIT_S)
    print("  [idle-check] FAILED after retries, proceeding anyway (flagged in manifest)",
          file=sys.stderr)
    return idle_pct, False


def factorize_3d(n: int):
    """Split n into 3 positive factors (npx, npy, npz) with npx*npy*npz == n,
    as balanced as possible (close to n**(1/3) each). Needed for MPI apps like
    miniAMR that require an explicit process grid decomposition matching the
    rank count exactly -- confirmed necessary empirically: miniAMR aborts with
    "number of processors used does not match number allocated" without it.
    For prime n (e.g. 5) the best available factorization is 1x1xn."""
    best = (1, 1, n)
    best_spread = n  # smaller = more balanced
    for a in range(1, int(round(n ** (1 / 3))) + 2):
        if n % a:
            continue
        rem = n // a
        for b in range(a, int(rem ** 0.5) + 2):
            if rem % b:
                continue
            c = rem // b
            spread = c - a  # c is always >= b >= a here
            if spread < best_spread:
                best_spread = spread
                best = (a, b, c)
    return best


def build_command(app: dict, core_count: int) -> list:
    args_tpl = app.get("args") or ""
    args_str = args_tpl.replace("{threads}", str(core_count))
    if "{npx}" in args_str or "{npy}" in args_str or "{npz}" in args_str:
        npx, npy, npz = factorize_3d(core_count)
        args_str = (args_str.replace("{npx}", str(npx))
                             .replace("{npy}", str(npy))
                             .replace("{npz}", str(npz)))
    args = args_str.split() if args_str.strip() else []
    binary = app["binary"]
    cores = core_list_str(core_count)
    mem_nodes = ",".join(str(n) for n in numa_nodes_for(core_count))

    # numactl, not plain taskset: pins execution to `cores` (--physcpubind,
    # same effect as taskset -c) AND binds memory allocation to whichever
    # NUMA node(s) those cores are actually on (--membind) -- without this,
    # first-touch allocation could still land a thread's memory on the
    # "wrong" socket relative to where it's pinned to compute, especially
    # once core_count spans both nodes. See core_list()/numa_nodes_for().
    numa_prefix = ["numactl", f"--physcpubind={cores}", f"--membind={mem_nodes}"]

    if app["parallel_mode"] == "mpi":
        cmd = numa_prefix + ["mpirun", "-np", str(core_count), binary] + args
        env_extra = {"OMP_NUM_THREADS": "1"}
    else:
        cmd = numa_prefix + [binary] + args
        env_extra = {
            "OMP_NUM_THREADS": str(core_count),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }
    return cmd, env_extra


class RssSampler(threading.Thread):
    """Poll VmRSS (summed over the process tree rooted at pid) every SAMPLE_MS."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid = pid
        self.samples = []  # list of (elapsed_s, rss_bytes)
        self._stop_event = threading.Event()

    def run(self):
        start = time.time()
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop_event.is_set():
            total = 0
            try:
                procs = [root] + root.children(recursive=True)
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.NoSuchProcess:
                break
            self.samples.append((time.time() - start, total))
            time.sleep(SAMPLE_MS / 1000.0)

    def stop(self):
        self._stop_event.set()


def start_uncore_companion(out_csv: Path):
    """Launch a standalone (no wrapped command) perf process reading DRAM
    bandwidth from a single fixed CPU. Runs until stop_uncore_companion()
    signals it. See module docstring for why this must be its own process
    pinned to one CPU rather than folded into the -C <core_list> pass."""
    read_ev, write_ev = detect_uncore_imc_events()
    perf_cmd = [
        "perf", "stat", "-a", "-C", UNCORE_READER_CPU, "-I", str(SAMPLE_MS), "-x", ",",
        "-e", f"{read_ev},{write_ev}",
        "-o", str(out_csv),
    ]
    return subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_uncore_companion(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_pass(cmd, env_extra, cwd, core_count, pass_def, out_csv, sample_rss=False,
             uncore_csv=None):
    events = pass_def["events"]
    perf_cmd = ["perf", "stat", "-a", "-C", core_list_str(core_count), "-I", str(SAMPLE_MS), "-x", ","]
    perf_cmd += ["-e", events]
    if pass_def["extra_events"]:
        perf_cmd += ["-e", pass_def["extra_events"]]
    if pass_def["metrics"]:
        perf_cmd += ["-M", pass_def["metrics"]]
    perf_cmd += ["-o", str(out_csv), "--"] + cmd

    env = os.environ.copy()
    env.update(env_extra)

    uncore_proc = start_uncore_companion(uncore_csv) if uncore_csv else None
    if uncore_proc:
        time.sleep(0.1)  # let it start sampling before the timed pass begins

    start = time.time()
    proc = subprocess.Popen(perf_cmd, cwd=cwd, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    sampler = None
    if sample_rss and psutil is not None:
        # perf itself is pid; the benchmark is perf's child. Give it a moment to spawn.
        time.sleep(0.05)
        sampler = RssSampler(proc.pid)
        sampler.start()

    _, stderr = proc.communicate()
    wall_time = time.time() - start

    if sampler:
        sampler.stop()
        sampler.join(timeout=1)

    if uncore_proc:
        stop_uncore_companion(uncore_proc)

    if proc.returncode != 0:
        print(f"  [WARN] perf pass '{pass_def['name']}' exit code {proc.returncode}: "
              f"{stderr.decode(errors='replace')[:300]}", file=sys.stderr)

    rss_samples = sampler.samples if sampler else []
    return wall_time, proc.returncode, rss_samples


def parse_perf_csv(path: Path):
    """Return (intervals, min_enabled_pct, instr_cum). intervals: interval_index ->
    {name: value}. instr_cum: interval_index -> cumulative instructions-retired up
    through that interval within this pass — kept so cross-pass merging can align by
    *fractional progress through each pass's own run* instead of raw positional
    index or wall-clock time. This matters because the 4 metric-group passes are 4
    separate re-executions of the same benchmark and drift apart in interval count
    (confirmed empirically up to ~13% at higher core counts, where relative
    scheduling/sync overhead is proportionally larger) — positional alignment
    silently pairs rows that were never actually simultaneous.
    Cumulative instructions (not wall-clock time) is the alignment axis: instruction
    count tracks actual algorithmic progress and is immune to the scheduling-jitter/
    frequency-variation noise that wall-clock time carries — every pass already
    collects "instructions" as a normalizing denominator for IPC/MPKI, so this is
    free, no extra collection needed. Same principle SimPoint uses instruction-count
    intervals over wall-clock ones (see docs/phase_a_overnight_report.md §3.3/§5).
    min_enabled_pct is the worst-case counter duty cycle seen across every event/interval
    in this pass — the real multiplexing-coverage signal (see §2.2 validation check),
    as opposed to just whether the perf process exited cleanly."""
    intervals = {}
    instr_cum = {}
    timestamps = {}
    last_ts = None
    idx = -1
    running_instr = 0.0
    min_enabled_pct = None
    if not path.exists():
        return intervals, min_enabled_pct, instr_cum, timestamps
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or not row[0]:
                continue
            try:
                ts = float(row[0])
            except ValueError:
                continue
            if ts != last_ts:
                idx += 1
                last_ts = ts
                intervals[idx] = {}
            fields = row + [""] * (8 - len(row))
            value, unit, event, _enabled, runpct, metric_val, metric_unit = fields[1:8]
            if event:
                try:
                    fval = float(value) if value else None
                except ValueError:
                    fval = None
                intervals[idx][event] = fval
                if event == "instructions" and fval is not None:
                    running_instr += fval
            if metric_unit and metric_val:
                name = metric_unit.strip().lstrip("%").strip()
                if name.startswith("tma_"):
                    try:
                        intervals[idx][name] = float(metric_val)
                    except ValueError:
                        pass
            if runpct:
                try:
                    pct = float(runpct)
                    if min_enabled_pct is None or pct < min_enabled_pct:
                        min_enabled_pct = pct
                except ValueError:
                    pass
            instr_cum[idx] = running_instr
            timestamps[idx] = ts
    return intervals, min_enabled_pct, instr_cum, timestamps


def nearest_rss(rss_samples, elapsed_s):
    if not rss_samples:
        return None
    best = min(rss_samples, key=lambda s: abs(s[0] - elapsed_s))
    return best[1]


class FractionalAligner:
    """Method C: maps a reference-pass interval index to the nearest interval of
    another pass, matched by fractional progress (quantile matching on
    cumulative instructions retired) through each pass's own run — see
    parse_perf_csv's docstring for why. Falls back to identity alignment
    gracefully if a pass has no data.

    Kept as a secondary/comparison alignment (see ALT_C_DIR below) — empirically
    outperformed by align_windowed_dtw_ipc (Method F) on real drifted data (see
    docs/phase_a_overnight_report.md addenda: F wins significantly on both
    validation metrics, 28/30 cases on instructions-agreement). F is used for
    the primary dataset; C is retained specifically to quantify how much the
    alignment-method choice affects downstream results."""

    def __init__(self, progress: dict):
        self.progress = progress
        self.total = max(progress.values()) if progress else None
        # precompute (fraction, idx) pairs sorted by fraction for nearest lookup
        self._fracs = sorted(
            ((v / self.total if self.total else 0.0), idx) for idx, v in progress.items()
        ) if progress else []

    def nearest_index(self, frac: float):
        if not self._fracs:
            return None
        # binary search would be nicer at scale; these traces are at most a few
        # thousand intervals, linear scan via min() is plenty fast here
        return min(self._fracs, key=lambda p: abs(p[0] - frac))[1]


def build_ipc_list(pass_data: dict, n: int):
    """Per-interval IPC (instructions/cycle) as a 0..n-1 indexed list, for the
    DTW alignment below. Every pass already collects instructions+cycles as
    baseline events (needed for its own IPC/MPKI), so this is free."""
    out = []
    for i in range(n):
        d = pass_data.get(i, {})
        instr, cyc = d.get("instructions"), d.get("cycles")
        out.append((instr / cyc) if instr and cyc else None)
    return out


def align_windowed_dtw_ipc(ipc_ref, ipc_other, band_frac=0.15):
    """Method F (primary alignment method): banded Dynamic Time Warping on IPC.

    Implements the general technique described in US Patent 8,798,962 B2
    (Bose, Buyuktosunoglu, Isci, Kephart, Meng, Sarikaya; IBM; filed 2011,
    published 2014), "Virtualized abstraction with built-in data alignment
    and simultaneous event monitoring in performance counter based
    application characterization and tuning" — windowed DTW using IPC as the
    pivot metric to align traces from multiple sequential runs of the same
    program captured with different counter groups, exactly our cross-pass
    alignment problem. This is our own implementation of the general approach
    the patent describes (real dynamic programming, not a reproduction of any
    proprietary source), used here to correctly process our own collected
    data — not offered as a novel technique. Adapted from the patent's frame
    size (100-300 samples, longer than several of our traces) to a
    proportional Sakoe-Chiba-style band (Sakoe & Chiba, 1978) so it scales
    for short traces too.

    Chosen as the primary alignment method after empirical comparison against
    5 alternatives (naive index, wall-clock-time fraction, instructions
    fraction [Method C, kept as a secondary comparison dataset], and two
    IPC-based nearest-neighbor variants) on real drifted data: F won
    significantly on both validation metrics (instructions-per-interval
    agreement and TMA-sum-deviation-from-100%), paired Wilcoxon p<0.01 in both
    cases. Full comparison: analysis/compare_alignment_rigorous.py and
    docs/phase_a_overnight_report.md addenda."""
    n, m = len(ipc_ref), len(ipc_other)
    if n == 0 or m == 0:
        return [None] * n

    band = max(3, int(round(band_frac * max(n, m))))
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    def in_band(i, j):
        center = i * m / n if n else 0
        return abs(j - center) <= band

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if not in_band(i - 1, j - 1):
                continue
            vi, vj = ipc_ref[i - 1], ipc_other[j - 1]
            d = abs(vi - vj) if (vi is not None and vj is not None) else 0.0
            best_prev = min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
            if best_prev < INF:
                cost[i][j] = d + best_prev

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        options = [
            (cost[i - 1][j - 1], i - 1, j - 1),
            (cost[i - 1][j], i - 1, j),
            (cost[i][j - 1], i, j - 1),
        ]
        _, i, j = min(options, key=lambda o: o[0])
    path.reverse()

    mapping = [None] * n
    for ref_idx, other_idx in path:
        mapping[ref_idx] = other_idx
    last = 0
    for k in range(n):
        if mapping[k] is None:
            mapping[k] = last
        else:
            last = mapping[k]
    return mapping


def compute_tma_mappings(method, n, core_data, pass_instr_cum, pass_results):
    """Return {pass_name: {ref_idx: matched_idx}} for the 3 TMA passes, using
    either method="F" (banded DTW on IPC, primary) or method="C" (instructions
    quantile fraction, secondary comparison dataset). See align_windowed_dtw_ipc
    and FractionalAligner docstrings for the empirical basis of this choice."""
    names = ["tma_fe_be", "tma_bad_spec", "tma_retiring"]
    if method == "F":
        core_ipc = build_ipc_list(core_data, n)
        out = {}
        for name in names:
            pass_data = pass_results[name]
            other_n = len(pass_data)
            other_ipc = build_ipc_list(pass_data, other_n)
            mapping_list = align_windowed_dtw_ipc(core_ipc, other_ipc)
            out[name] = {i: mapping_list[i] for i in range(n)}
        return out
    elif method == "C":
        core_instr = pass_instr_cum["core_cache"]
        core_total = max(core_instr.values()) if core_instr else None
        out = {}
        for name in names:
            aligner = FractionalAligner(pass_instr_cum[name])
            mapping = {}
            for i in range(n):
                frac = (core_instr.get(i, 0.0) / core_total) if core_total else 0.0
                mapping[i] = aligner.nearest_index(frac)
            out[name] = mapping
        return out
    else:
        raise ValueError(f"unknown alignment method {method!r}")


def merge_and_write(app_name, core_count, rep, pass_results, pass_instr_cum,
                     rss_samples, out_path, method="F", pass_timestamps=None):
    """core_cache (+ its concurrent mem_bw companion + the RSS sampler) all share
    one real timeline, since they run simultaneously — merged by index directly.
    The 3 TMA passes are separate sequential re-executions of the benchmark and
    drift out of sync with core_cache's own timeline (empirically up to ~13% at
    higher core counts), so they're realigned to it — by banded DTW on IPC
    (method="F", the primary/default, see align_windowed_dtw_ipc) or by
    instructions-quantile fraction (method="C", kept as a secondary comparison
    dataset, see FractionalAligner) rather than raw positional index.

    Three timing corrections applied 2026-08-24 after the round-2 audit
    (docs/phase_a_round2_audit.md); all three need real perf timestamps, which
    is why pass_timestamps is now threaded through:

      1. DRAM_COMPANION_SHIFT — the uncore companion is launched ~0.1s BEFORE
         the main pass, so its interval k covers a window ~1 interval earlier
         than core_cache's interval k. Merging by raw index therefore skewed
         every DRAM value against the counters sharing its row. Determined
         empirically across all 460 usable trials by cross-correlating
         LLC-load-misses against uncore reads (physically simultaneous, so
         true lag is 0): mean r = 0.351 at shift 0 vs 0.477 at shift +1
         (0.456 at +2, 0.376 at +3). +1 is the best single global correction.

      2. Bandwidth is now divided by the REAL interval duration taken from
         perf's own timestamps, not a hardcoded 0.1s. Real intervals are
         ~101.7ms, so the old denominator inflated every bandwidth figure by
         ~1.76%; the final interval of a run is typically truncated (often
         <50% of a full window), where the old code understated bandwidth by
         up to ~4x.

      3. RSS is looked up using the real elapsed timestamp instead of a
         synthetic clock that advanced by exactly 0.1s per interval. The
         synthetic clock drifted ~1.76% cumulatively — up to 124 intervals of
         misalignment in the longest BT runs.

    Also adds a tma_sum column: the four TMA components come from three
    separate executions, so their per-row sum is a direct confidence signal
    for that row's alignment quality (it should be ~100; rows far from that
    are chimeras of mismatched phases and should be filtered before any
    per-interval TMA analysis)."""
    core_data = pass_results["core_cache"]
    mem_data = pass_results["mem_bw"]
    fe_be_data = pass_results["tma_fe_be"]
    bad_spec_data = pass_results["tma_bad_spec"]
    retiring_data = pass_results["tma_retiring"]

    n = len(core_data)
    n_mem = len(mem_data)
    mappings = compute_tma_mappings(method, n, core_data, pass_instr_cum, pass_results)

    pass_timestamps = pass_timestamps or {}
    core_ts = pass_timestamps.get("core_cache", {})
    mem_ts = pass_timestamps.get("mem_bw", {})

    def window_of(ts_map, k, fallback=SAMPLE_MS / 1000.0):
        """Real duration of interval k, from perf's own timestamps. Interval k
        spans (ts[k-1], ts[k]]; interval 0 spans (0, ts[0]]. Catches both the
        ~101.7ms true interval and the truncated final interval."""
        if k not in ts_map:
            return fallback
        prev = ts_map.get(k - 1, 0.0) if k > 0 else 0.0
        d = ts_map[k] - prev
        return d if d > 1e-6 else fallback

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "app", "core_count", "rep", "interval_index",
        "instructions", "cycles", "ipc",
        "tma_frontend", "tma_backend", "tma_bad_spec", "tma_retiring",
        "mpki", "llc_miss_rate", "llc_loads", "llc_load_misses",
        "dram_read_gbps", "dram_write_gbps", "rss_bytes",
        "page_faults", "context_switches", "cpu_migrations",
        "tma_sum", "interval_s", "task_clock_ms",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        elapsed = 0.0
        for i in range(n):
            c = core_data.get(i, {})
            fb_idx = mappings["tma_fe_be"].get(i)
            bs_idx = mappings["tma_bad_spec"].get(i)
            rt_idx = mappings["tma_retiring"].get(i)
            fb = fe_be_data.get(fb_idx, {}) if fb_idx is not None else {}
            bs = bad_spec_data.get(bs_idx, {}) if bs_idx is not None else {}
            rt = retiring_data.get(rt_idx, {}) if rt_idx is not None else {}
            # DRAM companion leads the main pass by ~1 interval (see docstring)
            mem_i = i + DRAM_COMPANION_SHIFT
            m = mem_data.get(mem_i, {}) if 0 <= mem_i < n_mem else {}

            instr = c.get("instructions")
            cycles = c.get("cycles")
            llc_loads = c.get("LLC-loads")
            llc_misses = c.get("LLC-load-misses")
            # perf's task-clock event, in ms -- accumulated CPU time for this
            # interval, summed with interval_s across the whole run gives
            # U_i(p) for the utilization feature u_i(p) = U_i(p)/(p*T_i(p))
            # (doc §1.2 eq.6). Only collected by the core_cache pass.
            task_clock_ms = c.get("task-clock")
            _read_ev, _write_ev = detect_uncore_imc_events()
            dram_r = m.get(_read_ev)
            dram_w = m.get(_write_ev)
            page_faults = c.get("page-faults")
            context_switches = c.get("context-switches")
            cpu_migrations = c.get("cpu-migrations")

            ipc = (instr / cycles) if instr and cycles else None
            mpki = (llc_misses / instr * 1000) if llc_misses is not None and instr else None
            miss_rate = (llc_misses / llc_loads) if llc_misses is not None and llc_loads else None
            # dram_r/w are MiB accumulated over that companion interval; divide
            # by its REAL duration, not a hardcoded 0.1s
            mem_win = window_of(mem_ts, mem_i)
            dram_r_gbps = (dram_r * 1024 * 1024 / mem_win / 1e9) if dram_r is not None else None
            dram_w_gbps = (dram_w * 1024 * 1024 / mem_win / 1e9) if dram_w is not None else None

            core_win = window_of(core_ts, i)
            # RSS by real elapsed time, not a synthetic 0.1s-per-interval clock
            elapsed = core_ts.get(i, elapsed + core_win)
            rss = nearest_rss(rss_samples, elapsed)

            tma_vals = [fb.get("tma_frontend_bound"), fb.get("tma_backend_bound"),
                        bs.get("tma_bad_speculation"), rt.get("tma_retiring")]
            tma_sum = sum(v for v in tma_vals if v is not None) if all(
                v is not None for v in tma_vals) else None

            w.writerow([
                app_name, core_count, rep, i,
                instr, cycles, ipc,
                fb.get("tma_frontend_bound"), fb.get("tma_backend_bound"),
                bs.get("tma_bad_speculation"), rt.get("tma_retiring"),
                mpki, miss_rate, llc_loads, llc_misses,
                dram_r_gbps, dram_w_gbps, rss,
                page_faults, context_switches, cpu_migrations,
                tma_sum, core_win, task_clock_ms,
            ])
    return n


def append_manifest(row):
    manifest = DATA_DIR / "manifest.csv"
    header = [
        "app", "core_count", "rep", "timestamp", "wall_time_s", "interval_ms",
        "idle_pct", "idle_threshold_met", "idle_enforced", "min_enabled_pct",
        "multiplex_status", "num_intervals", "output_path",
    ]
    new = not manifest.exists()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def main():
    global DATA_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", nargs="*", help="subset of app names, default all")
    ap.add_argument("--cores", nargs="*", type=int, help="subset of core counts, default all")
    ap.add_argument("--reps", type=int, help="override repetition count")
    ap.add_argument("--data-dir", type=str, default=None,
                     help="override output directory (default: data/solo). Use "
                          "a separate dir per machine -- e.g. --data-dir data/solo_banco "
                          "-- so results from different hardware never land in the "
                          "same app/cores_N/rep_K.csv path and get mixed or overwritten.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.data_dir:
        DATA_DIR = REPO / args.data_dir
        print(f"[INFO] writing output to {DATA_DIR} (not the default data/solo)", file=sys.stderr)

    if psutil is None:
        print("[WARN] psutil not installed -- RSS sampling disabled, rss_bytes "
              "will be empty in output CSVs. Everything else is unaffected.",
              file=sys.stderr)

    cfg = load_config()
    apps = cfg["applications"]
    if args.apps:
        apps = [a for a in apps if a["name"] in args.apps]
    core_counts = args.cores or cfg["core_counts"]
    reps = args.reps or cfg["repetitions"]

    for app in apps:
        app_dir = Path(expand(app["app_dir"]))
        if not app_dir.exists():
            print(f"[SKIP] {app['name']}: app_dir {app_dir} does not exist", file=sys.stderr)
            continue

        for core_count in core_counts:
            for rep in range(1, reps + 1):
                out_path = DATA_DIR / app["name"] / f"cores_{core_count}" / f"rep_{rep}.csv"
                print(f"[{app['name']}] cores={core_count} rep={rep}/{reps}")

                if args.dry_run:
                    cmd, env_extra = build_command(app, core_count)
                    print("   cmd:", " ".join(cmd), "env:", env_extra)
                    continue

                idle_pct, idle_met = check_idle(core_count, enforce=ENFORCE_IDLE)

                cmd, env_extra = build_command(app, core_count)
                pass_results = {}
                pass_instr_cum = {}
                pass_timestamps = {}
                rss_samples = []
                total_wall = 0.0
                any_fail = False
                min_enabled_pct = None
                # Raw per-pass perf CSVs are written to tmpfs (RAM-backed), not the real
                # disk, purely to avoid adding our own write/fsync/writeback load on top
                # of this machine's already-nonzero background kworker activity during
                # the idle-check window of the *next* trial. Only the final merged CSV
                # (out_path, written below by merge_and_write) persists to real disk.
                tmp_dir = TMP_ROOT / app["name"] / f"cores_{core_count}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                for i, pass_def in enumerate(PASSES):
                    out_csv = tmp_dir / f"rep_{rep}_{pass_def['name']}.csv"
                    uncore_csv = tmp_dir / f"rep_{rep}_mem_bw.csv" if i == 0 else None
                    wall_time, rc, samples = run_pass(
                        cmd, env_extra, str(app_dir), core_count, pass_def, out_csv,
                        sample_rss=(i == 0), uncore_csv=uncore_csv,
                    )
                    total_wall += wall_time
                    any_fail = any_fail or (rc != 0)
                    intervals, pass_min_pct, instr_cum, ts = parse_perf_csv(out_csv)
                    pass_results[pass_def["name"]] = intervals
                    pass_instr_cum[pass_def["name"]] = instr_cum
                    pass_timestamps[pass_def["name"]] = ts
                    if pass_min_pct is not None and (min_enabled_pct is None or pass_min_pct < min_enabled_pct):
                        min_enabled_pct = pass_min_pct
                    if i == 0:
                        rss_samples = samples
                        mem_intervals, _, _, mem_ts = parse_perf_csv(uncore_csv)
                        pass_results["mem_bw"] = mem_intervals
                        pass_timestamps["mem_bw"] = mem_ts

                n_intervals = merge_and_write(
                    app["name"], core_count, rep, pass_results, pass_instr_cum, rss_samples, out_path,
                    method="F", pass_timestamps=pass_timestamps,
                )
                alt_c_path = DATA_DIR_ALT_C / app["name"] / f"cores_{core_count}" / f"rep_{rep}.csv"
                merge_and_write(
                    app["name"], core_count, rep, pass_results, pass_instr_cum, rss_samples, alt_c_path,
                    method="C", pass_timestamps=pass_timestamps,
                )

                multiplex_ok = any_fail is False and (min_enabled_pct is None or min_enabled_pct >= 99.0)
                append_manifest([
                    app["name"], core_count, rep,
                    datetime.now(timezone.utc).isoformat(),
                    f"{total_wall:.3f}", SAMPLE_MS,
                    f"{idle_pct:.1f}" if idle_pct is not None else "",
                    idle_met, ENFORCE_IDLE,
                    f"{min_enabled_pct:.2f}" if min_enabled_pct is not None else "",
                    ("OK" if multiplex_ok else "FLAGGED"),
                    n_intervals, str(out_path),
                ])


if __name__ == "__main__":
    main()
