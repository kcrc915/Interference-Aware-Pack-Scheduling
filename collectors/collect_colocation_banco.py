#!/usr/bin/env python3
"""
Banco co-location ground-truth collection -- adapted from collect_colocation.py
(heliodor, 6-core, single NUMA node) for banco (24-core, dual-socket NUMA).

Measures real slowdown each app suffers when co-located with another, same
metric as the heliodor script:

    slowdown_i = T_co_run_i / T_solo_i   (both on the SAME exact physical cores)

No perf instrumentation during timing (clean wall-clock, no counting
overhead), disjoint core split, sustained-pressure relaunching so neither
app gets a head start -- all unchanged from the heliodor design. Two things
are banco-specific:

1. NUMA-aware disjoint core split. Plain contiguous taskset ranges only work
   correctly on a single-NUMA-node machine (heliodor). Banco's 24 cores are
   ordered via collect_solo.core_list(24) (node0 filled first, then node1 --
   node0 is the even cores, node1 the odd cores, confirmed via `numactl
   --hardware`), then split at `--split`: app A gets the first `split` cores
   of that order, app B gets the rest. split=12 gives A all of node0 and B
   all of node1 -- the cleanest possible single-socket-each co-run. Smaller
   or larger splits give one side a NUMA-local subset and the other a set
   that spans both sockets.

2. numactl --physcpubind/--membind instead of plain taskset -c, matching
   collect_solo.py, so memory allocation stays local to whichever node(s)
   each app's cores are actually on.

Solo baselines are ALWAYS freshly measured here, never reused from the
existing solo sweep (data/solo_wholerun_banco_full_*). Reason: under most
splits, the "B" side's co-run core set is the COMPLEMENT of A's cores in the
NUMA fill order, which is generally a DIFFERENT specific set of physical
cores than collect_solo_wholerun's own core_list(core_count) would assign
that same app if profiled solo in isolation at that core count. Reusing the
recorded wall time at the matching core COUNT would silently compare
performance on different physical cores (different NUMA-locality mix),
defeating the point of a fair T_co/T_solo ratio. Measuring fresh, on the
exact core assignment the co-run itself will use, side-steps this entirely.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_solo as cs  # reuse NUMA topology discovery + config loading

REPO = cs.REPO
CONFIG_DIR = REPO / "config"
OUT_DIR = REPO / "data" / "colocation_banco"
TOTAL_CORES = 24
DEFAULT_SPLIT = 12   # symmetric: 12 cores each, one NUMA node per app


def load_apps():
    return {a["name"]: a for a in cs.load_config()["applications"]}


def load_pairs():
    with open(CONFIG_DIR / "colocation_pairs_banco.yaml") as f:
        return yaml.safe_load(f)


def numa_nodes_for_cores(cores: list):
    """Generalized version of collect_solo.numa_nodes_for -- takes an
    explicit core list (not necessarily core_list(N)'s first-N) and returns
    which NUMA node IDs it touches, for --membind."""
    topology = cs.discover_numa_topology()
    cores_set = set(cores)
    touched = [i for i, node_cpus in enumerate(topology) if cores_set & set(node_cpus)]
    return touched or [0]


def split_cores(split: int):
    """The full 24-core NUMA fill order, sliced at `split`. App A gets the
    first `split` cores, app B gets the rest -- disjoint, covering all 24
    cores exactly once. See module docstring for why this (not independent
    core_list(split) / core_list(24-split) calls) is the correct way to
    assign co-run cores."""
    full_order = cs.core_list(TOTAL_CORES)
    return full_order[:split], full_order[split:]


def build_cmd_for_cores(app: dict, cores: list):
    """Same shape as collect_solo.build_command, but pinned to an explicit
    core list rather than always core_list(core_count) from scratch --
    needed because app B's co-run cores are the complement of A's, not the
    first N in NUMA order."""
    core_count = len(cores)
    args_str = (app.get("args") or "").replace("{threads}", str(core_count))
    if "{npx}" in args_str or "{npy}" in args_str or "{npz}" in args_str:
        npx, npy, npz = cs.factorize_3d(core_count)
        args_str = (args_str.replace("{npx}", str(npx))
                            .replace("{npy}", str(npy))
                            .replace("{npz}", str(npz)))
    args = args_str.split() if args_str.strip() else []
    cores_str = ",".join(str(c) for c in cores)
    mem_nodes = ",".join(str(n) for n in numa_nodes_for_cores(cores))
    numa_prefix = ["numactl", f"--physcpubind={cores_str}", f"--membind={mem_nodes}"]

    if app["parallel_mode"] == "mpi":
        cmd = numa_prefix + ["mpirun", "-np", str(core_count), app["binary"]] + args
        env_extra = {"OMP_NUM_THREADS": "1"}
    else:
        cmd = numa_prefix + [app["binary"]] + args
        env_extra = {"OMP_NUM_THREADS": str(core_count),
                     "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    return cmd, env_extra, cs.expand(app["app_dir"])


def launch(app, cores):
    cmd, env_extra, cwd = build_cmd_for_cores(app, cores)
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), time.time()


def measure_solo(app, cores, reps=3):
    """Clean wall-clock baseline: no perf, SAME exact cores the co-run will use."""
    times = []
    for _ in range(reps):
        p, t0 = launch(app, cores)
        p.wait()
        times.append(time.time() - t0)
    times.sort()
    return times[len(times) // 2], times


def run_pair(app_a, app_b, cores_a, cores_b, timeout_factor=6.0, solo_hint=None):
    """Co-run A and B on disjoint cores. Returns (t_a, t_b, relaunches_a,
    relaunches_b) -- each app's first fully-co-located completion. The app
    that finishes first is relaunched so the other never runs unopposed
    (sustained pressure), matching the heliodor script exactly."""
    pa, ta0 = launch(app_a, cores_a)
    pb, tb0 = launch(app_b, cores_b)

    t_a = t_b = None
    relaunches_a = relaunches_b = 0
    fillers = []
    deadline = time.time() + timeout_factor * max(solo_hint or (600, 600))
    try:
        while (t_a is None or t_b is None) and time.time() < deadline:
            time.sleep(0.05)
            if pa.poll() is not None:
                if t_a is None:
                    t_a = time.time() - ta0
                if t_b is None:
                    pa, _ = launch(app_a, cores_a)
                    fillers.append(pa)
                    relaunches_a += 1
            if pb.poll() is not None:
                if t_b is None:
                    t_b = time.time() - tb0
                if t_a is None:
                    pb, _ = launch(app_b, cores_b)
                    fillers.append(pb)
                    relaunches_b += 1
    finally:
        for p in [pa, pb] + fillers:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill()
    return t_a, t_b, relaunches_a, relaunches_b


def idle_ok():
    out = subprocess.run(["mpstat", "1", "1"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = line.split()
        if p and p[0] == "Average:" and "%idle" not in p:
            try: return float(p[-1])
            except ValueError: pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--split", type=int, default=DEFAULT_SPLIT,
                    help="cores given to app A; app B gets TOTAL_CORES - split")
    ap.add_argument("--pairs", nargs="*", help="limit to pairs named A+B")
    ap.add_argument("--solo-reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo", action="store_true",
                    help="re-run pairs already present in the manifest "
                         "(default: skip them, so the script is resumable)")
    args = ap.parse_args()

    apps = load_apps()
    pairs = [p for p in load_pairs()["pairs"]
             if not args.pairs or f"{p['a']}+{p['b']}" in args.pairs]

    if not args.redo and (OUT_DIR / "manifest.csv").exists():
        done = {(r["app_a"], r["app_b"], r["core_split_a"])
                for r in csv.DictReader(open(OUT_DIR / "manifest.csv"))}
        before = len(pairs)
        pairs = [p for p in pairs if (p["a"], p["b"], str(args.split)) not in done]
        if before != len(pairs):
            print(f"Resuming: {before - len(pairs)} pair(s) already collected, "
                  f"{len(pairs)} remaining. Use --redo to force.", flush=True)

    cores_a, cores_b = split_cores(args.split)

    if args.dry_run:
        print(f"split={args.split}: A gets cores {cores_a}  B gets cores {cores_b}\n")
        for p in pairs:
            a, b = apps[p["a"]], apps[p["b"]]
            ca, _, _ = build_cmd_for_cores(a, cores_a)
            cb, _, _ = build_cmd_for_cores(b, cores_b)
            print(f"[{p['a']}+{p['b']}] ({p.get('rationale','')})")
            print("   A:", " ".join(ca))
            print("   B:", " ".join(cb))
        print(f"\n{len(pairs)} pairs x {args.reps} reps")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    needed = sorted({(p["a"], tuple(cores_a)) for p in pairs} |
                     {(p["b"], tuple(cores_b)) for p in pairs})
    solo = {}
    print(f"Measuring clean solo baselines for {len(needed)} (app, core-set) combos "
          f"({args.solo_reps} reps each, no perf)...", flush=True)
    for name, cores in needed:
        t, allt = measure_solo(apps[name], list(cores), args.solo_reps)
        solo[(name, cores)] = t
        print(f"  {name:12s} @ cores={list(cores)}  T_solo = {t:8.2f}s   (runs: "
              f"{', '.join(f'{x:.1f}' for x in allt)})", flush=True)
    with open(OUT_DIR / "solo_baselines.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["app", "cores", "t_solo_s"])
        for (k, cores), v in sorted(solo.items()):
            w.writerow([k, ",".join(str(c) for c in cores), f"{v:.4f}"])

    man = OUT_DIR / "manifest.csv"
    new = not man.exists()
    mf = open(man, "a", newline=""); mw = csv.writer(mf)
    if new:
        mw.writerow(["app_a", "app_b", "core_split_a", "core_split_b",
                     "cores_a", "cores_b", "rep", "timestamp",
                     "idle_pct_before", "t_solo_a", "t_co_a", "slowdown_a",
                     "t_solo_b", "t_co_b", "slowdown_b"])

    for p in pairs:
        A, B = p["a"], p["b"]
        solo_a = solo[(A, tuple(cores_a))]
        solo_b = solo[(B, tuple(cores_b))]
        d = OUT_DIR / f"pair_{A}_{B}_s{args.split}"; d.mkdir(parents=True, exist_ok=True)
        for rep in range(1, args.reps + 1):
            idle = idle_ok()
            print(f"[{A}+{B}] split={args.split} rep {rep}/{args.reps} (idle={idle})", flush=True)
            ta, tb, _ra, _rb = run_pair(apps[A], apps[B], cores_a, cores_b,
                                         solo_hint=(solo_a, solo_b))
            sa = ta / solo_a if ta and solo_a else None
            sb = tb / solo_b if tb and solo_b else None
            print(f"    {A}: {ta:.2f}s -> slowdown {sa:.3f}   "
                  f"{B}: {tb:.2f}s -> slowdown {sb:.3f}" if ta and tb
                  else f"    INCOMPLETE ta={ta} tb={tb}", flush=True)
            row = [A, B, args.split, TOTAL_CORES - args.split,
                   ";".join(map(str, cores_a)), ";".join(map(str, cores_b)),
                   rep, datetime.now(timezone.utc).isoformat(),
                   f"{idle:.1f}" if idle else "",
                   f"{solo_a:.4f}", f"{ta:.4f}" if ta else "", f"{sa:.4f}" if sa else "",
                   f"{solo_b:.4f}", f"{tb:.4f}" if tb else "", f"{sb:.4f}" if sb else ""]
            mw.writerow(row); mf.flush()
            with open(d / f"rep_{rep}.csv", "w", newline="") as rf:
                w = csv.writer(rf)
                w.writerow(["app_a", "app_b", "core_split_a", "core_split_b", "rep",
                            "t_solo_a", "t_co_a", "slowdown_a",
                            "t_solo_b", "t_co_b", "slowdown_b"])
                w.writerow([A, B, args.split, TOTAL_CORES - args.split, rep,
                            f"{solo_a:.4f}", f"{ta:.4f}" if ta else "", f"{sa:.4f}" if sa else "",
                            f"{solo_b:.4f}", f"{tb:.4f}" if tb else "", f"{sb:.4f}" if sb else ""])
    mf.close()
    print("\nDone. Results in data/colocation_banco/ (manifest.csv + solo_baselines.csv).")


if __name__ == "__main__":
    main()
