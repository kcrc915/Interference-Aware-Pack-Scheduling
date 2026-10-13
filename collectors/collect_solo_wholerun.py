#!/usr/bin/env python3
"""Whole-run-only solo collection -- no `perf -I <ms>` interval sampling.

Per the ML prediction document's stated scope ("Whole-run counters suffice
for this version; 100ms traces are not required training inputs") and a
direct request to avoid any risk of interval-sampling overhead perturbing
measurements on new, not-yet-characterized hardware (banco) -- this is a
separate, deliberately simpler script from collect_solo.py, not a flag on
it. Dropping -I removes the entire cross-pass DTW-alignment problem that
dominates collect_solo.py's complexity (there's nothing to align when each
pass produces exactly one number per event instead of a time series), so
this is a genuinely simpler pipeline, not just collect_solo.py with a
branch.

Reuses collect_solo.py's config loading, NUMA-aware core selection, command
building, and idle-check logic directly (imported, not reimplemented) --
only the perf invocation and output parsing differ.

Output: one row per (app, core_count, rep) in <data_dir>/manifest.csv,
with whole-run aggregates matching the ML document's feature definitions
(eq. 3-6): IPC = total_instructions/total_cycles (not a mean of interval
IPCs), MPKI, BW = total_bytes/wall_time, utilization = CPU_time/(p*wall_time).
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_solo as cs  # reuse config/NUMA/command/idle-check logic directly

REPO = cs.REPO
SAMPLE_MS_RSS = 100  # RSS polling is independent of perf's interval mode --
# psutil sampling isn't a perf counter, has negligible overhead either way,
# and we still want a peak/mean RSS number even in whole-run mode.

WHOLE_RUN_PASSES = [
    {
        "name": "core_cache",
        "events": "instructions,cycles,task-clock,LLC-loads,LLC-load-misses,"
                   "page-faults,context-switches,cpu-migrations",
        "metrics": None,
    },
    {"name": "tma_fe_be", "events": "instructions,cycles,task-clock",
     "metrics": "tma_frontend_bound,tma_backend_bound"},
    {"name": "tma_bad_spec", "events": "instructions,cycles,task-clock",
     "metrics": "tma_bad_speculation"},
    {"name": "tma_retiring", "events": "instructions,cycles,task-clock",
     "metrics": "tma_retiring"},
]


def start_wholerun_uncore_companion(out_csv: Path):
    """Whole-run version of collect_solo.start_uncore_companion -- that one
    hardcodes -I <ms> internally (a bug if reused here unchanged: it was
    silently writing interval-format rows with a timestamp column, which
    parse_wholerun_csv doesn't expect, so DRAM bandwidth came out empty --
    caught by actually running this end to end before handing it off, not
    just a syntax check). Verified directly: perf stat without -I, even when
    stopped via SIGINT rather than a natural command exit, still produces
    one clean whole-run-format row per event, no timestamp column."""
    read_ev, write_ev = cs.detect_uncore_imc_events()
    perf_cmd = [
        "perf", "stat", "-a", "-C", cs.UNCORE_READER_CPU, "-x", ",",
        "-e", f"{read_ev},{write_ev}",
        "-o", str(out_csv),
    ]
    return subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_wholerun_csv(path: Path):
    """Parses perf's NO-INTERVAL -x , output: one line per event, no
    timestamp column (unlike collect_solo.py's parse_perf_csv, which groups
    rows by timestamp into intervals). Returns (values: {event_name: float},
    tma: {metric_name: float}, min_enabled_pct)."""
    values, tma = {}, {}
    min_enabled_pct = None
    if not path.exists():
        return values, tma, min_enabled_pct
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or not row[0]:
                continue
            fields = row + [""] * (7 - len(row))
            value, unit, event, _enabled, runpct, metric_val, metric_unit = fields[:7]
            if event:
                try:
                    values[event] = float(value) if value and value != "<not counted>" else None
                except ValueError:
                    values[event] = None
            if metric_unit and metric_val:
                name = metric_unit.strip().lstrip("%").strip()
                if name.startswith("tma_"):
                    try:
                        tma[name] = float(metric_val)
                    except ValueError:
                        pass
            if runpct:
                try:
                    pct = float(runpct)
                    if min_enabled_pct is None or pct < min_enabled_pct:
                        min_enabled_pct = pct
                except ValueError:
                    pass
    return values, tma, min_enabled_pct


def run_wholerun_pass(cmd, env_extra, cwd, core_count, pass_def, out_csv,
                       sample_rss=False, uncore_csv=None):
    """Same structure as collect_solo.run_pass but WITHOUT -I <ms> -- perf
    reports one aggregate value per event for the whole run instead of a
    time series. Reuses the uncore-companion / RSS-sampler machinery from
    collect_solo.py unchanged (neither depends on interval mode)."""
    perf_cmd = ["perf", "stat", "-a", "-C", cs.core_list_str(core_count), "-x", ","]
    perf_cmd += ["-e", pass_def["events"]]
    if pass_def["metrics"]:
        perf_cmd += ["-M", pass_def["metrics"]]
    perf_cmd += ["-o", str(out_csv), "--"] + cmd

    env = os.environ.copy()
    env.update(env_extra)

    uncore_proc = start_wholerun_uncore_companion(uncore_csv) if uncore_csv else None
    if uncore_proc:
        time.sleep(0.1)

    start = time.time()
    proc = subprocess.Popen(perf_cmd, cwd=cwd, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    sampler = None
    if sample_rss and cs.psutil is not None:
        time.sleep(0.05)
        sampler = cs.RssSampler(proc.pid)
        sampler.start()

    _, stderr = proc.communicate()
    wall_time = time.time() - start

    if sampler:
        sampler.stop()
        sampler.join(timeout=1)
    if uncore_proc:
        cs.stop_uncore_companion(uncore_proc)

    if proc.returncode != 0:
        print(f"  [WARN] perf pass '{pass_def['name']}' exit code {proc.returncode}: "
              f"{stderr.decode(errors='replace')[:300]}", file=sys.stderr)

    rss_samples = sampler.samples if sampler else []
    return wall_time, proc.returncode, rss_samples


def append_manifest(data_dir: Path, row, header):
    manifest = data_dir / "manifest.csv"
    new = not manifest.exists()
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


HEADER = [
    "app", "core_count", "rep", "timestamp", "wall_time_s",
    "instructions", "cycles", "ipc", "task_clock_ms",
    "tma_frontend", "tma_backend", "tma_bad_spec", "tma_retiring", "tma_sum",
    "mpki", "llc_miss_rate", "llc_loads", "llc_load_misses",
    "dram_read_gbps", "dram_write_gbps",
    "rss_peak_bytes", "rss_mean_bytes",
    "page_faults", "context_switches", "cpu_migrations",
    "idle_pct", "idle_threshold_met", "min_enabled_pct", "multiplex_status",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", nargs="*", help="subset of app names, default all")
    ap.add_argument("--cores", nargs="*", type=int, help="subset of core counts, default all")
    ap.add_argument("--reps", type=int, help="override repetition count")
    ap.add_argument("--data-dir", type=str, default="data/solo_wholerun",
                     help="output dir, default data/solo_wholerun (deliberately "
                          "distinct from data/solo and data/solo_banco)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if cs.psutil is None:
        print("[WARN] psutil not installed -- rss_peak_bytes/rss_mean_bytes "
              "will be empty.", file=sys.stderr)

    data_dir = REPO / args.data_dir

    cfg = cs.load_config()
    apps = cfg["applications"]
    if args.apps:
        apps = [a for a in apps if a["name"] in args.apps]
    core_counts = args.cores or cfg["core_counts"]
    reps = args.reps or cfg["repetitions"]

    print(f"[INFO] whole-run-only collection (no -I interval sampling), "
          f"output -> {data_dir}", file=sys.stderr)

    for app in apps:
        app_dir = Path(cs.expand(app["app_dir"]))
        if not app_dir.exists():
            print(f"[SKIP] {app['name']}: app_dir {app_dir} does not exist", file=sys.stderr)
            continue

        for core_count in core_counts:
            for rep in range(1, reps + 1):
                print(f"[{app['name']}] cores={core_count} rep={rep}/{reps}")

                if args.dry_run:
                    cmd, env_extra = cs.build_command(app, core_count)
                    print("   cmd:", " ".join(cmd), "env:", env_extra)
                    continue

                idle_pct, idle_met = cs.check_idle(core_count, enforce=cs.ENFORCE_IDLE)
                cmd, env_extra = cs.build_command(app, core_count)

                tmp_dir = cs.TMP_ROOT / app["name"] / f"cores_{core_count}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                pass_values = {}
                pass_tma = {}
                rss_samples = []
                total_wall = 0.0
                min_enabled_pct = None
                any_fail = False

                for i, pass_def in enumerate(WHOLE_RUN_PASSES):
                    out_csv = tmp_dir / f"rep_{rep}_{pass_def['name']}.csv"
                    uncore_csv = tmp_dir / f"rep_{rep}_mem_bw.csv" if i == 0 else None
                    wall_time, rc, samples = run_wholerun_pass(
                        cmd, env_extra, str(app_dir), core_count, pass_def, out_csv,
                        sample_rss=(i == 0), uncore_csv=uncore_csv,
                    )
                    total_wall += wall_time
                    any_fail = any_fail or (rc != 0)
                    values, tma, pct = parse_wholerun_csv(out_csv)
                    pass_values[pass_def["name"]] = values
                    pass_tma[pass_def["name"]] = tma
                    if pct is not None and (min_enabled_pct is None or pct < min_enabled_pct):
                        min_enabled_pct = pct
                    if i == 0:
                        rss_samples = samples
                        wall_time_core = wall_time  # representative T_solo(p)
                        mem_values, _, _ = parse_wholerun_csv(uncore_csv)

                core = pass_values["core_cache"]
                instr = core.get("instructions")
                cycles = core.get("cycles")
                ipc = (instr / cycles) if instr and cycles else None
                llc_loads = core.get("LLC-loads")
                llc_misses = core.get("LLC-load-misses")
                mpki = (llc_misses / instr * 1000) if llc_misses is not None and instr else None
                miss_rate = (llc_misses / llc_loads) if llc_misses is not None and llc_loads else None
                task_clock_ms = core.get("task-clock")

                _read_ev, _write_ev = cs.detect_uncore_imc_events()
                dram_r = mem_values.get(_read_ev)
                dram_w = mem_values.get(_write_ev)
                # dram_r/w are MiB accumulated over the WHOLE run (this companion
                # also has no -I now) -- divide by the whole run's wall time for
                # the document's BW_i(p) = (Dread+Dwrite)/T_i(p), eq. 5
                dram_r_gbps = (dram_r * 1024 * 1024 / wall_time_core / 1e9) if dram_r is not None else None
                dram_w_gbps = (dram_w * 1024 * 1024 / wall_time_core / 1e9) if dram_w is not None else None

                fb, bs, rt = pass_tma["tma_fe_be"], pass_tma["tma_bad_spec"], pass_tma["tma_retiring"]
                tma_vals = [fb.get("tma_frontend_bound"), fb.get("tma_backend_bound"),
                            bs.get("tma_bad_speculation"), rt.get("tma_retiring")]
                tma_sum = sum(v for v in tma_vals if v is not None) if all(
                    v is not None for v in tma_vals) else None

                rss_bytes_list = [s[1] for s in rss_samples] if rss_samples else []
                rss_peak = max(rss_bytes_list) if rss_bytes_list else None
                rss_mean = (sum(rss_bytes_list) / len(rss_bytes_list)) if rss_bytes_list else None

                multiplex_ok = any_fail is False and (min_enabled_pct is None or min_enabled_pct >= 99.0)

                row = [
                    app["name"], core_count, rep, datetime.now(timezone.utc).isoformat(),
                    f"{wall_time_core:.3f}",
                    instr, cycles, ipc, task_clock_ms,
                    tma_vals[0], tma_vals[1], tma_vals[2], tma_vals[3], tma_sum,
                    mpki, miss_rate, llc_loads, llc_misses,
                    dram_r_gbps, dram_w_gbps,
                    rss_peak, rss_mean,
                    core.get("page-faults"), core.get("context-switches"), core.get("cpu-migrations"),
                    f"{idle_pct:.1f}" if idle_pct is not None else "", idle_met,
                    f"{min_enabled_pct:.2f}" if min_enabled_pct is not None else "",
                    ("OK" if multiplex_ok else "FLAGGED"),
                ]
                append_manifest(data_dir, row, HEADER)
                print(f"    wall={wall_time_core:.2f}s ipc={ipc} coverage={min_enabled_pct} "
                      f"status={'OK' if multiplex_ok else 'FLAGGED'}")

    print(f"\nDone. Results in {data_dir}/manifest.csv")


if __name__ == "__main__":
    main()
