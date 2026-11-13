#!/usr/bin/env python3
"""Experiment driver for Assignment 3 (reads configs/experiment.json).

Every timed process goes through run_measured.py (machine-wide lock, peak RSS, timeouts), one
process per configuration and round, so the lock is held only for one configuration at a time.

Interleaving: the full list of configurations is executed `rounds` times; inside each round the
order is shuffled (seeded by round number), so slow drifts (thermal state, background activity of
other agents) are spread over all configurations instead of biasing one of them.

Each process run executes `warmup` untimed + `reps` timed in-process repetitions (per-size values
in the config). Failed configurations (timeout / oom / error) are recorded by run_measured.py as
rows without a time and are not retried in later rounds.

Usage
  python scripts/run_experiments.py --config configs/experiment.json            # everything
  python scripts/run_experiments.py --pilot                                      # smoke test
  python scripts/run_experiments.py --experiments scaling --sizes 3072 --rounds 6 7
  python scripts/run_experiments.py --resume            # skip (config, round) already in results/raw
  python scripts/run_experiments.py --resume --dry-run  # only count what would run
"""
import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_MEASURED = ROOT / ".claude/skills/benchmark-protocol/scripts/run_measured.py"


def jobs_for(exp, cfg, pilot=False):
    """Expand one experiment block of the config into a list of job dicts."""
    jobs = []
    sizes = exp["sizes"] if not pilot else exp.get("pilot_sizes", exp["sizes"][:1])
    threads = exp["threads"] if not pilot else exp.get("pilot_threads", exp["threads"])
    kernel = exp["kernel"]
    for n in sizes:
        if exp.get("baseline", False):
            jobs.append(dict(experiment=exp["name"], method=f"seq_{kernel}", variant="seq",
                             kernel=kernel, n=n, threads=1, rounds=exp.get("rounds")))
        for variant in exp.get("variants", ["omp"]):
            for p in threads:
                method = f"{variant}_{kernel}"
                jobs.append(dict(experiment=exp["name"], method=method, variant=variant,
                                 kernel=kernel, n=n, threads=p, rounds=exp.get("rounds")))
    return jobs


def command(job, cfg, rnd, pilot):
    n, p = job["n"], job["threads"]
    seed = cfg["seed"]
    warm, reps = cfg["inproc_reps"][str(n)]
    if pilot:
        reps = 1
    binary = ROOT / "bin" / f"matmul_{job['variant']}"
    prog = [str(binary), "--n", str(n),
            "--a", str(ROOT / f"data/inputs/n{n}_s{seed}_A.bin"),
            "--b", str(ROOT / f"data/inputs/n{n}_s{seed}_B.bin"),
            "--kernel", job["kernel"], "--bs", str(cfg["bs"]),
            "--warmup", str(warm), "--reps", str(reps), "--round", str(rnd)]
    env = dict(cfg["common_env"])
    if job["variant"] == "seq":
        # pure sequential binary: pin it to the same physical core as OpenMP place 0
        prog = ["taskset", "-c", cfg["seq_cpus"]] + prog
        env["OMP_NUM_THREADS"] = "1"
    else:
        env.update(cfg["affinity"])
        env["OMP_NUM_THREADS"] = str(p)
    if job["experiment"] == "freq_probe":
        prog = [sys.executable, str(ROOT / "scripts/freq_probe.py"), "--"] + prog
    return prog, env


def run_job(job, cfg, rnd, pilot, out_dir):
    prog, env = command(job, cfg, rnd, pilot)
    out = out_dir / (f"pilot_{job['experiment']}.csv" if pilot else f"{job['experiment']}.csv")
    labels = [f"assignment=A3", f"experiment={job['experiment']}", f"method={job['method']}",
              "language=c", f"n={job['n']}", "dtype=float64", f"seed={cfg['seed']}",
              f"threads={job['threads']}"]
    cmd = [sys.executable, str(RUN_MEASURED), "--out", str(out), "--warmup", "0", "--reps", "1",
           "--timeout", str(cfg["budgets"]["timeout_s"]), "--label", *labels,
           "--env", *[f"{k}={v}" for k, v in env.items()], "--", *prog]
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    return rc, time.monotonic() - t0


def done_set(out_dir, pilot):
    """(experiment, method, n, threads, round) of every process run already in the raw CSVs."""
    done = set()
    for f in out_dir.glob("pilot_*.csv" if pilot else "*.csv"):
        if not pilot and f.name.startswith("pilot_"):
            continue
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rnd = json.loads(row["extra_json"]).get("round")
                except (json.JSONDecodeError, KeyError):
                    continue
                done.add((row["experiment"], row["method"], int(row["n"]), int(row["threads"]), rnd))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/experiment.json"))
    ap.add_argument("--pilot", action="store_true", help="1 round, small sizes, 1 rep (pilot_*.csv)")
    ap.add_argument("--experiments", nargs="*", help="subset of experiment names")
    ap.add_argument("--sizes", type=int, nargs="*", help="subset of sizes")
    ap.add_argument("--threads", type=int, nargs="*", help="subset of thread counts")
    ap.add_argument("--rounds", type=int, nargs="*", help="explicit round numbers (default 1..R)")
    ap.add_argument("--resume", action="store_true",
                    help="skip process runs whose (experiment, method, n, threads, round) is already recorded")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = ROOT / "results/raw"
    jobs = []
    for exp in cfg["experiments"]:
        if args.experiments and exp["name"] not in args.experiments:
            continue
        jobs += jobs_for(exp, cfg, args.pilot)
    if args.sizes:
        jobs = [j for j in jobs if j["n"] in args.sizes]
    if args.threads:
        jobs = [j for j in jobs if j["threads"] in args.threads or j["variant"] == "seq"]
    rounds = [1] if args.pilot else (args.rounds or list(range(1, cfg["rounds"] + 1)))

    already = done_set(out_dir, args.pilot) if args.resume else set()
    failed = {}  # (experiment, method, threads) -> smallest n that failed
    total = len(jobs) * len(rounds)
    done = 0
    t_start = time.monotonic()
    busy = 0.0
    for rnd in rounds:
        order = list(jobs)
        random.Random(1000 + rnd).shuffle(order)
        for job in order:
            key = (job["experiment"], job["method"], job["threads"])
            done += 1
            if job["rounds"] and rnd > job["rounds"] and not args.pilot:
                continue  # this experiment uses fewer rounds than the global setting
            if (job["experiment"], job["method"], job["n"], job["threads"], rnd) in already:
                continue  # --resume: this process run is already in results/raw
            if args.dry_run:
                print(f"[dry-run] round {rnd} {job['experiment']} {job['method']} n={job['n']} p={job['threads']}")
                continue
            if key in failed and job["n"] >= failed[key]:
                print(f"[driver] skip {job} (failed earlier at this or a smaller n)", flush=True)
                continue
            rc, dt = run_job(job, cfg, rnd, args.pilot, out_dir)
            busy += dt
            print(f"[driver] round {rnd} {done}/{total} {job['experiment']} {job['method']} "
                  f"n={job['n']} p={job['threads']} rc={rc} {dt:.1f}s "
                  f"(elapsed {(time.monotonic() - t_start) / 60:.1f} min)", flush=True)
            if rc == 3:
                failed[key] = min(job["n"], failed.get(key, job["n"]))
    print(f"[driver] finished: {done} jobs, {(time.monotonic() - t_start) / 60:.1f} min wall "
          f"(includes waiting for the lock)", flush=True)


if __name__ == "__main__":
    main()
