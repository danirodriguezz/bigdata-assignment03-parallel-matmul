#!/usr/bin/env python3
"""Run a benchmark command while sampling the CPU clock frequency (experiment `freq_probe`).

Hardware counters are not available on this laptop (kernel.perf_event_paranoid = 4), so the
frequency evidence comes from the kernel's cpufreq interface: every `--interval` seconds the
script reads /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq (on intel_pstate this is the
average effective frequency since the previous read, derived from APERF/MPERF) and /proc/stat
per-CPU busy time. A logical CPU counts as "busy" in an interval if it was >= 90% busy.

The child's JSON lines are passed through unchanged except that the process-level statistics
  freq_busy_mhz_median   median frequency over all (busy CPU, interval) samples
  freq_busy_mhz_p10/p90  10th / 90th percentile of the same samples
  busy_cpus_median       median number of busy logical CPUs per interval
  freq_samples           number of (busy CPU, interval) samples
are added to every line. The sampler sleeps between samples (negligible CPU use), but it is still
an extra process, so these runs are kept separate from the timing experiments.
Usage: freq_probe.py [--interval 0.1] -- <command ...>
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

CPUS = sorted(int(p.name[3:]) for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))


def read_freqs():
    out = {}
    for c in CPUS:
        try:
            out[c] = int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()) / 1000
        except OSError:
            pass
    return out


def read_stat():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:9]))
            idle = v[3] + v[4]
            out[int(f[0][3:])] = (sum(v) - idle, sum(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    samples, busy_counts = [], []
    prev = read_stat()
    read_freqs()  # reset the averaging window of scaling_cur_freq
    while proc.poll() is None:
        time.sleep(args.interval)
        cur, freqs = read_stat(), read_freqs()
        nbusy = 0
        for c, (busy, tot) in cur.items():
            db, dt = busy - prev[c][0], tot - prev[c][1]
            if dt > 0 and db / dt >= 0.9 and c in freqs:
                samples.append(freqs[c])
                nbusy += 1
        busy_counts.append(nbusy)
        prev = cur
    stdout = proc.stdout.read()
    extra = {}
    if samples:
        q = statistics.quantiles(samples, n=10) if len(samples) >= 10 else [min(samples)] * 9
        extra = {"freq_busy_mhz_median": statistics.median(samples), "freq_busy_mhz_p10": q[0],
                 "freq_busy_mhz_p90": q[-1], "busy_cpus_median": statistics.median(busy_counts),
                 "freq_samples": len(samples)}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            rec = json.loads(s)
            rec.update(extra)
            print(json.dumps(rec), flush=True)
        else:
            print(line)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
