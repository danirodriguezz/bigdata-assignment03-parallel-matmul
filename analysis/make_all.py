#!/usr/bin/env python3
"""Regenerate every table, figure and number of the report from results/raw/ (make analysis).

Inputs : results/raw/{scaling,scaling_ikj,vectorisation,freq_probe}.csv  (run_measured.py schema)
Outputs: results/summary/*.csv, report/figures/*.pdf|png, report/generated/*.tex

Statistics (stated in the report):
  * T_seq(n) and T_p(n) are MEDIANS of the measured (non-warm-up, status ok) repetitions;
    variability = interquartile range (IQR); k = number of measured repetitions.
  * S(p) = T_seq / T_p, E(p) = S(p) / p; 95% CI of S by bootstrap of both samples (2000 resamples).
  * Overhead ratio T_par(1) / T_seq with the same bootstrap.
  * Exclusion rule: none for the main results (all rows kept). Sensitivity analysis: the same
    speedups recomputed without repetitions with other_busy_cpus > 1.0 (interference by other
    processes on the shared laptop).
Usage: python analysis/make_all.py [--prefix pilot_]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import plot_style as ps  # noqa: E402
from summarize import summarise, limits  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after plot_style selected the Agg backend)

RAW, SUM = ROOT / "results/raw", ROOT / "results/summary"
FIG, GEN = ROOT / "report/figures", ROOT / "report/generated"
PHYS, LOGICAL = 6, 12
CONTAMINATION = 1.0  # other_busy_cpus threshold of the sensitivity analysis
RNG = np.random.default_rng(12345)
EXPERIMENTS = ["scaling", "scaling_ikj", "vectorisation", "freq_probe"]
N_COLORS = {}


# ----------------------------------------------------------------------------------- loading
def load(prefix):
    frames = []
    for exp in EXPERIMENTS:
        p = RAW / f"{prefix}{exp}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, dtype=str, keep_default_na=False))
    if not frames:
        raise SystemExit(f"no raw files with prefix {prefix!r} in {RAW}")
    df = pd.concat(frames, ignore_index=True)
    for c in ("n", "threads", "time_s", "process_wall_s", "peak_rss_kib", "memory_bytes", "is_warmup",
              "loadavg_1m", "checksum"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    extra = df["extra_json"].map(lambda s: json.loads(s) if s else {})
    for key in ("kernel", "variant", "threads_actual", "cpus", "gflops", "cpu_util", "other_busy_cpus",
                "round", "bs", "freq_busy_mhz_median", "freq_busy_mhz_p10", "freq_busy_mhz_p90",
                "busy_cpus_median", "build_flags"):
        df[key] = extra.map(lambda d: d.get(key))
    for c in ("threads_actual", "gflops", "cpu_util", "other_busy_cpus", "round", "bs",
              "freq_busy_mhz_median", "freq_busy_mhz_p10", "freq_busy_mhz_p90", "busy_cpus_median"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def measured(df):
    return df[(df["is_warmup"] == 0) & (df["status"] == "ok")]


# ----------------------------------------------------------------------------------- statistics
def boot_ratio(a, b, reps=2000):
    """95% bootstrap CI of median(a) / median(b)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    ra = np.median(RNG.choice(a, size=(reps, len(a))), axis=1)
    rb = np.median(RNG.choice(b, size=(reps, len(b))), axis=1)
    r = ra / rb
    return float(np.quantile(r, .025)), float(np.quantile(r, .975))


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else np.nan


def speedups(m, experiment, kernel, seq_method, par_method):
    """S(p), E(p) per n from the measured rows m."""
    rows = []
    sub = m[m["experiment"] == experiment]
    for n, g in sub.groupby("n"):
        tb = g.loc[g["method"] == seq_method, "time_s"].to_numpy()
        if len(tb) == 0:
            continue
        for p, gp in g[g["method"] == par_method].groupby("threads"):
            tp = gp["time_s"].to_numpy()
            s = np.median(tb) / np.median(tp)
            lo, hi = boot_ratio(tb, tp)
            rows.append({"experiment": experiment, "kernel": kernel, "n": int(n), "threads": int(p),
                         "T_seq_median_s": np.median(tb), "T_seq_iqr_s": q(tb, .75) - q(tb, .25),
                         "T_p_median_s": np.median(tp), "T_p_q1_s": q(tp, .25), "T_p_q3_s": q(tp, .75),
                         "T_p_iqr_s": q(tp, .75) - q(tp, .25), "k_seq": len(tb), "k_p": len(tp),
                         "speedup": s, "speedup_ci_low": lo, "speedup_ci_high": hi,
                         "efficiency": s / p, "gflops_p": 2 * n ** 3 / np.median(tp) / 1e9,
                         "region": region(p)})
    return pd.DataFrame(rows)


def region(p):
    return "physical" if p <= PHYS else ("smt" if p <= LOGICAL else "oversubscribed")


# ----------------------------------------------------------------------------------- latex helpers
def fmt_s(x):
    if pd.isna(x):
        return "--"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    if x >= 0.1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def fmt_ms(x):
    return "--" if pd.isna(x) else (f"{x * 1e3:.2f}" if x < 1 else f"{x * 1e3:.0f}")


class Numbers:
    """Collects \\newcommand macros for report/generated/numbers.tex."""

    def __init__(self):
        self.items = {}

    def add(self, name, value, comment=""):
        assert name.isalpha(), name
        self.items[name] = (value, comment)

    def write(self, path):
        lines = ["% Generated by analysis/make_all.py -- do not edit by hand."]
        for k, (v, c) in self.items.items():
            lines.append(f"\\newcommand{{\\{k}}}{{{v}}}" + (f"  % {c}" if c else ""))
        path.write_text("\n".join(lines) + "\n")


def word(i):
    return {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 8: "Eight", 10: "Ten",
            12: "Twelve", 16: "Sixteen", 24: "TwentyFour", 256: "TwoFiftySix", 512: "FiveTwelve",
            1024: "TenTwentyFour", 2048: "TwentyFortyEight", 3072: "ThirtySeventyTwo",
            4096: "FortyNinetySix"}[i]


# ----------------------------------------------------------------------------------- figures
def shade_regions(ax, pmax):
    ax.axvspan(PHYS + 0.5, LOGICAL + 0.5, color="0.85", alpha=0.5, lw=0)
    ax.axvspan(LOGICAL + 0.5, pmax + 2, color="0.7", alpha=0.35, lw=0, hatch="///", fill=False)
    ax.axvline(PHYS + 0.5, color="0.5", lw=0.6, ls=":")
    ax.axvline(LOGICAL + 0.5, color="0.5", lw=0.6, ls=":")


def color_n(n):
    pal = {256: "#56B4E9", 512: "#009E73", 1024: "#E69F00", 2048: "#D55E00", 3072: "#0072B2",
           4096: "#CC79A7"}
    mk = {256: "v", 512: "^", 1024: "s", 2048: "D", 3072: "o", 4096: "P"}
    return pal.get(n, "#000000"), mk.get(n, "x")


def fig_speedup_eff(sp, path):
    ps.apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.5))
    pmax = sp["threads"].max()
    for ax in (a1, a2):
        shade_regions(ax, pmax)
    xs = np.arange(1, pmax + 1)
    a1.plot(xs[xs <= LOGICAL], xs[xs <= LOGICAL], color="0.4", lw=0.8, ls="--", label="ideal $S=p$")
    a1.axhline(PHYS, color="0.4", lw=0.6, ls="-.", label="6 physical cores")
    for n, g in sp.groupby("n"):
        c, m = color_n(n)
        g = g.sort_values("threads")
        yerr = [g["speedup"] - g["speedup_ci_low"], g["speedup_ci_high"] - g["speedup"]]
        a1.errorbar(g["threads"], g["speedup"], yerr=yerr, color=c, marker=m, label=f"$n={n}$", ms=3.5)
        a2.plot(g["threads"], g["efficiency"], color=c, marker=m, label=f"$n={n}$", ms=3.5)
    a2.axhline(1.0, color="0.4", lw=0.8, ls="--")
    for ax in (a1, a2):
        ax.set_xlabel("Threads $p$")
        ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24])
        ax.set_xlim(0.5, pmax + 1)
    a1.set_ylabel("Speedup $S(p)=T_{seq}/T_p$")
    a2.set_ylabel("Efficiency $E(p)=S(p)/p$")
    a1.set_ylim(0, max(7.5, sp["speedup_ci_high"].max() * 1.08))
    a2.set_ylim(0, 1.15)
    a1.text(9, 0.3, "SMT", ha="center", fontsize=7, color="0.3")
    a1.text(19, 0.3, "oversubscribed", ha="center", fontsize=7, color="0.3")
    a1.legend(loc="upper left", ncol=2, fontsize=6.5)
    a1.set_title("(a) speedup", fontsize=8)
    a2.set_title("(b) efficiency", fontsize=8)
    fig.tight_layout()
    ps.savefig(fig, path)


def fig_runtime(m, sp, path):
    fig, ax = ps.new_figure("column", 2.6)
    pmax = sp["threads"].max()
    shade_regions(ax, pmax)
    for n, g in sp.groupby("n"):
        c, mk = color_n(n)
        g = g.sort_values("threads")
        ax.errorbar(g["threads"], g["T_p_median_s"],
                    yerr=[g["T_p_median_s"] - g["T_p_q1_s"], g["T_p_q3_s"] - g["T_p_median_s"]],
                    color=c, marker=mk, ms=3.5, label=f"$n={n}$")
        ax.scatter([0.75], [g["T_seq_median_s"].iloc[0]], color=c, marker="*", s=22, zorder=5)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 16, 24])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(0.65, pmax * 1.15)
    ax.set_xlabel("Threads $p$ (log scale); $\\star$ = sequential baseline")
    ax.set_ylabel("Median kernel time [s]")
    ax.legend(fontsize=6.5, ncol=3, loc="lower left")
    ps.savefig(fig, path)


def fig_kernels(sp_all, path):
    fig, ax = ps.new_figure("column", 2.5)
    pmax = sp_all["threads"].max()
    shade_regions(ax, pmax)
    styles = {"blocked": "-", "ikj": "--"}
    for (kern, n), g in sp_all.groupby(["kernel", "n"]):
        if n not in (1024, 2048):
            continue
        c, mk = color_n(n)
        g = g.sort_values("threads")
        ax.plot(g["threads"], g["gflops_p"], ls=styles[kern], color=c, marker=mk, ms=3.2,
                mfc="white" if kern == "ikj" else c, label=f"{kern}, $n={n}$")
    ax.set_xlabel("Threads $p$")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24])
    ax.set_xlim(0.5, pmax + 1)
    ax.set_ylabel("Throughput $2n^3/T_p$ [GFLOP/s]")
    ax.legend(fontsize=6.5, loc="upper left")
    ps.savefig(fig, path)


def fig_vec(vt, path):
    fig, axes = plt.subplots(1, 2, figsize=(ps.COLUMN_WIDTH_IN, 2.3), sharey=False)
    ps.apply_style()
    variants = ["omp_scalar", "omp_avx2", "omp_fma"]
    labels = {"omp_scalar": "scalar", "omp_avx2": "AVX2", "omp_fma": "AVX2+FMA"}
    cols = {"omp_scalar": "#999999", "omp_avx2": "#0072B2", "omp_fma": "#D55E00"}
    for ax, n in zip(axes, sorted(vt["n"].unique())):
        g = vt[vt["n"] == n]
        for i, p in enumerate(sorted(g["threads"].unique())):
            for j, v in enumerate(variants):
                r = g[(g["threads"] == p) & (g["variant"] == v)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i * 4 + j
                ax.bar(x, r["gflops"], color=cols[v], width=0.85, label=labels[v] if i == 0 else None)
                ax.errorbar(x, r["gflops"], yerr=[[r["gflops"] - r["gflops_q3"]], [r["gflops_q1"] - r["gflops"]]],
                            color="k", lw=0.7, capsize=1.5)
        ax.set_xticks([1, 5])
        ax.set_xticklabels([f"p={p}" for p in sorted(g["threads"].unique())])
        ax.set_title(f"$n={n}$", fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Throughput [GFLOP/s]")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    ps.savefig(fig, path)


def fig_freq(fq, path):
    fig, ax = ps.new_figure("column", 2.2)
    g = fq.sort_values("threads")
    yerr = [(g["mhz"] - g["p10"]).clip(lower=0), (g["p90"] - g["mhz"]).clip(lower=0)]
    ax.errorbar(g["threads"], g["mhz"], yerr=yerr,
                marker="o", color="#0072B2", ms=3.5, label="busy-CPU frequency (median, p10–p90)")
    ax.set_xlabel("Threads $p$ (blocked kernel, $n=2048$)")
    ax.set_ylabel("Effective frequency [MHz]")
    ax.set_xticks(sorted(g["threads"].unique()))
    ax.legend(fontsize=6.5, loc="lower left")
    ps.savefig(fig, path)


def fig_memory(sm, path):
    fig, ax = ps.new_figure("column", 2.2)
    for n, g in sm.groupby("n"):
        c, mk = color_n(n)
        g = g.sort_values("threads")
        ax.plot(g["threads"], g["median_peak_rss_mib"], color=c, marker=mk, ms=3.2, label=f"$n={n}$")
        ax.axhline(3 * 8 * n * n / 2 ** 20, color=c, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Threads $p$ (0 = sequential baseline)")
    ax.set_ylabel("Peak RSS [MiB] (GNU time)")
    ax.set_xticks([0, 1, 2, 4, 6, 8, 12, 16, 24])
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    ps.savefig(fig, path)


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="", help="raw file prefix (e.g. pilot_)")
    args = ap.parse_args()
    for d in (SUM, FIG, GEN):
        d.mkdir(parents=True, exist_ok=True)
    num = Numbers()
    df = load(args.prefix)
    m = measured(df)
    timed = m[m["experiment"] != "freq_probe"]

    # ---- sanity checks reported in the text --------------------------------------------------
    omp = timed[timed["method"].str.startswith("omp")]
    mismatch = int((omp["threads_actual"] != omp["threads"]).sum())
    num.add("NumRowsTotal", len(df[df["experiment"] != "freq_probe"]), "all timing rows incl. warm-ups")
    num.add("NumRowsMeasured", len(timed), "measured ok rows of the timing experiments")
    num.add("NumWarmupRows", int((df["is_warmup"] == 1).sum()))
    num.add("NumFailedRows", int((df["status"] != "ok").sum()))
    num.add("NumThreadMismatch", mismatch, "rows where the real team size != requested p")
    # bit-identity: one checksum per (n) for all scalar builds of the blocked/ikj kernels
    chk = timed[timed["method"].isin(["seq_blocked", "omp_blocked", "omp_scalar_blocked", "omp_avx2_blocked",
                                       "seq_ikj", "omp_ikj"])]
    distinct = chk.groupby("n")["checksum"].nunique()
    num.add("ChecksumDistinctMax", int(distinct.max()) if len(distinct) else 0,
            "distinct checksums per n across seq/omp/scalar/avx2 builds and all p")
    num.add("NumProcessRuns", df[df["experiment"] != "freq_probe"]["run_id"].nunique())
    num.add("MachineMinutes", f"{df.drop_duplicates('run_id')['process_wall_s'].sum() / 60:.0f}",
            "sum of process wall times of all experiments")
    obc = timed["other_busy_cpus"]
    long_reps = timed[timed["time_s"] >= 0.5]
    num.add("OtherBusyMedian", f"{long_reps['other_busy_cpus'].median():.2f}")
    num.add("OtherBusyPninety", f"{long_reps['other_busy_cpus'].quantile(.9):.2f}")
    contaminated = long_reps[long_reps["other_busy_cpus"] > CONTAMINATION]
    num.add("NumContaminated", len(contaminated))
    num.add("NumLongReps", len(long_reps))
    num.add("ContaminationThreshold", f"{CONTAMINATION:.1f}")
    num.add("LoadMax", f"{df['loadavg_1m'].max():.1f}")

    # ---- per-configuration summary --------------------------------------------------------------
    cfg_cols = ["experiment", "method", "language", "n", "dtype", "threads", "timer_boundary"]
    dfx = df[df["experiment"] != "freq_probe"].copy()
    dfx["threads"] = dfx["threads"].astype("Int64")
    summary = summarise(dfx, cfg_cols).sort_values(["experiment", "method", "n", "threads"])
    summary.to_csv(SUM / "summary.csv", index=False)
    limits(summary, cfg_cols).to_csv(SUM / "limits.csv", index=False)

    # ---- speedup / efficiency -------------------------------------------------------------------
    sp = speedups(timed, "scaling", "blocked", "seq_blocked", "omp_blocked")
    sp_ikj = speedups(timed, "scaling_ikj", "ikj", "seq_ikj", "omp_ikj")
    sp_all = pd.concat([sp, sp_ikj], ignore_index=True)
    sp_all.to_csv(SUM / "speedup.csv", index=False)

    # sensitivity: drop contaminated reps
    clean = timed[~((timed["time_s"] >= 0.5) & (timed["other_busy_cpus"] > CONTAMINATION))]
    sp_clean = speedups(clean, "scaling", "blocked", "seq_blocked", "omp_blocked")
    if len(sp_clean):
        cmp_ = sp.merge(sp_clean, on=["n", "threads"], suffixes=("", "_clean"))
        cmp_["rel_diff"] = cmp_["speedup_clean"] / cmp_["speedup"] - 1
        cmp_[["n", "threads", "speedup", "speedup_clean", "k_p", "k_p_clean", "rel_diff"]].to_csv(
            SUM / "sensitivity_contamination.csv", index=False)
        num.add("SensMaxRelDiffPct", f"{100 * cmp_['rel_diff'].abs().max():.1f}",
                "max |relative change| of S(p) when contaminated reps are excluded")

    nmax = int(sp["n"].max())
    num.add("NLargest", nmax)
    big = sp[sp["n"] == nmax].set_index("threads")
    for p in big.index:
        w = word(int(p))
        num.add(f"SLarge{w}", f"{big.loc[p, 'speedup']:.2f}")
        num.add(f"SLargeLo{w}", f"{big.loc[p, 'speedup_ci_low']:.2f}")
        num.add(f"SLargeHi{w}", f"{big.loc[p, 'speedup_ci_high']:.2f}")
        num.add(f"ELarge{w}", f"{big.loc[p, 'efficiency']:.2f}")
        num.add(f"TLarge{w}", fmt_s(big.loc[p, "T_p_median_s"]))
        num.add(f"TLargeIqr{w}", fmt_s(big.loc[p, "T_p_iqr_s"]))
    num.add("TseqLarge", fmt_s(big["T_seq_median_s"].iloc[0]))
    num.add("TseqLargeIqr", fmt_s(big["T_seq_iqr_s"].iloc[0]))
    num.add("KseqLarge", int(big["k_seq"].iloc[0]))
    num.add("KpLarge", int(big["k_p"].min()))
    pbest = int(big["speedup"].idxmax())
    num.add("PBestLarge", pbest)
    num.add("SBestLarge", f"{big['speedup'].max():.2f}")
    for n, g in sp.groupby("n"):
        g = g.set_index("threads")
        w = word(int(n))
        for p in (2, 6, 12, 24):
            if p in g.index:
                num.add(f"S{word(p)}N{w}", f"{g.loc[p, 'speedup']:.2f}")
                num.add(f"E{word(p)}N{w}", f"{g.loc[p, 'efficiency']:.2f}")
        num.add(f"SBestN{w}", f"{g['speedup'].max():.2f}")
        num.add(f"PBestN{w}", int(g["speedup"].idxmax()))
        num.add(f"TseqN{w}", fmt_s(g["T_seq_median_s"].iloc[0]))
    for n, g in sp_ikj.groupby("n"):
        g = g.set_index("threads")
        w = word(int(n))
        for p in (6, 12):
            if p in g.index:
                num.add(f"SIkj{word(p)}N{w}", f"{g.loc[p, 'speedup']:.2f}")
        num.add(f"SIkjBestN{w}", f"{g['speedup'].max():.2f}")
        num.add(f"PIkjBestN{w}", int(g["speedup"].idxmax()))
        num.add(f"GflopsIkjBestN{w}", f"{g['gflops_p'].max():.1f}")
        num.add(f"TseqIkjN{w}", fmt_s(g["T_seq_median_s"].iloc[0]))
        gb = sp[sp["n"] == n].set_index("threads")
        if len(gb):
            num.add(f"GflopsBlockedBestN{w}", f"{gb['gflops_p'].max():.1f}")
            num.add(f"GflopsBlockedOneN{w}", f"{gb.loc[1, 'gflops_p']:.2f}")
            num.add(f"GflopsIkjOneN{w}", f"{g.loc[1, 'gflops_p']:.2f}")

    # ---- overhead of the parallel binary with one thread ---------------------------------------
    ov_rows = []
    for (exp, kern), (sm, pm) in {("scaling", "blocked"): ("seq_blocked", "omp_blocked"),
                                  ("scaling_ikj", "ikj"): ("seq_ikj", "omp_ikj")}.items():
        sub = timed[timed["experiment"] == exp]
        for n, g in sub.groupby("n"):
            tb = g.loc[g["method"] == sm, "time_s"].to_numpy()
            t1 = g.loc[(g["method"] == pm) & (g["threads"] == 1), "time_s"].to_numpy()
            if len(tb) and len(t1):
                lo, hi = boot_ratio(t1, tb)
                ov_rows.append({"kernel": kern, "n": int(n), "T_seq_median_s": np.median(tb),
                                "T_seq_iqr_s": q(tb, .75) - q(tb, .25), "T_par1_median_s": np.median(t1),
                                "T_par1_iqr_s": q(t1, .75) - q(t1, .25), "k_seq": len(tb), "k_par1": len(t1),
                                "ratio": np.median(t1) / np.median(tb), "ci_low": lo, "ci_high": hi})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(SUM / "overhead_one_thread.csv", index=False)
    ovb = ov[ov["kernel"] == "blocked"]
    num.add("OverheadMinPct", f"{100 * (ovb['ratio'].min() - 1):+.1f}")
    num.add("OverheadMaxPct", f"{100 * (ovb['ratio'].max() - 1):+.1f}")
    r = ovb[ovb["n"] == nmax].iloc[0]
    num.add("OverheadLarge", f"{r['ratio']:.3f}")
    num.add("OverheadLargeLo", f"{r['ci_low']:.3f}")
    num.add("OverheadLargeHi", f"{r['ci_high']:.3f}")

    # ---- vectorisation --------------------------------------------------------------------------
    vec = timed[timed["experiment"] == "vectorisation"].copy()
    vt = pd.DataFrame()
    if len(vec):
        vec["variant"] = vec["method"].str.replace("_blocked", "", regex=False)
        vrows = []
        for (n, p), g in vec.groupby(["n", "threads"]):
            base = g.loc[g["variant"] == "omp_scalar", "time_s"].to_numpy()
            for v, gv in g.groupby("variant"):
                t = gv["time_s"].to_numpy()
                lo, hi = boot_ratio(base, t) if len(base) else (np.nan, np.nan)
                vrows.append({"n": int(n), "threads": int(p), "variant": v, "median_s": np.median(t),
                              "q1_s": q(t, .25), "q3_s": q(t, .75), "iqr_s": q(t, .75) - q(t, .25), "k": len(t),
                              "gflops": 2 * n ** 3 / np.median(t) / 1e9,
                              "gflops_q1": 2 * n ** 3 / q(t, .25) / 1e9, "gflops_q3": 2 * n ** 3 / q(t, .75) / 1e9,
                              "speedup_vs_scalar": np.median(base) / np.median(t) if len(base) else np.nan,
                              "ci_low": lo, "ci_high": hi})
        vt = pd.DataFrame(vrows)
        # thread speedup of each variant (p=6 vs p=1, same binary): separates SIMD from threads
        for (n, v), g in vt.groupby(["n", "variant"]):
            g = g.set_index("threads")
            if 1 in g.index and 6 in g.index:
                vt.loc[(vt["n"] == n) & (vt["variant"] == v), "thread_speedup_6_vs_1"] = \
                    g.loc[1, "median_s"] / g.loc[6, "median_s"]
        vt.to_csv(SUM / "vectorisation.csv", index=False)
        for _, rr in vt.iterrows():
            tag = {"omp_scalar": "Scalar", "omp_avx2": "Avx", "omp_fma": "Fma"}[rr["variant"]]
            w = f"{tag}P{word(int(rr['threads']))}N{word(int(rr['n']))}"
            num.add(f"Vec{w}", f"{rr['speedup_vs_scalar']:.2f}")
            num.add(f"VecLo{w}", f"{rr['ci_low']:.2f}")
            num.add(f"VecHi{w}", f"{rr['ci_high']:.2f}")
            num.add(f"VecGflops{w}", f"{rr['gflops']:.1f}")
            if not pd.isna(rr.get("thread_speedup_6_vs_1", np.nan)) and rr["threads"] == 6:
                num.add(f"VecThr{tag}N{word(int(rr['n']))}", f"{rr['thread_speedup_6_vs_1']:.2f}")
        fig_vec(vt, FIG / "vectorisation")
        write_vec_table(vt, GEN / "table_vectorisation.tex")

    # ---- frequency probe -----------------------------------------------------------------------
    fp = m[m["experiment"] == "freq_probe"].copy()
    if len(fp):
        fp["p_eff"] = np.where(fp["method"].str.startswith("seq"), 0, fp["threads"])
        fq = (fp[fp["method"].str.startswith("omp")].groupby("threads")
              .agg(mhz=("freq_busy_mhz_median", "median"), p10=("freq_busy_mhz_p10", "median"),
                   p90=("freq_busy_mhz_p90", "median"), busy=("busy_cpus_median", "median"),
                   k=("freq_busy_mhz_median", "size"), cpu_util=("cpu_util", "median")).reset_index())
        fseq = fp[fp["method"].str.startswith("seq")]
        fq.to_csv(SUM / "frequency.csv", index=False)
        for _, rr in fq.iterrows():
            num.add(f"Freq{word(int(rr['threads']))}", f"{rr['mhz']:.0f}")
        if len(fseq):
            num.add("FreqSeq", f"{fseq['freq_busy_mhz_median'].median():.0f}")
        fig_freq(fq, FIG / "frequency")

    # ---- CPU utilisation (from the main sweep, long reps) --------------------------------------
    cu = long_reps[(long_reps["experiment"] == "scaling") & (long_reps["n"] == nmax)]
    cu_t = cu.groupby(["method", "threads"])["cpu_util"].median().reset_index()
    cu_t.to_csv(SUM / "cpu_util.csv", index=False)
    for _, rr in cu_t[cu_t["method"] == "omp_blocked"].iterrows():
        num.add(f"CpuUtil{word(int(rr['threads']))}", f"{rr['cpu_util']:.2f}")

    # ---- memory ---------------------------------------------------------------------------------
    sm = summary[summary["experiment"] == "scaling"].copy()
    sm["threads"] = np.where(sm["method"].str.startswith("seq"), 0, sm["threads"].astype(float))
    fig_memory(sm, FIG / "memory")
    for n, g in sm.groupby("n"):
        w = word(int(n))
        num.add(f"RssSeqN{w}", f"{g.loc[g['threads'] == 0, 'median_peak_rss_mib'].iloc[0]:.1f}")
        num.add(f"RssMaxN{w}", f"{g['median_peak_rss_mib'].max():.1f}")
        num.add(f"ArrayMibN{w}", f"{3 * 8 * n * n / 2 ** 20:.1f}")

    # ---- figures ------------------------------------------------------------------------------
    fig_speedup_eff(sp, FIG / "speedup_efficiency")
    fig_runtime(timed, sp, FIG / "runtime_vs_threads")
    if len(sp_ikj):
        fig_kernels(sp_all, FIG / "kernels_gflops")

    # ---- tables ---------------------------------------------------------------------------------
    write_speedup_table(sp, GEN / "table_speedup.tex")
    write_overhead_table(ov, GEN / "table_overhead.tex")
    write_full_tables(summary, GEN)
    num.write(GEN / "numbers.tex")
    print(f"analysis done: {len(timed)} measured rows; figures -> {FIG}; tables -> {GEN}")
    print(sp[sp["n"] == nmax][["threads", "T_p_median_s", "speedup", "speedup_ci_low", "speedup_ci_high",
                               "efficiency"]].to_string(index=False))


def write_speedup_table(sp, path):
    ns = sorted(sp["n"].unique())
    nmax = ns[-1]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Speedup $S(p)=T_{seq}/T_p$ of the blocked kernel (medians of the measured "
             r"repetitions; $k$ per cell in \cref{tab:full1,tab:full2}). For $n=" + str(nmax) + r"$ the 95\% bootstrap "
             r"confidence interval of $S$ and the efficiency $E=S/p$ are also given. "
             r"$p>6$ uses both hardware threads of a core (SMT); $p>12$ is oversubscribed.}",
             r"\label{tab:speedup}", r"\small",
             r"\begin{tabular}{@{}rl" + "r" * len(ns) + r"cr@{}}", r"\toprule",
             r"$p$ & region & " + " & ".join(f"$n={n}$" for n in ns) +
             rf" & 95\% CI ($n={nmax}$) & $E$ ($n={nmax}$) \\", r"\midrule"]
    for p, g in sp.groupby("threads"):
        g = g.set_index("n")
        cells = [f"{g.loc[n, 'speedup']:.2f}" if n in g.index else "--" for n in ns]
        b = g.loc[nmax]
        lines.append(f"{p} & {region(p).replace('oversubscribed', 'oversub.').replace('smt', 'SMT')} & "
                     + " & ".join(cells)
                     + f" & [{b['speedup_ci_low']:.2f}, {b['speedup_ci_high']:.2f}] & {b['efficiency']:.2f} \\\\")
        if p in (PHYS, LOGICAL):
            lines.append(r"\midrule")
    lines += [r"\midrule", r"\multicolumn{2}{@{}l}{$T_{seq}$ [s] (median)} & "
              + " & ".join(fmt_s(sp.loc[sp['n'] == n, 'T_seq_median_s'].iloc[0]) for n in ns) + r" & & \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def write_overhead_table(ov, path):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Overhead of the parallel binary run with one thread: ratio of medians "
             r"$T_{par}(1)/T_{seq}$ with 95\% bootstrap CI; times are medians [IQR] in seconds over $k$ "
             r"repetitions each.}", r"\label{tab:overhead}", r"\small",
             r"\begin{tabular}{@{}lrllcr@{}}", r"\toprule",
             r"Kernel & $n$ & $T_{seq}$ [IQR] & $T_{par}(1)$ [IQR] & ratio [95\% CI] & $k$ \\", r"\midrule"]
    for _, r in ov.iterrows():
        lines.append(f"{r['kernel']} & {r['n']} & {fmt_s(r['T_seq_median_s'])} [{fmt_s(r['T_seq_iqr_s'])}] & "
                     f"{fmt_s(r['T_par1_median_s'])} [{fmt_s(r['T_par1_iqr_s'])}] & "
                     f"{r['ratio']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] & {min(r['k_seq'], r['k_par1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_vec_table(vt, path):
    lab = {"omp_scalar": "scalar (\\texttt{-O2 -fno-tree-vectorize})", "omp_avx2": "AVX2 (\\texttt{-O3 -march=native})",
           "omp_fma": "AVX2+FMA (+\\texttt{-ffp-contract=fast})"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Vectorisation at fixed thread count (blocked kernel, OpenMP binaries). Median [IQR] "
             r"over $k$ repetitions; SIMD speedup = $T_{scalar}/T_{variant}$ at the same $p$ (95\% bootstrap CI).}",
             r"\label{tab:vec}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}rrlrrc@{}}", r"\toprule",
             r"$n$ & $p$ & build & median [IQR] (s) & GFLOP/s & SIMD speedup \\", r"\midrule"]
    order = {"omp_scalar": 0, "omp_avx2": 1, "omp_fma": 2}
    vt = vt.assign(o=vt["variant"].map(order)).sort_values(["n", "threads", "o"])
    for _, r in vt.iterrows():
        sp = "1 (ref.)" if r["variant"] == "omp_scalar" else \
            f"{r['speedup_vs_scalar']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        lines.append(f"{r['n']} & {r['threads']} & {lab[r['variant']].split(' (')[0]} & "
                     f"{fmt_s(r['median_s'])} [{fmt_s(r['iqr_s'])}] & {r['gflops']:.2f} & {sp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_full_tables(summary, gen):
    s = summary[summary["experiment"].isin(["scaling", "scaling_ikj"])].copy()
    s["kernel"] = s["method"].str.split("_").str[-1]
    s["impl"] = np.where(s["method"].str.startswith("seq"), "seq", "omp")
    s = s.sort_values(["kernel", "n", "impl", "threads"], ascending=[True, True, False, True])
    rows = []
    for _, r in s.iterrows():
        p = "seq" if r["impl"] == "seq" else str(int(r["threads"]))
        rows.append(f"{r['kernel']} & {int(r['n'])} & {p} & {fmt_s(r['median_s'])} & {fmt_s(r['iqr_s'])} & "
                    f"{int(r['n_ok'])} & {r['median_peak_rss_mib']:.1f} \\\\")
    half = (len(rows) + 1) // 2
    for idx, chunk in enumerate((rows[:half], rows[half:]), start=1):
        lines = [r"\begin{table}[!ht]", r"\centering",
                 rf"\caption{{All timing configurations ({idx}/2): median and IQR of the kernel time [s], "
                 r"number of measured repetitions $k$ and median peak RSS [MiB] (GNU time).}",
                 rf"\label{{tab:full{idx}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{@{}lrlrrrr@{}}", r"\toprule",
                 r"Kernel & $n$ & $p$ & median & IQR & $k$ & RSS \\", r"\midrule"] + chunk + \
                [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (gen / f"table_full{idx}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
