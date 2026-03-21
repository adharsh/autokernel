"""
AutoKernel -- Analysis & visualization of experiment results.

Reads results.tsv (multi-agent kernel optimization log), produces:
  - progress.png   : scatter of candidate latency over experiments
  - speedup.png    : scatter of speedup over experiments
  - report.md      : markdown session report
  - terminal output: summary statistics + delta ranking

Usage:  uv run analysis.py
"""
from __future__ import annotations
import os
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from profile_utils import TSV_COLUMNS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PNG = os.path.join(SCRIPT_DIR, "progress.png")
SPEEDUP_PNG = os.path.join(SCRIPT_DIR, "speedup.png")
REPORT_MD = os.path.join(SCRIPT_DIR, "report.md")

AGENT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
COLOR_KEEP, COLOR_DISCARD, COLOR_CRASH = "#2ecc71", "#cccccc", "#e74c3c"
COLOR_FRONTIER, COLOR_BASELINE = "#27ae60", "#3498db"
GPU_VRAM_CAPACITY_MB = 81920  # 80 GB default


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str = "results.tsv") -> pd.DataFrame | None:
    """Read TSV, normalize columns, convert numerics. None if missing/empty."""
    resolved = path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)
    if not os.path.exists(resolved):
        return None
    df = pd.read_csv(resolved, sep="\t")
    if len(df) == 0:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("candidate_us", "reference_us", "speedup", "peak_vram_mb", "agent_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("_ts").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_row(row) -> str:
    """Classify a row as 'keep', 'discard', or 'crash'."""
    raw_status = row.get("status", "")
    status = str(raw_status).strip().lower() if pd.notna(raw_status) else ""
    if status in ("keep", "kept"):
        return "keep"
    if status == "discard":
        return "discard"
    if status == "crash":
        return "crash"
    raw_corr = row.get("correctness", "")
    corr = str(raw_corr).strip().upper() if pd.notna(raw_corr) else ""
    if corr in ("CRASH", "ERROR"):
        return "crash"
    if corr == "FAIL":
        return "discard"
    return "keep" if corr == "PASS" else "discard"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _agent_marker(agent_id) -> str:
    try:
        return AGENT_MARKERS[int(agent_id) % len(AGENT_MARKERS)]
    except (ValueError, TypeError):
        return "o"


def _scatter_by_agent(ax, sub, xcol, ycol, color, label, alpha, z):
    """Scatter with per-agent markers. Only the first agent gets the legend."""
    used = False
    for agent in sorted(sub["agent_id"].fillna(0).unique()):
        chunk = sub[sub["agent_id"].fillna(0) == agent]
        ax.scatter(chunk[xcol], chunk[ycol], c=color, marker=_agent_marker(agent),
                   s=45, alpha=alpha, edgecolors="none",
                   label=(label if not used else None), zorder=z)
        used = True


def _annotate_top3(ax, kept, xcol, ycol, lower_better):
    if len(kept) == 0:
        return
    top = kept.sort_values(ycol, ascending=lower_better).head(3)
    for rank, (_, r) in enumerate(top.iterrows()):
        desc = str(r.get("description", "")).strip()
        if len(desc) > 40:
            desc = desc[:37] + "..."
        ax.annotate(f"#{rank+1}: {desc}", xy=(r[xcol], r[ycol]),
                    xytext=(10, 10 + rank * 15), textcoords="offset points",
                    fontsize=7.5, color=COLOR_FRONTIER, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLOR_FRONTIER, lw=0.8),
                    zorder=6)


def _ref_and_best(df, cats):
    """Extract reference_us and best kept candidate_us."""
    ref_us = None
    if "reference_us" in df.columns:
        vals = df["reference_us"].dropna()
        if len(vals) > 0:
            ref_us = float(vals.iloc[0])
    best_us = None
    kept = df[cats == "keep"]
    if "candidate_us" in kept.columns:
        v = kept["candidate_us"].dropna()
        if len(v) > 0:
            best_us = float(v.min())
    return ref_us, best_us


# ---------------------------------------------------------------------------
# Task 5: Main progress plot (latency, lower is better)
# ---------------------------------------------------------------------------

def make_progress_plot(df: pd.DataFrame) -> None:
    """Scatter of candidate_us over experiments. Running min frontier."""
    if "candidate_us" not in df.columns:
        print("WARNING: candidate_us column missing -- skipping progress plot.")
        return
    df = df.copy()
    df["_cat"] = df.apply(classify_row, axis=1)
    df["_n"] = range(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(13, 6))
    for cat, col, lbl, a, z in [("discard", COLOR_DISCARD, "Discard", .55, 2),
                                 ("crash", COLOR_CRASH, "Crash", .65, 3),
                                 ("keep", COLOR_KEEP, "Keep", .85, 4)]:
        s = df[df["_cat"] == cat]
        if len(s):
            _scatter_by_agent(ax, s, "_n", "candidate_us", col, lbl, a, z)
    valid = df[(df["_cat"] == "keep") & df["candidate_us"].notna()]
    if len(valid):
        ax.step(valid["_n"], valid["candidate_us"].cummin(), where="post",
                color=COLOR_FRONTIER, linewidth=2, alpha=.8, label="Running best", zorder=3)
    if "reference_us" in df.columns and len(df):
        ref = df.iloc[0].get("reference_us")
        if pd.notna(ref) and float(ref) > 0:
            ax.axhline(y=float(ref), color=COLOR_BASELINE, linestyle="--",
                       linewidth=1.5, alpha=.7, label=f"Reference ({float(ref):.1f} us)", zorder=1)
    _annotate_top3(ax, df[df["_cat"] == "keep"].dropna(subset=["candidate_us"]),
                   "_n", "candidate_us", lower_better=True)
    nt, nk = len(df), (df["_cat"] == "keep").sum()
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Candidate Latency (us) -- lower is better", fontsize=11)
    ax.set_title(f"AutoKernel -- Optimization Progress: {nt} experiments, {nk} kept",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=.9)
    ax.grid(True, alpha=.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    fig.savefig(PROGRESS_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PROGRESS_PNG}")


# ---------------------------------------------------------------------------
# Task 15: Secondary speedup plot (higher is better)
# ---------------------------------------------------------------------------

def make_speedup_plot(df: pd.DataFrame) -> None:
    """Scatter of speedup over experiments. Running max frontier."""
    if "speedup" not in df.columns:
        print("WARNING: speedup column missing -- skipping speedup plot.")
        return
    df = df.copy()
    df["_cat"] = df.apply(classify_row, axis=1)
    df["_n"] = range(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(13, 6))
    for cat, col, lbl, a, z in [("discard", COLOR_DISCARD, "Discard", .55, 2),
                                 ("crash", COLOR_CRASH, "Crash", .65, 3),
                                 ("keep", COLOR_KEEP, "Keep", .85, 4)]:
        s = df[df["_cat"] == cat]
        if len(s):
            _scatter_by_agent(ax, s, "_n", "speedup", col, lbl, a, z)
    valid = df[(df["_cat"] == "keep") & df["speedup"].notna()]
    if len(valid):
        ax.step(valid["_n"], valid["speedup"].cummax(), where="post",
                color=COLOR_FRONTIER, linewidth=2, alpha=.8, label="Running best", zorder=3)
    ax.axhline(y=1.0, color=COLOR_BASELINE, linestyle="--", linewidth=1.5,
               alpha=.7, label="1.0x baseline", zorder=1)
    _annotate_top3(ax, df[df["_cat"] == "keep"].dropna(subset=["speedup"]),
                   "_n", "speedup", lower_better=False)
    nt, nk = len(df), (df["_cat"] == "keep").sum()
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Speedup (higher is better)", fontsize=11)
    ax.set_title(f"AutoKernel -- Speedup Progress: {nt} experiments, {nk} kept",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=.9)
    ax.grid(True, alpha=.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(SPEEDUP_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SPEEDUP_PNG}")


# ---------------------------------------------------------------------------
# Task 9: Suggestions engine
# ---------------------------------------------------------------------------

def generate_suggestions(df: pd.DataFrame, baseline: float | None,
                         best: float | None) -> list[str]:
    """Actionable suggestions based on experiment history."""
    suggestions: list[str] = []
    n_total = len(df)
    if n_total == 0:
        return ["Run some experiments first to generate suggestions."]
    cats = df.apply(classify_row, axis=1)
    n_crash = int((cats == "crash").sum())

    if n_crash / n_total > 0.4:
        suggestions.append(
            f"High crash rate ({n_crash/n_total*100:.0f}%). Consider more "
            "conservative changes, better error handling, or input validation.")
    last5 = cats.tail(5)
    if len(last5) >= 5 and all(c in ("discard", "crash") for c in last5):
        suggestions.append(
            "Last 5 experiments all discard/crash -- possible plateau. "
            "Try a fundamentally different approach (algorithm, memory layout, "
            "or kernel fusion strategy).")
    if baseline and best and baseline > 0:
        spd = baseline / best
        if spd < 1.1:
            suggestions.append(
                "Speedup is modest (<1.1x). Consider: autotuning block sizes, "
                "persistent kernels, split-K strategies.")
        elif spd < 1.5:
            suggestions.append(
                "Decent speedup. Next: software pipelining, warp specialization, "
                "or TMA-based data movement.")
        else:
            suggestions.append(
                f"Strong speedup (>{spd:.2f}x). Consider: fine-grained autotuning "
                "across more configs, profiling bottlenecks with ncu.")
    if "peak_vram_mb" in df.columns:
        kept_vram = df.loc[cats == "keep", "peak_vram_mb"].dropna()
        kept_vram = kept_vram[kept_vram > 0]
        thresh = GPU_VRAM_CAPACITY_MB * 0.8
        if len(kept_vram) and float(kept_vram.max()) > thresh:
            suggestions.append(
                f"Peak VRAM high ({float(kept_vram.max()):.0f} MB, "
                f">{thresh:.0f} MB). Consider memory-efficient techniques.")
    if not suggestions:
        suggestions.append(
            "Continue iterating. Try systematic autotuning of block sizes "
            "and Triton-specific optimizations (num_warps, num_stages).")
    return suggestions


# ---------------------------------------------------------------------------
# Task 5 + 16: Terminal summary (includes per-agent breakdown)
# ---------------------------------------------------------------------------

def print_terminal_summary(df: pd.DataFrame) -> None:
    """Print bordered summary with key stats, top-5, per-agent breakdown."""
    cats = df.apply(classify_row, axis=1)
    n_total = len(df)
    n_keep = int((cats == "keep").sum())
    n_discard = int((cats == "discard").sum())
    n_crash = int((cats == "crash").sum())
    ref_us, best_us = _ref_and_best(df, cats)
    spd = (ref_us / best_us) if (ref_us and best_us and best_us > 0) else None

    print()
    print("=" * 60)
    print("  AutoKernel -- Session Summary")
    print("=" * 60)
    if ref_us:
        print(f"\n  Reference latency:     {ref_us:.2f} us")
    if best_us:
        print(f"  Best candidate:        {best_us:.2f} us")
    if spd:
        print(f"  Total speedup:         {spd:.2f}x")
    kp = (n_keep / n_total * 100) if n_total else 0
    cp = (n_crash / n_total * 100) if n_total else 0
    print(f"\n  Experiments:           {n_total}")
    print(f"  Kept:                  {n_keep} ({kp:.0f}%)")
    print(f"  Discarded:             {n_discard}")
    print(f"  Crashed:               {n_crash} ({cp:.0f}%)")

    # Top 5
    kept_df = df[cats == "keep"]
    if "candidate_us" in df.columns and n_keep > 0:
        top5 = kept_df.dropna(subset=["candidate_us"]).sort_values("candidate_us").head(5)
        if len(top5):
            print("\n  Top 5 improvements:")
            for rank, (_, r) in enumerate(top5.iterrows(), 1):
                c = float(r["candidate_us"])
                s = f"{float(r['speedup']):.2f}x" if pd.notna(r.get("speedup")) else "N/A"
                print(f"    {rank}. {c:.2f} us ({s}) -- {r.get('description','')}")

    # Per-agent breakdown (Task 16)
    if "agent_id" in df.columns:
        agents = sorted(df["agent_id"].dropna().unique())
        if agents:
            print("\n  Per-agent breakdown:")
            for ag in agents:
                m = df["agent_id"] == ag
                ad, ac = df[m], cats[m]
                at, ak = len(ad), int((ac == "keep").sum())
                ab = None
                akd = ad[ac == "keep"]
                if "candidate_us" in akd.columns:
                    v = akd["candidate_us"].dropna()
                    if len(v):
                        ab = float(v.min())
                bs = f", best {ab:.2f} us" if ab else ""
                sp = f" ({ref_us/ab:.2f}x)" if (ab and ref_us and ab > 0) else ""
                print(f"    Agent {int(ag)}: {at} experiments, {ak} kept{bs}{sp}")
    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Task 18: Delta ranking
# ---------------------------------------------------------------------------

def print_delta_ranking(df: pd.DataFrame) -> None:
    """Each kept experiment's incremental improvement, sorted largest first."""
    if "candidate_us" not in df.columns:
        return
    cats = df.apply(classify_row, axis=1)
    kept = df[cats == "keep"].dropna(subset=["candidate_us"]).reset_index(drop=True)
    if len(kept) < 2:
        return
    deltas: list[tuple[float, float, str]] = []
    best = float(kept.iloc[0]["candidate_us"])
    for i in range(1, len(kept)):
        cand = float(kept.iloc[i]["candidate_us"])
        delta = best - cand
        if delta > 0:
            deltas.append((delta, cand, str(kept.iloc[i].get("description", "")).strip()))
            best = cand
    if not deltas:
        return
    deltas.sort(key=lambda x: x[0], reverse=True)
    print("Delta Ranking (incremental improvement per kept experiment)")
    print(f"{'Rank':>4}  {'Delta (us)':>10}  {'Latency':>10}  Description")
    print("-" * 72)
    for rank, (d, lat, desc) in enumerate(deltas, 1):
        print(f"{rank:4d}  {-d:>+10.2f}  {lat:10.2f}  {desc}")
    total = sum(x[0] for x in deltas)
    print("-" * 72)
    print(f"{'':>4}  {-total:>+10.2f}  {'':>10}  TOTAL improvement\n")


# ---------------------------------------------------------------------------
# Task 5: Report generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame) -> None:
    """Save report.md with summary table, key discoveries, suggestions."""
    cats = df.apply(classify_row, axis=1)
    nt = len(df)
    nk, nd, nc = int((cats == "keep").sum()), int((cats == "discard").sum()), int((cats == "crash").sum())
    ref_us, best_us = _ref_and_best(df, cats)
    spd = (ref_us / best_us) if (ref_us and best_us and best_us > 0) else None
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = [
        "# AutoKernel Session Report", "", f"Generated: {ts}", "",
        "## Summary", "", "| Metric | Value |", "|--------|-------|",
        f"| Total experiments | {nt} |", f"| Kept | {nk} |",
        f"| Discarded | {nd} |", f"| Crashed | {nc} |",
    ]
    if ref_us:
        L.append(f"| Reference latency | {ref_us:.2f} us |")
    if best_us:
        L.append(f"| Best candidate latency | {best_us:.2f} us |")
    if spd:
        L.append(f"| Speedup | {spd:.2f}x |")
    L.append("")

    kept_df = df[cats == "keep"]
    if len(kept_df) and "candidate_us" in kept_df.columns:
        L += ["## Key Discoveries (Kept)", ""]
        for _, r in kept_df.sort_values("candidate_us").iterrows():
            c = f"{float(r['candidate_us']):.2f} us" if pd.notna(r.get("candidate_us")) else "N/A"
            s = f"{float(r['speedup']):.2f}x" if pd.notna(r.get("speedup")) else "N/A"
            L.append(f"- **{r.get('experiment_id','?')}**: {c} (speedup: {s}) -- "
                      f"{r.get('description','')}")
        L.append("")

    failed = df[cats.isin(["crash", "discard"])]
    if len(failed):
        L += ["## Failed / Discarded Experiments", ""]
        for _, r in failed.iterrows():
            L.append(f"- **{r.get('experiment_id','?')}** "
                      f"[{r.get('status','?')}/{r.get('correctness','?')}]: "
                      f"{r.get('description','')}")
        L.append("")

    L += ["## Suggestions for Next Session", ""]
    for s in generate_suggestions(df, ref_us, best_us):
        L.append(f"- {s}")
    L.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Saved: {REPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_results()
    if df is None:
        print("No results.tsv found.")
        return
    if len(df) == 0:
        print("No experiments yet (results.tsv contains only the header).")
        return
    make_progress_plot(df)
    make_speedup_plot(df)
    print_terminal_summary(df)
    print_delta_ranking(df)
    generate_report(df)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
