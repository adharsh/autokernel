"""
AutoKernel -- Analysis & visualization of experiment results.

Reads results/experiments.tsv (multi-agent kernel optimization log), produces:
  - progress.html  : interactive scatter of candidate latency over experiments
  - speedup.html   : interactive scatter of speedup over experiments
  - report.md      : markdown session report
  - terminal output: summary statistics + delta ranking

Usage:  uv run analysis.py
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from profile_utils import TSV_COLUMNS

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROGRESS_HTML = RESULTS_DIR / "progress.html"
SPEEDUP_HTML = RESULTS_DIR / "speedup.html"
REPORT_MD = RESULTS_DIR / "report.md"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
REQUIRED_NOTE_SECTIONS = (
    "## NCU Profile",
    "## Speed-of-Light Gap",
    "## Design Decision From Profile",
    "## Codegen/PTX/SASS",
)

AGENT_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down",
                 "cross", "x", "star"]
COLOR_KEEP, COLOR_DISCARD, COLOR_CRASH = "#2ecc71", "#cccccc", "#e74c3c"
COLOR_FRONTIER, COLOR_BASELINE = "#27ae60", "#3498db"
GPU_VRAM_CAPACITY_MB = 81920  # 80 GB default


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str = "results/experiments.tsv") -> pd.DataFrame | None:
    """Read TSV, normalize columns, convert numerics. None if missing/empty."""
    resolved = Path(path) if Path(path).is_absolute() else SCRIPT_DIR / path
    if not resolved.exists():
        return None
    df = pd.read_csv(resolved, sep="\t")
    if len(df) == 0:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("candidate_us", "reference_us", "speedup", "peak_vram_mb", "agent_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
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

def _agent_symbol(agent_id) -> str:
    try:
        return AGENT_SYMBOLS[int(agent_id) % len(AGENT_SYMBOLS)]
    except (ValueError, TypeError):
        return "circle"


def _ref_and_best(df, cats):
    """Extract best kept candidate_us and best per-row speedup."""
    ref_us = None
    if "reference_us" in df.columns:
        vals = df["reference_us"].dropna()
        if len(vals) > 0:
            ref_us = float(vals.iloc[0])
    best_us = None
    best_speedup = None
    kept = df[cats == "keep"]
    if "candidate_us" in kept.columns:
        v = kept["candidate_us"].dropna()
        if len(v) > 0:
            best_us = float(v.min())
    if "speedup" in kept.columns:
        v = kept["speedup"].dropna()
        if len(v) > 0:
            best_speedup = float(v.max())
    return ref_us, best_us, best_speedup


def _safe_experiment_id(experiment_id) -> str:
    return str(experiment_id).replace("/", "_")


def profiling_coverage(df: pd.DataFrame) -> dict:
    """Check whether experiments have required NCU artifacts and note sections."""
    missing_ncu: list[str] = []
    missing_note_sections: list[str] = []
    total = len(df)

    for _, row in df.iterrows():
        eid = row.get("experiment_id", "")
        if pd.isna(eid) or str(eid).strip() == "":
            continue
        safe_id = _safe_experiment_id(eid)
        experiment_dir = EXPERIMENTS_DIR / safe_id
        ncu_report = experiment_dir / "ncu" / "profile.ncu-rep"
        ncu_log = experiment_dir / "ncu" / "profile.log"
        if not ncu_report.exists() and not ncu_log.exists():
            missing_ncu.append(str(eid))

        note_path = experiment_dir / "note.md"
        if not note_path.exists():
            missing_note_sections.append(f"{eid}: missing note")
            continue
        note = note_path.read_text(encoding="utf-8", errors="replace")
        missing = [section for section in REQUIRED_NOTE_SECTIONS if section not in note]
        if missing:
            missing_note_sections.append(f"{eid}: missing {', '.join(missing)}")

    return {
        "total": total,
        "with_ncu": total - len(missing_ncu),
        "missing_ncu": missing_ncu,
        "with_profile_notes": total - len(missing_note_sections),
        "missing_note_sections": missing_note_sections,
    }


def _hover_text(row) -> str:
    """Build rich hover text for a single experiment row."""
    parts = []
    for col, label in [("experiment_id", "Branch"),
                       ("description", "Description"),
                       ("candidate_us", "Latency"),
                       ("speedup", "Speedup"),
                       ("correctness", "Correctness"),
                       ("status", "Status"),
                       ("commit", "Commit"),
                       ("agent_id", "Agent"),
                       ("peak_vram_mb", "VRAM (MB)"),
                       ("parent_id", "Parent")]:
        val = row.get(col)
        if pd.isna(val):
            continue
        if col == "candidate_us":
            parts.append(f"<b>{label}:</b> {float(val):.2f} us")
        elif col == "speedup":
            parts.append(f"<b>{label}:</b> {float(val):.3f}x")
        elif col == "peak_vram_mb" and float(val) > 0:
            parts.append(f"<b>{label}:</b> {float(val):.0f}")
        elif col == "agent_id":
            parts.append(f"<b>{label}:</b> a{int(val)}")
        elif col not in ("candidate_us", "speedup", "peak_vram_mb", "agent_id"):
            parts.append(f"<b>{label}:</b> {val}")
    return "<br>".join(parts)


def _add_scatter_traces(fig, df, ycol):
    """Add scatter traces grouped by category and agent, with hover info."""
    cat_style = [
        ("discard", COLOR_DISCARD, "Discard", 0.55),
        ("crash",   COLOR_CRASH,   "Crash",   0.65),
        ("keep",    COLOR_KEEP,    "Keep",    0.85),
    ]
    for cat, color, label, opacity in cat_style:
        sub = df[df["_cat"] == cat]
        if len(sub) == 0:
            continue
        for i, agent in enumerate(sorted(sub["agent_id"].fillna(0).unique())):
            chunk = sub[sub["agent_id"].fillna(0) == agent]
            fig.add_trace(go.Scatter(
                x=chunk["_n"], y=chunk[ycol], mode="markers",
                marker=dict(color=color, symbol=_agent_symbol(agent),
                            size=8, opacity=opacity),
                text=chunk["_hover"], hoverinfo="text",
                name=label if i == 0 else None,
                legendgroup=cat, showlegend=(i == 0),
            ))


# ---------------------------------------------------------------------------
# Interactive progress plot (latency, lower is better)
# ---------------------------------------------------------------------------

def make_progress_plot(df: pd.DataFrame) -> None:
    """Interactive scatter of candidate_us over experiments."""
    if "candidate_us" not in df.columns:
        print("WARNING: candidate_us column missing -- skipping progress plot.")
        return
    df = df.copy()
    df["_cat"] = df.apply(classify_row, axis=1)
    df["_n"] = range(1, len(df) + 1)
    df["_hover"] = df.apply(_hover_text, axis=1)

    fig = go.Figure()
    _add_scatter_traces(fig, df, "candidate_us")

    # Running-best frontier
    valid = df[(df["_cat"] == "keep") & df["candidate_us"].notna()]
    if len(valid):
        frontier = valid["candidate_us"].cummin()
        fig.add_trace(go.Scatter(
            x=valid["_n"], y=frontier, mode="lines",
            line=dict(color=COLOR_FRONTIER, width=2, shape="hv"),
            name="Running best", hoverinfo="skip",
        ))

    # Reference baseline
    if "reference_us" in df.columns and len(df):
        ref = df.iloc[0].get("reference_us")
        if pd.notna(ref) and float(ref) > 0:
            fig.add_hline(y=float(ref), line_dash="dash", line_color=COLOR_BASELINE,
                          annotation_text=f"Reference ({float(ref):.1f} us)",
                          annotation_position="top left")

    nt, nk = len(df), (df["_cat"] == "keep").sum()
    fig.update_layout(
        title=f"AutoKernel -- Optimization Progress: {nt} experiments, {nk} kept",
        xaxis_title="Experiment #",
        yaxis_title="Candidate Latency (us) -- lower is better",
        hovermode="closest",
        template="plotly_white",
        height=600, width=1100,
    )
    fig.write_html(PROGRESS_HTML)
    print(f"Saved: {PROGRESS_HTML}")


# ---------------------------------------------------------------------------
# Interactive speedup plot (higher is better)
# ---------------------------------------------------------------------------

def make_speedup_plot(df: pd.DataFrame) -> None:
    """Interactive scatter of speedup over experiments."""
    if "speedup" not in df.columns:
        print("WARNING: speedup column missing -- skipping speedup plot.")
        return
    df = df.copy()
    df["_cat"] = df.apply(classify_row, axis=1)
    df["_n"] = range(1, len(df) + 1)
    df["_hover"] = df.apply(_hover_text, axis=1)

    fig = go.Figure()
    _add_scatter_traces(fig, df, "speedup")

    # Running-best frontier
    valid = df[(df["_cat"] == "keep") & df["speedup"].notna()]
    if len(valid):
        frontier = valid["speedup"].cummax()
        fig.add_trace(go.Scatter(
            x=valid["_n"], y=frontier, mode="lines",
            line=dict(color=COLOR_FRONTIER, width=2, shape="hv"),
            name="Running best", hoverinfo="skip",
        ))

    # 1.0x baseline
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLOR_BASELINE,
                  annotation_text="1.0x baseline", annotation_position="top left")

    nt, nk = len(df), (df["_cat"] == "keep").sum()
    fig.update_layout(
        title=f"AutoKernel -- Speedup Progress: {nt} experiments, {nk} kept",
        xaxis_title="Experiment #",
        yaxis_title="Speedup (higher is better)",
        yaxis=dict(rangemode="tozero"),
        hovermode="closest",
        template="plotly_white",
        height=600, width=1100,
    )
    fig.write_html(SPEEDUP_HTML)
    print(f"Saved: {SPEEDUP_HTML}")


# ---------------------------------------------------------------------------
# Task 9: Suggestions engine
# ---------------------------------------------------------------------------

def generate_suggestions(df: pd.DataFrame, best_speedup: float | None) -> list[str]:
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
    if best_speedup and best_speedup > 0:
        spd = best_speedup
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
    ref_us, best_us, best_speedup = _ref_and_best(df, cats)

    print()
    print("=" * 60)
    print("  AutoKernel -- Session Summary")
    print("=" * 60)
    if ref_us:
        print(f"\n  Reference latency:     {ref_us:.2f} us")
    if best_us:
        print(f"  Best candidate:        {best_us:.2f} us")
    if best_speedup:
        print(f"  Best speedup:          {best_speedup:.2f}x")
    kp = (n_keep / n_total * 100) if n_total else 0
    cp = (n_crash / n_total * 100) if n_total else 0
    print(f"\n  Experiments:           {n_total}")
    print(f"  Kept:                  {n_keep} ({kp:.0f}%)")
    print(f"  Discarded:             {n_discard}")
    print(f"  Crashed:               {n_crash} ({cp:.0f}%)")
    coverage = profiling_coverage(df)
    print(f"  NCU artifacts:         {coverage['with_ncu']}/{coverage['total']}")
    print(f"  Profile note sections: {coverage['with_profile_notes']}/{coverage['total']}")

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
                agent_spd = None
                if "speedup" in akd.columns:
                    v = akd["speedup"].dropna()
                    if len(v):
                        agent_spd = float(v.max())
                bs = f", best {ab:.2f} us" if ab else ""
                sp = f" ({agent_spd:.2f}x)" if agent_spd else ""
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
    ref_us, best_us, best_speedup = _ref_and_best(df, cats)
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
    if best_speedup:
        L.append(f"| Best speedup | {best_speedup:.2f}x |")
    L.append("")

    coverage = profiling_coverage(df)
    L += [
        "## Profiling Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| NCU artifacts present | {coverage['with_ncu']}/{coverage['total']} |",
        f"| Notes with required profile sections | {coverage['with_profile_notes']}/{coverage['total']} |",
        "",
    ]
    if coverage["missing_ncu"]:
        L.append("Missing NCU artifacts:")
        for eid in coverage["missing_ncu"][:20]:
            L.append(f"- {eid}")
        if len(coverage["missing_ncu"]) > 20:
            L.append(f"- ... {len(coverage['missing_ncu']) - 20} more")
        L.append("")
    if coverage["missing_note_sections"]:
        L.append("Missing note profile sections:")
        for item in coverage["missing_note_sections"][:20]:
            L.append(f"- {item}")
        if len(coverage["missing_note_sections"]) > 20:
            L.append(f"- ... {len(coverage['missing_note_sections']) - 20} more")
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
    for s in generate_suggestions(df, best_speedup):
        L.append(f"- {s}")
    L.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Saved: {REPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_results()
    if df is None:
        print("No results/experiments.tsv found.")
        return
    if len(df) == 0:
        print("No experiments yet (experiments.tsv contains only the header).")
        return
    make_progress_plot(df)
    make_speedup_plot(df)
    print_terminal_summary(df)
    print_delta_ranking(df)
    generate_report(df)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
