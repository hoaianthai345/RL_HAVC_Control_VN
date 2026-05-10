#!/usr/bin/env python3
"""
Generate results figures for the manuscript from raw CSVs.

Reads results/raw/<run_id>/**/*.csv (recursive), aggregates per
(context_id, method), and emits manuscript-ready PDFs + PNGs.

Figures produced (saved to --output-dir):
    fig5_energy_by_method.pdf      Energy_kWh per method × context (bar)
    fig6_comfort_by_method.pdf     Comfort violation rate per method × context (bar)
    fig7_transfer_efficiency.pdf   Energy ratio target/source per method (bar)
    fig8_latency_distribution.pdf  Mean & p95 latency edge vs cloud (bar)
    fig9_amy_shift.pdf             TMYx baseline vs AMY years (line per method)

Usage:
    python scripts/plot_results.py \
        --input-dir results/raw/run_20260508_063450 \
        --output-dir manuscript_eaai/figures \
        --plots all

    # single plot:
    python scripts/plot_results.py --input-dir ... --plots energy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── plot style: paper-friendly defaults ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

METHOD_ORDER = [
    "static", "ashrae", "ppo_static", "ppo_multicontext",
    "edge_only", "cloud_only", "edge_cloud_adaptive",
]
METHOD_LABEL = {
    "static": "Static",
    "ashrae": "ASHRAE",
    "ppo_static": "PPO (single-ctx)",
    "ppo_multicontext": "PPO (multi-ctx)",
    "edge_only": "Edge-only PPO",
    "cloud_only": "Cloud-only",
    "edge_cloud_adaptive": "Edge-cloud adaptive",
}
METHOD_COLOR = {
    "static": "#9e9e9e",
    "ashrae": "#607d8b",
    "ppo_static": "#1f77b4",
    "ppo_multicontext": "#2ca02c",
    "edge_only": "#ff7f0e",
    "cloud_only": "#d62728",
    "edge_cloud_adaptive": "#9467bd",
}

CITY_FROM_CTX = {
    "vn_hcmc":   "HCMC",
    "vn_cantho": "CanTho",
    "vn_danang": "DaNang",
    "vn_hanoi":  "Hanoi",
}


# ── data loading ─────────────────────────────────────────────────────────────

def load_control(input_dir: Path) -> pd.DataFrame:
    """Load all control-rollout CSVs (energy/comfort) into a single DataFrame."""
    paths = sorted(input_dir.rglob("*.csv"))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "energy_kwh" in df.columns and "comfort_violation_rate" in df.columns:
            df["_source"] = p.name
            frames.append(df)
    if not frames:
        raise SystemExit(f"No control CSVs found under {input_dir}")
    return pd.concat(frames, ignore_index=True)


def load_latency(input_dir: Path) -> pd.DataFrame | None:
    paths = sorted(input_dir.rglob("*.csv"))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "p95_latency_ms" in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def city_of(ctx_id: str) -> str:
    for prefix, name in CITY_FROM_CTX.items():
        if ctx_id.startswith(prefix):
            return name
    return ctx_id.split("_")[1] if "_" in ctx_id else ctx_id


def short_ctx(ctx_id: str) -> str:
    """e.g. vn_hcmc_officesmall → HCMC/OfficeSmall."""
    parts = ctx_id.split("_")
    if len(parts) >= 3:
        city = CITY_FROM_CTX.get(f"{parts[0]}_{parts[1]}", parts[1])
        bldg = parts[2]
        bldg = bldg.replace("officesmall", "OffSm").replace("officemedium", "OffMd").replace("hospital", "Hosp")
        return f"{city}/{bldg}"
    return ctx_id


def aggregate_mean_ci(df: pd.DataFrame, group: list[str], metric: str) -> pd.DataFrame:
    """Mean ± 95% CI across seeds (assume seed in groupby of original)."""
    g = df.groupby(group, dropna=False)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci95"] = 1.96 * out["se"]
    return out


def _legend_above(ax: plt.Axes, ncol: int = 4) -> None:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        fontsize=7.5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.4,
    )


# ── plot 5: energy by method × context ───────────────────────────────────────

def plot_energy_by_method(ctrl: pd.DataFrame, out_dir: Path) -> Path:
    df = ctrl[ctrl["weather_type"] == "TMYx"].copy() if "weather_type" in ctrl else ctrl.copy()
    agg = aggregate_mean_ci(df, ["context_id", "method"], "energy_kwh")
    contexts = sorted(agg["context_id"].unique(), key=lambda c: (city_of(c), c))
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(contexts))
    for i, m in enumerate(methods):
        sub = agg[agg["method"] == m].set_index("context_id").reindex(contexts)
        ax.bar(x + (i - len(methods) / 2 + 0.5) * width, sub["mean"],
               yerr=sub["ci95"], width=width,
               label=METHOD_LABEL.get(m, m), color=METHOD_COLOR.get(m, None),
               capsize=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_ctx(c) for c in contexts], rotation=30, ha="right")
    ax.set_ylabel("Daily energy (kWh)")
    ax.set_title("Energy consumption by method across Vietnam contexts", pad=34)
    _legend_above(ax)
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.10, right=0.99)
    return _save(fig, out_dir / "fig5_energy_by_method")


# ── plot 6: comfort violation by method × context ────────────────────────────

def plot_comfort_by_method(ctrl: pd.DataFrame, out_dir: Path) -> Path:
    df = ctrl[ctrl["weather_type"] == "TMYx"].copy() if "weather_type" in ctrl else ctrl.copy()
    agg = aggregate_mean_ci(df, ["context_id", "method"], "comfort_violation_rate")
    contexts = sorted(agg["context_id"].unique(), key=lambda c: (city_of(c), c))
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(contexts))
    for i, m in enumerate(methods):
        sub = agg[agg["method"] == m].set_index("context_id").reindex(contexts)
        ax.bar(x + (i - len(methods) / 2 + 0.5) * width, sub["mean"],
               yerr=sub["ci95"], width=width,
               label=METHOD_LABEL.get(m, m), color=METHOD_COLOR.get(m, None),
               capsize=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_ctx(c) for c in contexts], rotation=30, ha="right")
    ax.set_ylabel("Comfort violation rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comfort violation rate by method across Vietnam contexts", pad=34)
    _legend_above(ax)
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.10, right=0.99)
    return _save(fig, out_dir / "fig6_comfort_by_method")


# ── plot 7: transfer efficiency target/source ratio ──────────────────────────

def plot_transfer_efficiency(ctrl: pd.DataFrame, out_dir: Path) -> Path:
    """For each method, average target-city energy / source-city energy (lower is better)."""
    if "transfer_bucket" not in ctrl.columns:
        # Fallback: derive from city
        bucket_map = {"HCMC": "source", "CanTho": "source", "DaNang": "target", "Hanoi": "target"}
        ctrl = ctrl.copy()
        ctrl["transfer_bucket"] = ctrl["context_id"].apply(lambda c: bucket_map.get(city_of(c), "unknown"))
    df = ctrl[ctrl["weather_type"] == "TMYx"].copy() if "weather_type" in ctrl else ctrl.copy()
    src = df[df["transfer_bucket"] == "source"].groupby("method")["energy_kwh"].mean()
    tgt = df[df["transfer_bucket"] == "target"].groupby("method")["energy_kwh"].mean()
    methods = [m for m in METHOD_ORDER if m in src.index and m in tgt.index]
    if not methods:
        print("[plot] skip transfer_efficiency: no source/target rows")
        return Path()
    ratios = [tgt[m] / src[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = [METHOD_LABEL.get(m, m) for m in methods]
    y = np.arange(len(methods))
    bars = ax.barh(
        y, ratios,
        color=[METHOD_COLOR.get(m, "#888") for m in methods],
        edgecolor="white", linewidth=0.5,
    )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, r in zip(bars, ratios):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{r:.2f}",
            ha="left", va="center", fontsize=8.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Target / Source energy ratio")
    ax.set_title("Transfer efficiency on target cities (DaNang, Hanoi)")
    ax.set_xlim(0, max(ratios) * 1.18)
    fig.subplots_adjust(left=0.33, right=0.96, top=0.88, bottom=0.14)
    return _save(fig, out_dir / "fig7_transfer_efficiency")


# ── plot 8: latency distribution (edge vs cloud) ─────────────────────────────

def plot_latency(lat: pd.DataFrame | None, out_dir: Path) -> Path:
    if lat is None or lat.empty:
        print("[plot] skip latency: no latency CSV")
        return Path()
    metrics = [c for c in ["mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "max_latency_ms"]
               if c in lat.columns]
    if not metrics:
        print("[plot] skip latency: no latency metric columns")
        return Path()
    agg = lat.groupby("method")[metrics].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.8))
    methods = agg["method"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        vals = agg.loc[agg["method"] == m, metrics].iloc[0].values
        ax.bar(x + (i - len(methods) / 2 + 0.5) * width, vals, width,
               label=METHOD_LABEL.get(m, m), color=METHOD_COLOR.get(m, None),
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_latency_ms", "").replace("_", " ") for m in metrics])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference latency: edge vs cloud control loop")
    ax.legend()
    return _save(fig, out_dir / "fig8_latency_distribution")


# ── plot 9: AMY shift vs TMYx baseline ───────────────────────────────────────

def plot_amy_shift(ctrl: pd.DataFrame, out_dir: Path) -> Path:
    if "weather_type" not in ctrl.columns:
        print("[plot] skip amy_shift: missing weather_type column")
        return Path()
    has_amy = ctrl["weather_type"].astype(str).str.startswith("AMY").any()
    if not has_amy:
        print("[plot] skip amy_shift: no AMY rows in CSV")
        return Path()
    df = ctrl.copy()
    # Tag year (TMYx → 2009-2023 baseline; AMY_2022 → 2022)
    def year_label(w):
        s = str(w)
        if s.startswith("AMY_"):
            return s.split("_", 1)[1]
        return "TMYx"
    df["year"] = df["weather_type"].apply(year_label)
    # Restrict to HCMC (only city with AMY in this project)
    df = df[df["context_id"].str.contains("hcmc")]
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    years = sorted(df["year"].unique(), key=lambda y: (y != "TMYx", y))

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        sub = df[df["method"] == m].groupby("year")["energy_kwh"].agg(["mean", "std", "count"])
        sub["se"] = sub["std"] / np.sqrt(sub["count"].clip(lower=1))
        sub = sub.reindex(years)
        ax.errorbar(years, sub["mean"], yerr=1.96 * sub["se"],
                    label=METHOD_LABEL.get(m, m), color=METHOD_COLOR.get(m, None),
                    marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("Weather year")
    ax.set_ylabel("Daily energy (kWh)")
    ax.set_title("AMY weather shift: HCMC energy consumption by method across years")
    ax.legend(loc="best", ncol=2, fontsize=9)
    return _save(fig, out_dir / "fig9_amy_shift")


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)
    print(f"[plot] wrote {base.with_suffix('.pdf')}")
    return base.with_suffix(".pdf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Path to results/raw/<run_id> (recursive CSV scan)")
    p.add_argument("--output-dir", default="manuscript_eaai/figures",
                   help="Where to write fig5+ (default: manuscript_eaai/figures)")
    p.add_argument("--plots", default="all",
                   choices=["all", "energy", "comfort", "transfer", "latency", "amy"])
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        sys.exit(f"input-dir not found: {in_dir}")

    ctrl = load_control(in_dir)
    print(f"[plot] loaded {len(ctrl)} rows from {in_dir}")
    lat = load_latency(in_dir)

    if args.plots in ("all", "energy"):
        plot_energy_by_method(ctrl, out_dir)
    if args.plots in ("all", "comfort"):
        plot_comfort_by_method(ctrl, out_dir)
    if args.plots in ("all", "transfer"):
        plot_transfer_efficiency(ctrl, out_dir)
    if args.plots in ("all", "latency"):
        plot_latency(lat, out_dir)
    if args.plots in ("all", "amy"):
        plot_amy_shift(ctrl, out_dir)


if __name__ == "__main__":
    main()
