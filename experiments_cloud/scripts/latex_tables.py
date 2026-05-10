#!/usr/bin/env python3
"""
Generate LaTeX-ready table rows for the manuscript directly from raw CSVs.

Reads results/raw/<run_id>/**/*.csv (recursive) and prints fragments you can
paste into main.tex. Each table follows the placeholder structure already in
the manuscript so cells line up without re-editing column counts.

Tables emitted (to stdout, also written to files when --output-dir is given):
    table3_control.tex     Energy / comfort by method × context (mean ± 95% CI)
    table4_transfer.tex    Source-vs-target energy ratio per method
    table5_latency.tex     Mean / p50 / p95 / max latency per method

Usage:
    python scripts/latex_tables.py \
        --input-dir results/raw/run_20260508_063450 \
        --output-dir manuscript_eaai/tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_LABEL = {
    "static": "Static",
    "ashrae": "ASHRAE",
    "ppo_static": "Single-context PPO",
    "ppo_multicontext": "Multi-context PPO",
    "edge_only": "Edge-only PPO",
    "cloud_only": "Cloud-only PPO",
    "edge_cloud_adaptive": "Edge-cloud adaptive RL",
}
METHOD_ORDER = list(METHOD_LABEL.keys())

CITY_PREFIX = {"vn_hcmc": "HCMC", "vn_cantho": "CanTho",
               "vn_danang": "DaNang", "vn_hanoi": "Hanoi"}


def short_ctx(ctx_id: str) -> str:
    parts = ctx_id.split("_")
    if len(parts) >= 3:
        city = CITY_PREFIX.get(f"{parts[0]}_{parts[1]}", parts[1])
        bldg = parts[2].replace("officesmall", "OffSm").replace("officemedium", "OffMd").replace("hospital", "Hosp")
        return f"{city}/{bldg}"
    return ctx_id


def load(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    paths = sorted(input_dir.rglob("*.csv"))
    ctrl, lat = [], []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "energy_kwh" in df.columns and "comfort_violation_rate" in df.columns:
            ctrl.append(df)
        if "p95_latency_ms" in df.columns:
            lat.append(df)
    if not ctrl:
        sys.exit(f"No control CSVs under {input_dir}")
    return pd.concat(ctrl, ignore_index=True), (pd.concat(lat, ignore_index=True) if lat else None)


def fmt_mean_ci(s: pd.Series) -> str:
    mu = s.mean()
    n = len(s)
    if n <= 1 or s.std(ddof=1) == 0 or pd.isna(s.std(ddof=1)):
        return f"{mu:.2f}"
    ci = 1.96 * s.std(ddof=1) / np.sqrt(n)
    return f"{mu:.2f} $\\pm$ {ci:.2f}"


def table3_control(ctrl: pd.DataFrame) -> str:
    """Energy + comfort + cumulative_reward by method × context."""
    df = ctrl[ctrl.get("weather_type", "TMYx") == "TMYx"] if "weather_type" in ctrl else ctrl
    contexts = sorted(df["context_id"].unique(), key=lambda c: (c.split("_")[1], c))
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    lines = ["% --- Auto-generated: Table 3 control results --------------",
             "% energy_kwh / comfort_violation_rate (mean ± 95% CI across seeds)",
             "% Columns: Context | Method | Energy (kWh) | Comfort viol. | Reward"]
    for ctx in contexts:
        for m in methods:
            sub = df[(df["context_id"] == ctx) & (df["method"] == m)]
            if sub.empty:
                continue
            row = (
                f"{short_ctx(ctx)} & {METHOD_LABEL[m]} & "
                f"{fmt_mean_ci(sub['energy_kwh'])} & "
                f"{fmt_mean_ci(sub['comfort_violation_rate'])} & "
                f"{fmt_mean_ci(sub['cumulative_reward'])} \\\\"
            )
            lines.append(row)
    return "\n".join(lines) + "\n"


def table4_transfer(ctrl: pd.DataFrame) -> str:
    """Per-method target/source energy ratio (transfer efficiency)."""
    if "transfer_bucket" not in ctrl.columns:
        bucket = {"hcmc": "source", "cantho": "source", "danang": "target", "hanoi": "target"}
        ctrl = ctrl.copy()
        ctrl["transfer_bucket"] = ctrl["context_id"].apply(
            lambda c: bucket.get(c.split("_")[1], "unknown"))
    df = ctrl[ctrl.get("weather_type", "TMYx") == "TMYx"] if "weather_type" in ctrl else ctrl
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    lines = ["% --- Auto-generated: Table 4 transfer efficiency ----------",
             "% Columns: Method | Source energy | Target energy | Ratio (T/S) | Comfort viol. (target)"]
    for m in methods:
        src = df[(df["method"] == m) & (df["transfer_bucket"] == "source")]
        tgt = df[(df["method"] == m) & (df["transfer_bucket"] == "target")]
        if src.empty or tgt.empty:
            continue
        ratio = tgt["energy_kwh"].mean() / src["energy_kwh"].mean()
        row = (
            f"{METHOD_LABEL[m]} & "
            f"{fmt_mean_ci(src['energy_kwh'])} & "
            f"{fmt_mean_ci(tgt['energy_kwh'])} & "
            f"{ratio:.3f} & "
            f"{fmt_mean_ci(tgt['comfort_violation_rate'])} \\\\"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def table5_latency(lat: pd.DataFrame | None) -> str:
    if lat is None or lat.empty:
        return "% Table 5 latency: no latency CSV found, skipped\n"
    cols = [c for c in ["mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "max_latency_ms"]
            if c in lat.columns]
    methods = [m for m in METHOD_ORDER if m in lat["method"].unique()]

    lines = ["% --- Auto-generated: Table 5 latency ----------------------",
             "% Columns: Method | mean (ms) | p50 (ms) | p95 (ms) | max (ms)"]
    for m in methods:
        sub = lat[lat["method"] == m]
        if sub.empty:
            continue
        cells = [METHOD_LABEL[m]] + [f"{sub[c].mean():.2f}" for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def amy_summary(ctrl: pd.DataFrame) -> str:
    """Optional: AMY years vs TMYx mean energy/comfort per method."""
    if "weather_type" not in ctrl.columns or not ctrl["weather_type"].astype(str).str.startswith("AMY").any():
        return "% AMY summary: no AMY rows, skipped\n"
    df = ctrl[ctrl["context_id"].str.contains("hcmc")].copy()
    def year(w):
        s = str(w);
        return s.split("_", 1)[1] if s.startswith("AMY_") else "TMYx"
    df["year"] = df["weather_type"].apply(year)
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    years = sorted(df["year"].unique(), key=lambda y: (y != "TMYx", y))

    lines = ["% --- Auto-generated: AMY adaptation summary -------------",
             "% Columns: Method | " + " | ".join(years) + " (energy kWh)"]
    for m in methods:
        cells = [METHOD_LABEL[m]]
        for y in years:
            sub = df[(df["method"] == m) & (df["year"] == y)]["energy_kwh"]
            cells.append(fmt_mean_ci(sub) if len(sub) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    ctrl, lat = load(Path(args.input_dir))
    print(f"% loaded {len(ctrl)} control rows, {0 if lat is None else len(lat)} latency rows")

    sections = {
        "table3_control.tex":  table3_control(ctrl),
        "table4_transfer.tex": table4_transfer(ctrl),
        "table5_latency.tex":  table5_latency(lat),
        "amy_summary.tex":     amy_summary(ctrl),
    }

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name, body in sections.items():
        print(f"\n% ============== {name} ==============")
        print(body, end="")
        if out_dir:
            (out_dir / name).write_text(body)
            print(f"% (written to {out_dir / name})")


if __name__ == "__main__":
    main()
