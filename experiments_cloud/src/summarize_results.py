from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ensure_dir, load_yaml
from .paper_metrics import (
    adaptation_gain_static_vs_adaptive,
    add_energy_oracle_regret,
    attach_latency_and_rescore,
    load_context_buckets,
    summarize_mean_regret,
    summarize_worst_context_regret,
    transfer_efficiency_energy_ratio,
)


CONTROL_METRICS = [
    "energy_kwh",
    "comfort_violation_rate",
    "mean_temperature_deviation",
    "action_instability",
    "cumulative_reward",
    "deployment_score",
]

LATENCY_METRICS = [
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "max_latency_ms",
]


def mean_ci(series: pd.Series) -> pd.Series:
    count = series.count()
    mean = series.mean()
    if count <= 1:
        return pd.Series({"mean": mean, "ci95": 0.0})
    ci95 = 1.96 * series.std(ddof=1) / (count ** 0.5)
    return pd.Series({"mean": mean, "ci95": ci95})


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    frames = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        part = df.groupby(group_cols)[metric].apply(mean_ci).unstack()
        part = part.rename(columns={"mean": f"{metric}_mean", "ci95": f"{metric}_ci95"})
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).reset_index()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw HVAC experiment results.")
    parser.add_argument("--input-dir", default="results/raw")
    parser.add_argument("--output-dir", default="results/summary")
    parser.add_argument("--config", default="configs/experiment_grid.yaml")
    parser.add_argument("--contexts", default="configs/contexts_min.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)
    cfg = load_yaml(args.config)

    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    control_frames = []
    latency_frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "energy_kwh" in df.columns and "comfort_violation_rate" in df.columns:
            control_frames.append(df)
        elif "energy_kwh" in df.columns:
            control_frames.append(df)
        if "p95_latency_ms" in df.columns:
            latency_frames.append(df)

    buckets = load_context_buckets(args.contexts)

    if control_frames:
        control = pd.concat(control_frames, ignore_index=True)

        latency_concat = pd.concat(latency_frames, ignore_index=True) if latency_frames else pd.DataFrame()
        score_enriched = attach_latency_and_rescore(control, latency_concat, cfg.get("policy_update"))
        if "deployment_score_with_latency" in score_enriched.columns:
            ctl_for_summary = score_enriched.assign(
                deployment_score_joint=score_enriched["deployment_score_with_latency"],
            )
        else:
            ctl_for_summary = score_enriched

        metrics_dyn = list(CONTROL_METRICS)
        if "deployment_score_joint" in ctl_for_summary.columns:
            metrics_dyn = metrics_dyn + ["deployment_score_joint"]
        metrics_dyn = [m for m in metrics_dyn if m in ctl_for_summary.columns]

        summary = summarize(ctl_for_summary, ["mode", "context_id", "method"], metrics_dyn)
        output_path = output_dir / "control_summary.csv"
        summary.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")

        regret_ready = add_energy_oracle_regret(control)
        worst_df = summarize_worst_context_regret(regret_ready)
        mean_df = summarize_mean_regret(regret_ready)
        xfer = transfer_efficiency_energy_ratio(control, buckets)
        adapt_df = adaptation_gain_static_vs_adaptive(control)

        transfer_tables = worst_df.merge(mean_df, on=["mode", "method"], how="outer")
        xfer_path = output_dir / "transfer_metrics.csv"
        transfer_tables.to_csv(xfer_path, index=False)
        print(f"Wrote {xfer_path}")

        if xfer is not None and not xfer.empty:
            xfer_path_te = output_dir / "transfer_efficiency_by_bucket.csv"
            xfer.to_csv(xfer_path_te, index=False)
            print(f"Wrote {xfer_path_te}")

        if adapt_df is not None and not adapt_df.empty:
            adapt_path = output_dir / "adaptation_summary.csv"
            adapt_df.to_csv(adapt_path, index=False)
            print(f"Wrote {adapt_path}")

        if latency_frames and not latency_concat.empty:
            deploy_ctx = score_enriched.groupby(["mode", "context_id", "method"], as_index=False).agg(
                deployment_score_with_latency_mean=("deployment_score_with_latency", "mean"),
            )
            dpath = output_dir / "deployment_score_by_context.csv"
            deploy_ctx.to_csv(dpath, index=False)
            print(f"Wrote {dpath}")

    if latency_frames:
        latency = pd.concat(latency_frames, ignore_index=True)
        summary = summarize(latency, ["mode", "context_id", "method"], LATENCY_METRICS)
        output_path = output_dir / "latency_summary.csv"
        summary.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
