"""Post-hoc metrics for transfer and adaptation narratives (oracle regrets, summaries)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import load_yaml
from .deployment import deployment_score, load_score_weights


def load_context_buckets(contexts_path: str) -> dict[str, str]:
    data = load_yaml(contexts_path)
    buckets: dict[str, str] = {}
    for ctx in data.get("contexts", []) or []:
        if not isinstance(ctx, dict):
            continue
        cid = ctx.get("id")
        if not cid:
            continue
        buckets[str(cid)] = str(ctx.get("transfer_bucket", "") or "").strip().lower()
    return buckets


def add_energy_oracle_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row gap between method energy and the best achievable energy at the same scenario draw."""

    out = df.copy()
    keys = ["mode", "context_id", "seed"]
    out["oracle_min_energy_kwh"] = out.groupby(keys)["energy_kwh"].transform("min")
    out["energy_regret_vs_oracle_kwh"] = out["energy_kwh"] - out["oracle_min_energy_kwh"]
    return out


def summarize_worst_context_regret(regret_df: pd.DataFrame) -> pd.DataFrame:
    worst_seed = regret_df.groupby(["mode", "method", "seed"], as_index=False)["energy_regret_vs_oracle_kwh"].max()

    def _seed_ci95(s: pd.Series) -> float:
        return 1.96 * s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0

    return worst_seed.groupby(["mode", "method"], as_index=False).agg(
        worst_context_regret_energy_kwh_mean=("energy_regret_vs_oracle_kwh", "mean"),
        worst_context_regret_energy_kwh_ci95=("energy_regret_vs_oracle_kwh", _seed_ci95),
    )


def summarize_mean_regret(regret_df: pd.DataFrame) -> pd.DataFrame:
    return regret_df.groupby(["mode", "method"], as_index=False).agg(
        mean_energy_regret_vs_oracle_kwh=("energy_regret_vs_oracle_kwh", "mean"),
    )


def transfer_efficiency_energy_ratio(df: pd.DataFrame, bucket_by_context: dict[str, str]) -> pd.DataFrame | None:
    """Mean target-bucket energy divided by mean source-bucket energy (lower global energy is better)."""

    if not bucket_by_context or not any(bucket_by_context.values()):
        return None
    work = df.copy()
    work["transfer_bucket"] = work["context_id"].map(lambda c: bucket_by_context.get(str(c), ""))
    src = work[work["transfer_bucket"] == "source"]
    tgt = work[work["transfer_bucket"] == "target"]
    if src.empty or tgt.empty:
        return None

    rows: list[dict[str, Any]] = []
    for mode in work["mode"].unique():
        for method in work["method"].unique():
            s = src[(src["mode"] == mode) & (src["method"] == method)]["energy_kwh"].mean()
            t = tgt[(tgt["mode"] == mode) & (tgt["method"] == method)]["energy_kwh"].mean()
            if pd.isna(s) or pd.isna(t) or s == 0:
                continue
            rows.append(
                {
                    "mode": mode,
                    "method": method,
                    "transfer_efficiency_energy_target_over_source": float(t / s),
                }
            )
    return pd.DataFrame(rows) if rows else None


def adaptation_gain_static_vs_adaptive(df: pd.DataFrame, static="static", adaptive="edge_cloud_adaptive") -> pd.DataFrame | None:
    """Energy difference static minus adaptive (positive => adaptive uses less energy)."""

    need = {static, adaptive}
    if not need.issubset(set(df["method"].unique())):
        return None
    sub = df[df["method"].isin(need)].copy()
    pivot = sub.pivot_table(
        index=["mode", "context_id", "seed"],
        columns="method",
        values="energy_kwh",
        aggfunc="first",
    )
    if static not in pivot.columns or adaptive not in pivot.columns:
        return None
    pivot["adaptation_gain_energy_kwh"] = pivot[static] - pivot[adaptive]

    def _seed_ci95(s: pd.Series) -> float:
        return 1.96 * s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0

    flat = pivot.reset_index()
    out = flat.groupby("mode", as_index=False).agg(
        adaptation_gain_energy_kwh_mean=("adaptation_gain_energy_kwh", "mean"),
        adaptation_gain_energy_kwh_ci95=("adaptation_gain_energy_kwh", _seed_ci95),
    )
    return out


def latency_lookup_mean(df_lat: pd.DataFrame) -> pd.Series:
    g = df_lat.groupby(["mode", "context_id", "method"], as_index=True)["p95_latency_ms"].mean()
    return g


def attach_latency_and_rescore(control: pd.DataFrame, latency: pd.DataFrame, policy_update: dict[str, Any] | None) -> pd.DataFrame:
    if latency.empty or "p95_latency_ms" not in latency.columns:
        control = control.copy()
        weights = load_score_weights(policy_update)
        control["deployment_score_with_latency"] = control.apply(
            lambda r: deployment_score(r.to_dict(), weights, None), axis=1
        )
        return control

    lut = latency_lookup_mean(latency)
    weights = load_score_weights(policy_update)

    def lat_for_row(r: pd.Series) -> float:
        key = (r["mode"], r["context_id"], r["method"])
        if key in lut.index:
            return float(lut.loc[key])
        return 0.0

    enriched = control.copy()
    enriched["p95_latency_ms_merged"] = enriched.apply(lat_for_row, axis=1)
    enriched["deployment_score_with_latency"] = enriched.apply(
        lambda r: deployment_score(r.to_dict(), weights, float(r["p95_latency_ms_merged"])),
        axis=1,
    )
    return enriched
