"""Deployment-oriented scoring used for policy_acceptance gates and reporting."""

from __future__ import annotations

from typing import Any, Mapping


def load_score_weights(policy_update_cfg: Mapping[str, Any] | None) -> dict[str, float]:
    """Weights for deployment_score; lower composite score implies better posture when metrics are penalized."""

    pu = policy_update_cfg or {}
    raw = pu.get(
        "deployment_score_weights",
        {
            "energy_kwh": 1.0,
            "comfort_violation_rate": 8.0,
            "action_instability": 2.0,
            "p95_latency_ms_penalty_per_ms": 0.004,
        },
    )
    return {str(k): float(v) for k, v in raw.items()}


def deployment_score(
    row: Mapping[str, Any],
    weights: Mapping[str, float],
    p95_latency_ms: float | None = None,
) -> float:
    """Scalar deployment cost: weighted energy, discomfort, jitter, optional latency."""

    energy = float(row.get("energy_kwh", 0.0))
    comfort = float(row.get("comfort_violation_rate", 0.0))
    instab = float(row.get("action_instability", 0.0))

    latency = row.get("p95_latency_ms")
    if latency is not None:
        latency = float(latency)
    elif p95_latency_ms is not None:
        latency = float(p95_latency_ms)
    else:
        latency = 0.0

    w_e = float(weights.get("energy_kwh", 1.0))
    w_c = float(weights.get("comfort_violation_rate", 1.0))
    w_i = float(weights.get("action_instability", 1.0))
    w_l = float(weights.get("p95_latency_ms_penalty_per_ms", 0.0))
    return w_e * energy + w_c * comfort + w_i * instab + w_l * latency


def comfort_regression_violates(
    candidate_comfort: float,
    incumbent_comfort: float,
    max_relative_increase: float,
) -> bool:
    """True if candidate comfort violation rate worsens versus incumbent beyond threshold."""

    if candidate_comfort > incumbent_comfort + float(max_relative_increase):
        return True
    return False
