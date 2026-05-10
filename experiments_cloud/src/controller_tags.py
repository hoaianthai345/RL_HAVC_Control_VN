"""Describe how a benchmark method maps to runnable code paths (paper transparency)."""

from __future__ import annotations


METHODS_USING_TRAINED_POLICY = frozenset({"ppo_static", "ppo_multicontext", "edge_only"})


def uses_trained_policy(method: str, policy_path: str | None) -> bool:
    if not policy_path or not str(policy_path).strip():
        return False
    return method in METHODS_USING_TRAINED_POLICY


def policy_backend(method: str, policy_path: str | None) -> str:
    """High-level classifier for manuscript methods section."""

    if uses_trained_policy(method, policy_path):
        return "stable_baselines3_ppo_zip"
    if method == "static":
        return "fixed_setpoints"
    if method == "ashrae":
        return "threshold_rule_based"
    if method in {"ppo_static", "ppo_multicontext", "edge_only"}:
        return "comfort_proportional_heuristic_stand_in"
    if method == "cloud_only":
        return "comfort_proportional_heuristic_stand_in_delayed"
    if method == "edge_cloud_adaptive":
        return "bias_adaptive_rule_heuristic"
    return "unknown"


def warn_if_misconfigured(method: str, policy_path: str | None) -> None:
    import warnings

    if policy_path and str(policy_path).strip() and method == "edge_cloud_adaptive":
        warnings.warn(
            "Ignoring policy_path for edge_cloud_adaptive: this method "
            "is implemented as AdaptiveEdgeCloudController (bias rule), "
            "not a trained neural policy. Use ppo_static, ppo_multicontext "
            "or edge_only with --policy-path to evaluate SB3 checkpoints.",
            UserWarning,
            stacklevel=3,
        )
    if method in METHODS_USING_TRAINED_POLICY and not (policy_path and str(policy_path).strip()):
        warnings.warn(
            f'Method "{method}" reports without a trained Stable-Baselines3 policy (--policy-path). '
            "The controller falls back to a proportional comfort heuristic (baseline proxy). ",
            UserWarning,
            stacklevel=3,
        )
