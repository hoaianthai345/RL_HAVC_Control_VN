from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .controller_tags import METHODS_USING_TRAINED_POLICY, policy_backend, uses_trained_policy


def file_sha256(path: str | Path | None) -> str:
    """Return SHA-256 for a file, or an empty string when no path is provided."""

    if path is None or not str(path).strip():
        return ""
    resolved = Path(path)
    h = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_policy_requirements(methods: list[str], policy_path: str | None, allow_heuristic_proxy: bool) -> None:
    """Fail closed for RL-named methods unless a checkpoint or explicit proxy override is supplied."""

    missing = sorted(
        method
        for method in set(methods)
        if method in METHODS_USING_TRAINED_POLICY and not uses_trained_policy(method, policy_path)
    )
    if missing and not allow_heuristic_proxy:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Refusing to run RL-named method(s) without a trained policy checkpoint: "
            f"{joined}. Provide --policy-path for paper-grade runs, or pass "
            "--allow-heuristic-proxy only for smoke/debug runs."
        )


def build_policy_provenance(
    method: str,
    policy_path: str | None,
    training_run_name: str | None = None,
    policy_total_timesteps: int | None = None,
) -> dict[str, Any]:
    """Result-row fields that make policy/checkpoint provenance explicit."""

    loaded = bool(uses_trained_policy(method, policy_path))
    return {
        "policy_backend": policy_backend(method, policy_path),
        "trained_policy_loaded": loaded,
        "policy_path": str(policy_path or ""),
        "policy_sha256": file_sha256(policy_path) if loaded else "",
        "training_run_name": str(training_run_name or ""),
        "policy_total_timesteps": int(policy_total_timesteps) if policy_total_timesteps is not None else "",
    }
