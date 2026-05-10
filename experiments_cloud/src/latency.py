from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ensure_dir, load_contexts, load_yaml
from .controllers import ControllerConfig, build_controller
from .envs import make_env
from .provenance import build_policy_provenance, file_sha256, validate_policy_requirements


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure controller inference latency.")
    parser.add_argument("--config", default="configs/experiment_grid.yaml")
    parser.add_argument("--contexts", default="configs/contexts_min.yaml")
    parser.add_argument("--mode", choices=["dummy", "hot"], default=None)
    parser.add_argument("--context-index", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["edge_only", "cloud_only", "edge_cloud_adaptive"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cloud-delay-ms", type=float, default=None)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--allow-heuristic-proxy",
        action="store_true",
        help="Allow RL-named methods to fall back to heuristic proxy controllers. Use only for smoke/debug runs.",
    )
    parser.add_argument("--training-run-name", default=None)
    parser.add_argument("--policy-total-timesteps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    contexts = load_contexts(args.contexts)
    if args.context_index is not None:
        contexts = [contexts[args.context_index]]
    mode = args.mode or cfg.get("mode", "dummy")
    latency_cfg = cfg.get("latency", {})
    repetitions = args.repetitions or int(latency_cfg.get("repetitions", 1000))
    warmup = args.warmup or int(latency_cfg.get("warmup", 50))
    cloud_delay_ms = args.cloud_delay_ms if args.cloud_delay_ms is not None else float(latency_cfg.get("cloud_delay_ms", 80))
    update_interval_steps = int(cfg.get("policy_update", {}).get("update_interval_steps", 96))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "results/raw"))
    run_name = args.run_name or datetime.utcnow().strftime("latency_%Y%m%d_%H%M%S")
    validate_policy_requirements(args.methods, args.policy_path, args.allow_heuristic_proxy)
    config_sha256 = file_sha256(args.config)
    contexts_sha256 = file_sha256(args.contexts)

    rows = []
    for context in contexts:
        for method in args.methods:
            # Recreate the environment per method so stateful controllers/adapters
            # see the same initial seeded trajectory during latency measurement.
            env = make_env(mode=mode, context=context, seed=args.seed, episode_steps=repetitions + warmup + 1)
            try:
                obs, info = env.reset(seed=args.seed)
                controller = build_controller(
                    ControllerConfig(
                        method=method,
                        cloud_delay_ms=cloud_delay_ms,
                        update_interval_steps=update_interval_steps,
                        policy_path=args.policy_path,
                    )
                )
                controller.reset()
                timings = []
                last_info = info
                for i in range(repetitions + warmup):
                    start = time.perf_counter()
                    action = controller.act(obs, last_info)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    obs, _, terminated, truncated, last_info = env.step(action)
                    if i >= warmup:
                        timings.append(elapsed_ms)
                    if terminated or truncated:
                        obs, last_info = env.reset(seed=args.seed)
            finally:
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
            row = {
                "run_name": run_name,
                "mode": mode,
                "context_id": context["id"],
                "method": method,
                "config_path": str(args.config),
                "config_sha256": config_sha256,
                "contexts_path": str(args.contexts),
                "contexts_sha256": contexts_sha256,
                **build_policy_provenance(
                    method=method,
                    policy_path=args.policy_path,
                    training_run_name=args.training_run_name,
                    policy_total_timesteps=args.policy_total_timesteps,
                ),
                "repetitions": len(timings),
                "mean_latency_ms": statistics.mean(timings),
                "p50_latency_ms": percentile(timings, 50),
                "p95_latency_ms": percentile(timings, 95),
                "max_latency_ms": max(timings),
            }
            rows.append(row)
            print(
                f"{context['id']} method={method} "
                f"p50={row['p50_latency_ms']:.3f}ms p95={row['p95_latency_ms']:.3f}ms"
            )

    output_path = Path(output_dir) / f"{run_name}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
