from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from .config import ensure_dir, load_contexts, load_yaml
from .controllers import ControllerConfig, build_controller
from .deployment import deployment_score, load_score_weights
from .envs import make_env
from .metrics import rollout
from .provenance import build_policy_provenance, file_sha256, validate_policy_requirements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HVAC control experiments.")
    parser.add_argument("--config", default="configs/experiment_grid.yaml")
    parser.add_argument("--contexts", default="configs/contexts_min.yaml")
    parser.add_argument("--mode", choices=["dummy", "hot"], default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--context-index", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
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
    methods = args.methods or cfg.get("methods", ["static", "ashrae"])
    seeds = args.seeds or cfg.get("seeds", [1])
    episode_steps = args.episode_steps or int(cfg.get("episode_steps", 288))
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir", "results/raw"))
    run_name = args.run_name or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    cloud_delay_ms = float(cfg.get("latency", {}).get("cloud_delay_ms", 80))
    update_interval_steps = int(cfg.get("policy_update", {}).get("update_interval_steps", 96))
    pu_cfg = cfg.get("policy_update", {}) or {}
    score_weights = load_score_weights(pu_cfg)
    validate_policy_requirements(methods, args.policy_path, args.allow_heuristic_proxy)
    config_sha256 = file_sha256(args.config)
    contexts_sha256 = file_sha256(args.contexts)

    result_rows = []
    for context in contexts:
        for seed in seeds:
            for method in methods:
                env = make_env(mode=mode, context=context, seed=seed, episode_steps=episode_steps)
                try:
                    controller = build_controller(
                        ControllerConfig(
                            method=method,
                            cloud_delay_ms=cloud_delay_ms,
                            update_interval_steps=update_interval_steps,
                            policy_path=args.policy_path,
                        )
                    )
                    row = rollout(env, controller, context=context, seed=seed, episode_steps=episode_steps)
                finally:
                    # Close env to stop the EnergyPlus thread + delete temp output dir;
                    # otherwise threads accumulate across the eval matrix and EP segfaults.
                    if hasattr(env, "close"):
                        try:
                            env.close()
                        except Exception:
                            pass
                row["mode"] = mode
                row["method"] = method
                row["run_name"] = run_name
                row["config_path"] = str(args.config)
                row["config_sha256"] = config_sha256
                row["contexts_path"] = str(args.contexts)
                row["contexts_sha256"] = contexts_sha256
                row.update(
                    build_policy_provenance(
                        method=method,
                        policy_path=args.policy_path,
                        training_run_name=args.training_run_name,
                        policy_total_timesteps=args.policy_total_timesteps,
                    )
                )
                row["deployment_score"] = deployment_score(row, score_weights, p95_latency_ms=None)
                result_rows.append(row)
                print(
                    f"{context['id']} seed={seed} method={method} "
                    f"energy={row['energy_kwh']:.3f} comfort={row['comfort_violation_rate']:.3f} "
                    f"reward={row['cumulative_reward']:.3f}"
                )

    if not result_rows:
        raise RuntimeError("No results were produced")

    output_path = Path(output_dir) / f"{run_name}_control.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

