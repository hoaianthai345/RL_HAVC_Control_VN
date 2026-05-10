from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

from .config import ensure_dir, load_contexts, load_yaml
from .envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for HVAC control.")
    parser.add_argument("--config", default="configs/experiment_grid.yaml")
    parser.add_argument("--contexts", default="configs/contexts_min.yaml")
    parser.add_argument("--mode", choices=["dummy", "hot"], default=None)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--device",
        default=None,
        choices=("auto", "cpu", "cuda"),
        help="Compute device for PPO (default: config training.device or 'auto').",
    )
    parser.add_argument(
        "--multi-context",
        action="store_true",
        help="Train simultaneously on one Gymnasium env per context via DummyVecEnv.",
    )
    return parser.parse_args()


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    args = parse_args()
    cfg = load_yaml(args.config)
    contexts = load_contexts(args.contexts)
    mode = args.mode or cfg.get("mode", "dummy")
    training = cfg.get("training", {})
    total_timesteps = args.total_timesteps or int(training.get("total_timesteps", 20000))
    episode_steps = int(cfg.get("episode_steps", 288))
    artifact_dir = ensure_dir(Path(args.artifact_dir or cfg.get("artifact_dir", "artifacts")) / "policies")

    if args.multi_context:
        run_name = args.run_name or datetime.utcnow().strftime(
            f"ppo_multicontext_{len(contexts)}ctx_%Y%m%d_%H%M%S"
        )
        env_fns = [
            partial(make_env, mode, contexts[i], args.seed + i, episode_steps) for i in range(len(contexts))
        ]
        env = DummyVecEnv(list(env_fns))
    else:
        context = contexts[args.context_index]
        run_name = args.run_name or datetime.utcnow().strftime(f"ppo_{context['id']}_%Y%m%d_%H%M%S")
        env = make_env(mode=mode, context=context, seed=args.seed, episode_steps=episode_steps)

    device = args.device or str(training.get("device", "auto"))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(training.get("learning_rate", 5e-4)),
        n_steps=int(training.get("n_steps", 128)),
        batch_size=int(training.get("batch_size", 128)),
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        clip_range=float(training.get("clip_range", 0.15)),
        seed=args.seed,
        device=device,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    output_path = artifact_dir / run_name
    model.save(str(output_path))
    print(f"Wrote {output_path}.zip")


if __name__ == "__main__":
    main()

