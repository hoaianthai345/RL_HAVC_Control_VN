#!/usr/bin/env python3
"""
Phase-4 style AMY eval: take pre-trained policies and evaluate on AMY weather
years (2022/2023/2024) for HCMC source buildings. Produces CSV output the
summarize step can consume to compute adaptation gain (TMYx vs AMY).

Pre-requisite: AMY contexts file (run scripts/gen_amy_contexts.py first).

Usage:
    python scripts/run_amy_eval.py \
        --amy-contexts configs/contexts_vietnam_amy.yaml \
        --output-dir results/raw/run_<id>/amy_eval \
        --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.config import load_contexts

PYTHON = sys.executable
POLICIES_DIR = Path("artifacts/policies")
ENV = {**os.environ,
       "HOT_ENV_FACTORY": "src.hot_adapter_vietnam:create_env",
       "PYTHONPATH": str(ROOT)}


def latest_policy(prefix: str) -> str | None:
    cands = sorted(POLICIES_DIR.glob(f"{prefix}*.zip"))
    return str(cands[-1]) if cands else None


def native_policy_for(amy_id: str) -> str | None:
    """Map an AMY context id (vn_hcmc_<bldg>_amy<year>) back to the TMYx-trained policy."""
    base_id = amy_id.rsplit("_amy", 1)[0]
    return latest_policy(f"ppo_{base_id}_")


def run(cmd: list[str], label: str) -> bool:
    print(f"[amy] START: {label}", flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, env=ENV, check=True)
        print(f"[amy] DONE  ({(time.perf_counter()-t0)/60:.1f} min): {label}", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[amy] FAIL  {label}: {e}", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--amy-contexts", default="configs/contexts_vietnam_amy.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--episode-steps", type=int, default=288)
    p.add_argument("--config", default="configs/experiment_grid.yaml")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    common = [
        "--config", args.config,
        "--contexts", args.amy_contexts,
        "--mode", "hot",
        "--seeds", *[str(s) for s in args.seeds],
        "--episode-steps", str(args.episode_steps),
        "--output-dir", str(out),
    ]

    contexts = load_contexts(args.amy_contexts)
    print(f"[amy] {len(contexts)} AMY contexts × {len(args.seeds)} seeds")

    ok = True

    # 1) heuristic baselines on AMY — establishes the "no-adaptation" reference
    ok &= run([
        PYTHON, "-m", "src.experiment_runner", *common,
        "--methods", "static", "ashrae", "cloud_only", "edge_cloud_adaptive",
        "--run-name", "amy_heuristic",
    ], "amy_heuristic")

    # 2) ppo_static / edge_only with the matching TMYx-trained native policy per context
    for idx, ctx in enumerate(contexts):
        policy = native_policy_for(ctx["id"])
        if policy is None:
            print(f"[amy] SKIP {ctx['id']}: no native policy in {POLICIES_DIR}")
            ok = False
            continue
        ok &= run([
            PYTHON, "-m", "src.experiment_runner", *common,
            "--context-index", str(idx),
            "--methods", "ppo_static", "edge_only",
            "--policy-path", policy,
            "--training-run-name", Path(policy).stem,
            "--run-name", f"amy_perctx_{ctx['id']}",
        ], f"amy_perctx_{ctx['id']}")

    # 3) multi-context policy on AMY (transfer ability under year shift)
    multi = latest_policy("ppo_multicontext_")
    if multi is None:
        print("[amy] SKIP multicontext: no ppo_multicontext_*.zip available")
        ok = False
    else:
        ok &= run([
            PYTHON, "-m", "src.experiment_runner", *common,
            "--methods", "ppo_multicontext",
            "--policy-path", multi,
            "--training-run-name", Path(multi).stem,
            "--run-name", "amy_multicontext",
        ], "amy_multicontext")

    print(f"[amy] FINISHED, all_ok={ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
