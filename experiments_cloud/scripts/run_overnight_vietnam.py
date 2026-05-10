#!/usr/bin/env python3
"""
Overnight experiment runner with checkpoint/resume support.

Usage:
    python scripts/run_overnight_vietnam.py              # fresh start
    python scripts/run_overnight_vietnam.py              # resume after interruption
    python scripts/run_overnight_vietnam.py --dry-run    # show what would run
    python scripts/run_overnight_vietnam.py --status     # show checkpoint status

The script tracks every job in logs/checkpoint.json.
Re-running the same command skips already-completed jobs.

Interrupt safely with Ctrl+C — the current job finishes before stopping.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── setup path ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.checkpoint import Checkpoint, GracefulExit
from src.config import load_contexts

# ── config ────────────────────────────────────────────────────────────────────
CONTEXTS_FILE  = "configs/contexts_vietnam_multicity.yaml"
CONFIG_FILE    = "configs/experiment_grid.yaml"
CHECKPOINT_FILE = "logs/checkpoint.json"
LOG_DIR        = Path("logs")
RAW_DIR        = Path("results/raw")
SUMMARY_DIR    = Path("results/summary/vietnam_final")

SEEDS           = [1, 2, 3]
TOTAL_TIMESTEPS = 500_000
EPISODE_STEPS   = 288
CLOUD_DELAY_MS  = 80

# Source context indices (HCMC: 0,1,2 | CanTho: 3,4,5)
SOURCE_INDICES  = [0, 1, 2, 3, 4, 5]

PYTHON = sys.executable
ENV = {**os.environ, "HOT_ENV_FACTORY": "src.hot_adapter_vietnam:create_env",
       "PYTHONPATH": str(ROOT)}


# ── helpers ───────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)


def run(cmd: list[str], job_id: str, ckpt: Checkpoint,
        dry_run: bool = False, output: str | None = None) -> bool:
    """
    Run a subprocess command.
    Returns True on success, False on failure.
    Skips if already done in checkpoint.
    """
    if ckpt.is_done(job_id):
        log(f"  SKIP (done): {job_id}")
        return True

    if dry_run:
        log(f"  WOULD RUN: {job_id}")
        log(f"    {' '.join(cmd)}")
        return True

    ckpt.mark_running(job_id)
    log(f"  START: {job_id}")
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            cmd, env=ENV, capture_output=False,
            text=True, check=True,
        )
        elapsed = time.perf_counter() - t0
        ckpt.mark_done(job_id, output=output,
                       meta={"elapsed_sec": round(elapsed, 1)})
        log(f"  DONE  ({elapsed/60:.1f} min): {job_id}")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        ckpt.mark_failed(job_id, error=str(e))
        log(f"  FAIL  ({elapsed/60:.1f} min): {job_id} — {e}")
        return False
    except Exception as e:
        ckpt.mark_failed(job_id, error=str(e))
        log(f"  ERROR: {job_id} — {e}")
        return False


# ── phases ────────────────────────────────────────────────────────────────────

def phase_baselines(ckpt: Checkpoint, stopper: GracefulExit,
                    dry_run: bool, run_dir: Path) -> bool:
    log("=== Phase 1: Baselines (per-context isolation) ===")
    out = str(run_dir / "baselines")
    contexts = load_contexts(CONTEXTS_FILE)
    jobs: list[tuple[str, list[str]]] = []
    for idx, ctx in enumerate(contexts):
        jobs.append((
            f"phase1__baselines_{ctx['id']}",
            [
                PYTHON, "-m", "src.experiment_runner",
                "--config", CONFIG_FILE,
                "--contexts", CONTEXTS_FILE,
                "--mode", "hot",
                "--context-index", str(idx),
                "--methods", "static", "ashrae",
                "--seeds", *[str(s) for s in SEEDS],
                "--episode-steps", str(EPISODE_STEPS),
                "--output-dir", out,
                "--run-name", f"baselines_{ctx['id']}",
            ],
        ))
    return _run_pool(jobs, ckpt, stopper, dry_run, workers=PARALLEL_WORKERS)


def phase_latency(ckpt: Checkpoint, stopper: GracefulExit,
                  dry_run: bool, run_dir: Path) -> bool:
    log("=== Phase 2: Latency (per-context isolation) ===")
    out = str(run_dir / "latency")
    contexts = load_contexts(CONTEXTS_FILE)
    jobs: list[tuple[str, list[str]]] = []
    for idx, ctx in enumerate(contexts):
        jobs.append((
            f"phase2__latency_{ctx['id']}",
            [
                PYTHON, "-m", "src.latency",
                "--config", CONFIG_FILE,
                "--contexts", CONTEXTS_FILE,
                "--mode", "hot",
                "--context-index", str(idx),
                "--methods", "edge_only", "cloud_only",
                "--repetitions", "200",
                "--warmup", "20",
                "--cloud-delay-ms", str(CLOUD_DELAY_MS),
                "--output-dir", out,
                "--run-name", f"latency_{ctx['id']}",
                "--allow-heuristic-proxy",
            ],
        ))
    return _run_pool(jobs, ckpt, stopper, dry_run, workers=PARALLEL_WORKERS)


PARALLEL_WORKERS = int(os.environ.get("OVERNIGHT_PARALLEL", "3"))


def _drain_pool(in_flight: list, ckpt: Checkpoint, block: bool) -> list:
    """Poll subprocesses; mark done/failed in checkpoint as they exit. Returns alive list."""
    alive = []
    for p, job_id, t0 in in_flight:
        rc = p.poll()
        if rc is None:
            alive.append((p, job_id, t0))
            continue
        elapsed = time.perf_counter() - t0
        if rc == 0:
            ckpt.mark_done(job_id, meta={"elapsed_sec": round(elapsed, 1)})
            log(f"  DONE  ({elapsed/60:.1f} min): {job_id}")
        else:
            ckpt.mark_failed(job_id, error=f"exit code {rc}")
            log(f"  FAIL  ({elapsed/60:.1f} min): {job_id} — exit {rc}")
    if block and alive:
        time.sleep(5)
    return alive


def _run_pool(jobs: list[tuple[str, list[str]]], ckpt: Checkpoint,
              stopper: GracefulExit, dry_run: bool, workers: int) -> bool:
    """
    Run a list of (job_id, cmd) pairs with up to `workers` concurrent
    subprocess.Popen, recording status in checkpoint. Skips already-done
    jobs and treats failed ones as pending so re-runs retry them.
    """
    pending = [(jid, cmd) for jid, cmd in jobs if not ckpt.is_done(jid)]
    if dry_run:
        for jid, cmd in pending:
            log(f"  WOULD RUN: {jid}")
        return True
    in_flight: list = []
    ok = True
    while pending or in_flight:
        if stopper.triggered:
            log("  Interrupt — letting in-flight jobs finish before exit")
            while in_flight:
                in_flight = _drain_pool(in_flight, ckpt, block=True)
            return False
        # Fill the pool
        while pending and len(in_flight) < workers:
            jid, cmd = pending.pop(0)
            ckpt.mark_running(jid)
            log(f"  START: {jid}  (pool {len(in_flight)+1}/{workers})")
            p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            in_flight.append((p, jid, time.perf_counter()))
        in_flight = _drain_pool(in_flight, ckpt, block=True)
        # Track failures (don't abort the pool — let other workers continue)
        # Status is reflected in checkpoint; ok flag computed at the end.
    # Final pass: aggregate status
    for jid, _ in jobs:
        if not ckpt.is_done(jid):
            ok = False
    return ok


def phase_ppo_single(ckpt: Checkpoint, stopper: GracefulExit,
                     dry_run: bool) -> bool:
    log(f"=== Phase 3: Single-context PPO training (parallel={PARALLEL_WORKERS}) ===")
    jobs: list[tuple[str, list[str]]] = []
    for ctx_idx in SOURCE_INDICES:
        for seed in SEEDS:
            jobs.append((
                f"phase3__ppo_ctx{ctx_idx}_seed{seed}",
                [
                    PYTHON, "-m", "src.train_ppo",
                    "--config", CONFIG_FILE,
                    "--contexts", CONTEXTS_FILE,
                    "--mode", "hot",
                    "--context-index", str(ctx_idx),
                    "--seed", str(seed),
                    "--total-timesteps", str(TOTAL_TIMESTEPS),
                ],
            ))
    return _run_pool(jobs, ckpt, stopper, dry_run, workers=PARALLEL_WORKERS)


def phase_ppo_multi(ckpt: Checkpoint, stopper: GracefulExit,
                    dry_run: bool) -> bool:
    log(f"=== Phase 4: Multi-context PPO training (parallel={PARALLEL_WORKERS}) ===")
    jobs: list[tuple[str, list[str]]] = []
    for seed in SEEDS:
        jobs.append((
            f"phase4__ppo_multi_seed{seed}",
            [
                PYTHON, "-m", "src.train_ppo",
                "--config", CONFIG_FILE,
                "--contexts", CONTEXTS_FILE,
                "--mode", "hot",
                "--seed", str(seed),
                "--total-timesteps", str(TOTAL_TIMESTEPS),
                "--multi-context",
            ],
        ))
    return _run_pool(jobs, ckpt, stopper, dry_run, workers=PARALLEL_WORKERS)


POLICIES_DIR = Path("artifacts/policies")


def _latest_policy(prefix: str) -> str | None:
    candidates = sorted(POLICIES_DIR.glob(f"{prefix}*.zip"))
    return str(candidates[-1]) if candidates else None


def phase_eval(ckpt: Checkpoint, stopper: GracefulExit,
               dry_run: bool, run_dir: Path) -> bool:
    """
    Split eval into:
      5a. heuristic methods (no policy)         — single call, all contexts
      5b. ppo_multicontext (one policy → all)   — single call, all contexts
      5c. ppo_static + edge_only (per-context)  — one call per context with that
          context's policy; target contexts (no native policy) fall back to the
          multi-context checkpoint as a transfer-evaluation baseline.
    """
    log("=== Phase 5: Full evaluation matrix (per-context isolation, parallel) ===")
    out = str(run_dir / "eval")
    common = [
        "--config", CONFIG_FILE,
        "--contexts", CONTEXTS_FILE,
        "--mode", "hot",
        "--seeds", *[str(s) for s in SEEDS],
        "--episode-steps", str(EPISODE_STEPS),
        "--output-dir", out,
    ]
    contexts = load_contexts(CONTEXTS_FILE)
    ok = True
    multi_policy = _latest_policy("ppo_multicontext_")
    if multi_policy is None:
        log("  WARN: no ppo_multicontext_*.zip in artifacts/policies/ — phase5b will be skipped")

    # 5a — heuristic methods, one subprocess per context
    a_jobs: list[tuple[str, list[str]]] = []
    for idx, ctx in enumerate(contexts):
        a_jobs.append((
            f"phase5a__heuristic_{ctx['id']}",
            [PYTHON, "-m", "src.experiment_runner", *common,
             "--context-index", str(idx),
             "--methods", "static", "ashrae", "cloud_only", "edge_cloud_adaptive",
             "--run-name", f"eval_heuristic_{ctx['id']}"],
        ))
    ok = _run_pool(a_jobs, ckpt, stopper, dry_run, workers=PARALLEL_WORKERS) and ok

    # 5b — ppo_multicontext, one subprocess per context
    if multi_policy is not None:
        b_jobs: list[tuple[str, list[str]]] = []
        for idx, ctx in enumerate(contexts):
            b_jobs.append((
                f"phase5b__multicontext_{ctx['id']}",
                [PYTHON, "-m", "src.experiment_runner", *common,
                 "--context-index", str(idx),
                 "--methods", "ppo_multicontext",
                 "--policy-path", multi_policy,
                 "--training-run-name", Path(multi_policy).stem,
                 "--policy-total-timesteps", str(TOTAL_TIMESTEPS),
                 "--run-name", f"eval_multicontext_{ctx['id']}"],
            ))
        ok = _run_pool(b_jobs, ckpt, stopper, dry_run, workers=PARALLEL_WORKERS) and ok

    # 5c — per-context ppo_static + edge_only
    contexts = load_contexts(CONTEXTS_FILE)
    for ctx_idx, ctx in enumerate(contexts):
        if stopper.triggered:
            return ok
        ctx_id = ctx["id"]
        native = _latest_policy(f"ppo_{ctx_id}_")
        if native is not None:
            policy, tag = native, "native"
        elif multi_policy is not None:
            policy, tag = multi_policy, "transfer"
        else:
            log(f"  SKIP phase5c__{ctx_id}: no policy available")
            ok = False
            continue
        log(f"  ctx {ctx_idx} {ctx_id}: {tag} policy = {policy}")
        ok = run([
            PYTHON, "-m", "src.experiment_runner", *common,
            "--context-index", str(ctx_idx),
            "--methods", "ppo_static", "edge_only",
            "--policy-path", policy,
            "--training-run-name", Path(policy).stem,
            "--policy-total-timesteps", str(TOTAL_TIMESTEPS),
            "--run-name", f"eval_perctx_{ctx_id}_{tag}",
        ], f"phase5c__{ctx_id}", ckpt, dry_run, output=out) and ok

    return ok


def phase_summary(ckpt: Checkpoint, stopper: GracefulExit,
                  dry_run: bool, run_dir: Path) -> bool:
    log("=== Phase 6: Summarize results ===")
    if stopper.triggered:
        return False
    job_id = "phase6__summary"
    return run([
        PYTHON, "-m", "src.summarize_results",
        "--input-dir", str(run_dir),
        "--output-dir", str(SUMMARY_DIR),
    ], job_id, ckpt, dry_run, output=str(SUMMARY_DIR))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print what would run without running")
    parser.add_argument("--status",    action="store_true",
                        help="Show checkpoint status and exit")
    parser.add_argument("--reset",     action="store_true",
                        help="Delete checkpoint and start fresh")
    parser.add_argument("--only-phase", type=int, default=None,
                        help="Run only this phase (1-6)")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset and Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        log("Checkpoint deleted — starting fresh")

    ckpt = Checkpoint(CHECKPOINT_FILE)

    if args.status:
        summary = ckpt.summary()
        log(f"Run ID: {ckpt.run_id}")
        log(f"Checkpoint: {CHECKPOINT_FILE}")
        for status, count in sorted(summary.items()):
            log(f"  {status}: {count} jobs")
        return

    stopper = GracefulExit()
    run_dir = RAW_DIR / ckpt.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log(f"Run ID: {ckpt.run_id}")
    log(f"Checkpoint: {CHECKPOINT_FILE}")
    log(f"Output: {run_dir}")
    if args.dry_run:
        log("DRY RUN — nothing will be executed")

    t_start = time.perf_counter()
    only = args.only_phase

    phases = [
        (1, lambda: phase_baselines(ckpt, stopper, args.dry_run, run_dir)),
        (2, lambda: phase_latency  (ckpt, stopper, args.dry_run, run_dir)),
        (3, lambda: phase_ppo_single(ckpt, stopper, args.dry_run)),
        (4, lambda: phase_ppo_multi (ckpt, stopper, args.dry_run)),
        (5, lambda: phase_eval     (ckpt, stopper, args.dry_run, run_dir)),
        (6, lambda: phase_summary  (ckpt, stopper, args.dry_run, run_dir)),
    ]

    for phase_num, phase_fn in phases:
        if only is not None and phase_num != only:
            continue
        if stopper.triggered:
            break
        phase_fn()

    elapsed = time.perf_counter() - t_start
    summary = ckpt.summary()
    log(f"=== FINISHED in {elapsed/3600:.1f}h ===")
    log(f"Jobs: {summary}")

    remaining = summary.get("pending", 0) + summary.get("running", 0)
    if remaining > 0 or stopper.triggered:
        log("Re-run the same command to resume remaining jobs.")
    else:
        log(f"All done! Results in: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
