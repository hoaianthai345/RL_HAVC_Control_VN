#!/usr/bin/env bash
# ============================================================
# Overnight full experiment run — Vietnam multi-city HVAC RL
# Chạy: bash scripts/run_overnight_vietnam.sh 2>&1 | tee logs/overnight_$(date +%Y%m%d).log
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env"
export PYTHONPATH="$PWD:$PYTHONPATH"
PYTHON="${PYTHON:-python3}"

CONTEXTS="configs/contexts_vietnam_multicity.yaml"
CONFIG="configs/experiment_grid.yaml"
SEEDS="1 2 3"
STEPS=500000    # 500k timesteps cho kết quả publishable
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase 1: Baselines (không cần GPU) ───────────────────────────────────────
log "=== Phase 1: Baselines ==="
RUN="baselines_$(date +%Y%m%d_%H%M%S)"
$PYTHON -m src.experiment_runner \
  --config "$CONFIG" \
  --contexts "$CONTEXTS" \
  --mode hot \
  --methods static ashrae \
  --seeds $SEEDS \
  --episode-steps 288 \
  --output-dir "results/raw/${RUN}" \
  --run-name "${RUN}"
log "Baselines done → results/raw/${RUN}"

# ── Phase 2: Latency experiments ─────────────────────────────────────────────
log "=== Phase 2: Latency ==="
RUN_LAT="latency_$(date +%Y%m%d_%H%M%S)"
$PYTHON -m src.latency \
  --config "$CONFIG" \
  --contexts "$CONTEXTS" \
  --mode hot \
  --methods edge_only cloud_only \
  --repetitions 200 \
  --warmup 20 \
  --cloud-delay-ms 80 \
  --output-dir "results/raw/${RUN_LAT}" \
  --run-name "${RUN_LAT}" \
  --allow-heuristic-proxy
log "Latency done → results/raw/${RUN_LAT}"

# ── Phase 3: Single-context PPO training (source cities only) ────────────────
log "=== Phase 3: Single-context PPO training ==="
# Source contexts: index 0-5 (HCMC: 0,1,2 | CanTho: 3,4,5)
SOURCE_INDICES="0 1 2 3 4 5"

for CTX_IDX in $SOURCE_INDICES; do
  for SEED in $SEEDS; do
    log "  Training PPO ctx=${CTX_IDX} seed=${SEED}"
    $PYTHON -m src.train_ppo \
      --config "$CONFIG" \
      --contexts "$CONTEXTS" \
      --mode hot \
      --context-index "$CTX_IDX" \
      --seed "$SEED" \
      --total-timesteps "$STEPS" 2>&1 | tail -3
  done
done
log "Single-context PPO done"

# ── Phase 4: Multi-context PPO training ──────────────────────────────────────
log "=== Phase 4: Multi-context PPO training ==="
for SEED in $SEEDS; do
  log "  Multi-context seed=${SEED}"
  MULTI_CONTEXT=1 $PYTHON -m src.train_ppo \
    --config "$CONFIG" \
    --contexts "$CONTEXTS" \
    --mode hot \
    --seed "$SEED" \
    --total-timesteps "$STEPS" \
    --multi-context 2>&1 | tail -3
done
log "Multi-context PPO done"

# ── Phase 5: Full evaluation matrix ──────────────────────────────────────────
log "=== Phase 5: Full evaluation ==="
RUN_EVAL="eval_$(date +%Y%m%d_%H%M%S)"
$PYTHON -m src.experiment_runner \
  --config "$CONFIG" \
  --contexts "$CONTEXTS" \
  --mode hot \
  --methods static ashrae ppo_static ppo_multicontext edge_only cloud_only edge_cloud_adaptive \
  --seeds $SEEDS \
  --episode-steps 288 \
  --output-dir "results/raw/${RUN_EVAL}" \
  --run-name "${RUN_EVAL}"
log "Full eval done → results/raw/${RUN_EVAL}"

# ── Phase 6: Summarize ───────────────────────────────────────────────────────
log "=== Phase 6: Summarize results ==="
$PYTHON -m src.summarize_results \
  --input-dir results/raw \
  --output-dir results/summary/vietnam_final
log "Summary done → results/summary/vietnam_final/"

log "=== OVERNIGHT RUN COMPLETE ==="
log "Key outputs:"
log "  results/summary/vietnam_final/control_summary.csv"
log "  results/summary/vietnam_final/latency_summary.csv"
log "  results/summary/vietnam_final/transfer_metrics.csv"
