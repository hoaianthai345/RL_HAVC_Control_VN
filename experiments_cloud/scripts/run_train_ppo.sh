#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dummy}"
CONTEXT_INDEX="${2:-0}"
SEED="${3:-1}"
DEVICE="${DEVICE:-}"
MULTI_CONTEXT="${MULTI_CONTEXT:-0}"
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"

EXTRA=()
if [ -n "$DEVICE" ]; then
  EXTRA+=(--device "$DEVICE")
fi
if [ "$MULTI_CONTEXT" = "1" ]; then
  EXTRA+=(--multi-context)
fi

"$PYTHON_BIN" -m src.train_ppo \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode "$MODE" \
  --context-index "$CONTEXT_INDEX" \
  --seed "$SEED" \
  "${EXTRA[@]}"
