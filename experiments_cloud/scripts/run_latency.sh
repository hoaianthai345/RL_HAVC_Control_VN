#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dummy}"
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
RUN_NAME="latency_${MODE}_$(date -u +%Y%m%d_%H%M%S)"
RAW_DIR="results/raw/${RUN_NAME}"
SUMMARY_DIR="results/summary/${RUN_NAME}"
mkdir -p "$RAW_DIR" "$SUMMARY_DIR"
EXTRA=()
if [ "$MODE" = "dummy" ]; then
  EXTRA+=(--allow-heuristic-proxy)
fi

"$PYTHON_BIN" -m src.latency \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode "$MODE" \
  --output-dir "$RAW_DIR" \
  --run-name "$RUN_NAME" \
  "${EXTRA[@]}"

"$PYTHON_BIN" -m src.summarize_results --input-dir "$RAW_DIR" --output-dir "$SUMMARY_DIR"
echo "Raw results: $RAW_DIR"
echo "Summary: $SUMMARY_DIR"
