#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
RUN_NAME="smoke_$(date -u +%Y%m%d_%H%M%S)"
RAW_DIR="results/raw/${RUN_NAME}"
SUMMARY_DIR="results/summary/${RUN_NAME}"
mkdir -p "$RAW_DIR" "$SUMMARY_DIR"

"$PYTHON_BIN" -m src.experiment_runner \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode dummy \
  --methods static ashrae edge_cloud_adaptive \
  --seeds 1 \
  --episode-steps 32 \
  --output-dir "$RAW_DIR" \
  --run-name "${RUN_NAME}_control"

"$PYTHON_BIN" -m src.latency \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode dummy \
  --methods edge_only cloud_only \
  --repetitions 20 \
  --warmup 2 \
  --cloud-delay-ms 5 \
  --output-dir "$RAW_DIR" \
  --run-name "${RUN_NAME}_latency" \
  --allow-heuristic-proxy

"$PYTHON_BIN" -m src.summarize_results --input-dir "$RAW_DIR" --output-dir "$SUMMARY_DIR"
echo "Raw results: $RAW_DIR"
echo "Summary: $SUMMARY_DIR"
