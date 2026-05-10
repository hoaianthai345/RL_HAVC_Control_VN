#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
PYTHON_BIN="${PYTHON:-python3}"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

mkdir -p results/raw results/summary artifacts/policies logs
RUN_NAME="bootstrap_smoke_$(date -u +%Y%m%d_%H%M%S)"
RAW_DIR="results/raw/${RUN_NAME}"
SUMMARY_DIR="results/summary/${RUN_NAME}"
mkdir -p "$RAW_DIR" "$SUMMARY_DIR"
"$PYTHON_BIN" -m src.experiment_runner \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode dummy \
  --methods static \
  --seeds 1 \
  --episode-steps 16 \
  --output-dir "$RAW_DIR" \
  --run-name "$RUN_NAME"

"$PYTHON_BIN" -m src.summarize_results --input-dir "$RAW_DIR" --output-dir "$SUMMARY_DIR"
echo "Raw results: $RAW_DIR"
echo "Summary: $SUMMARY_DIR"
