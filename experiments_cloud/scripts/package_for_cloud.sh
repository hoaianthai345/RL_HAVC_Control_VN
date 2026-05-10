#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
OUT="$DIST_DIR/eaai_hvac_cloud_bundle.tar.gz"

mkdir -p "$DIST_DIR"
cd "$ROOT_DIR"
tar \
  --exclude="experiments_cloud/.venv" \
  --exclude="experiments_cloud/**/__pycache__" \
  --exclude="experiments_cloud/**/*.pyc" \
  --exclude="experiments_cloud/.DS_Store" \
  --exclude="experiments_cloud/**/.DS_Store" \
  --exclude="experiments_cloud/my_hot_adapter_local.py" \
  --exclude="experiments_cloud/results" \
  --exclude="experiments_cloud/artifacts" \
  --exclude="experiments_cloud/logs" \
  -czf "$OUT" experiments_cloud

echo "Wrote $OUT"
