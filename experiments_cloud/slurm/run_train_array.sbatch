#!/usr/bin/env bash
#SBATCH --job-name=eaai_hvac_train
#SBATCH --output=logs/slurm_train_%A_%a.out
#SBATCH --error=logs/slurm_train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
PYTHON_BIN="${PYTHON:-python3}"

MODE="${MODE:-dummy}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON_BIN" -m src.train_ppo \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode "$MODE" \
  --context-index "$SLURM_ARRAY_TASK_ID" \
  --seed "$SEED" \
  --device "$DEVICE"
