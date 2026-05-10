#!/usr/bin/env bash
#SBATCH --job-name=eaai_hvac
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3

set -euo pipefail

# Rollout/control only — no GPU. For PPO training on GPU use slurm/run_train_array.sbatch.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
PYTHON_BIN="${PYTHON:-python3}"

MODE="${MODE:-hot}"
"$PYTHON_BIN" -m src.experiment_runner \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_min.yaml \
  --mode "$MODE" \
  --context-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir results/raw \
  --run-name "slurm_${MODE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
