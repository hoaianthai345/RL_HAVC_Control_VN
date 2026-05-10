# Pre-cloud Codebase Check — `experiments_cloud`

Date: 2026-05-07  
Scope: local preflight check before uploading/running the HVAC RL experiment package on cloud/HPC.

## Summary

The codebase is **ready for dummy smoke execution on cloud**, and the packaged tarball has been cleaned so it no longer includes local `.venv`, cached Python files, stale result CSVs, `.DS_Store`, or local policy artifacts. However, it is **not yet ready for final paper-grade HOT/EnergyPlus experiments** until a concrete HOT adapter and trained-policy provenance are added.

## Commands/checks run

| Check | Status | Evidence |
|---|---:|---|
| Python dependency imports in local venv | Passed | `.venv/bin/python` imports `numpy`, `pandas`, `yaml`, `gymnasium`, `torch`, `stable_baselines3` |
| Python compile | Passed | `.venv/bin/python -m compileall -q src` |
| Shell syntax | Passed | `bash -n experiments_cloud/scripts/*.sh` and `bash -n experiments_cloud/slurm/*.sbatch` |
| Config parse | Passed | `experiment_grid.yaml` and `contexts_min.yaml` parsed; 4 unique contexts found |
| Dummy control smoke | Passed | `src.experiment_runner` wrote `/tmp/precloud_raw/precloud_control_control.csv` |
| Dummy latency smoke | Passed | `src.latency` wrote `/tmp/precloud_raw/precloud_latency.csv` |
| Full smoke script after edits | Passed | `PYTHON=.venv/bin/python bash scripts/run_smoke.sh` wrote run-scoped raw/summary dirs |
| Package creation | Passed | `dist/eaai_hvac_cloud_bundle.tar.gz` created; size `23783 bytes` |
| Package contamination check | Passed after fix | tarball contains no `.venv`, `__pycache__`, `.DS_Store`, `results`, `artifacts`, or `logs` |
| Fresh package smoke import/execute | Passed | extracted package compiled and ran a 4-step static dummy rollout using local venv interpreter |

## Fixes applied during this pre-cloud check

1. **Cloud package cleanup fixed.**
   - Updated `experiments_cloud/scripts/package_for_cloud.sh` to exclude:
     - `.venv`
     - `__pycache__`
     - `*.pyc`
     - `.DS_Store`
     - all `results/`
     - all `artifacts/`
     - all `logs/`
   - Reason: the prior tarball accidentally included nested stale results under `results/results/...` and an audit PPO checkpoint under `artifacts/audit/...`.

2. **Run-scoped script outputs added.**
   - Updated:
     - `scripts/bootstrap_cloud.sh`
     - `scripts/run_smoke.sh`
     - `scripts/run_baselines.sh`
     - `scripts/run_latency.sh`
     - `scripts/run_matrix.sh`
   - These scripts now write to timestamped directories such as:
     - `results/raw/smoke_<timestamp>/`
     - `results/summary/smoke_<timestamp>/`
   - Reason: avoids summary contamination from older CSVs in `results/raw`.

## Remaining blockers before paper-grade cloud/HOT runs

1. **HOT adapter is still not implemented.**
   - `src/hot_adapter_template.py` raises `NotImplementedError`.
   - `mode=hot` requires `HOT_ENV_FACTORY="module:function"`.
   - Before final runs, implement and validate this adapter against HOT/EnergyPlus.

2. **RL-named methods still fall back to heuristics without checkpoints.**
   - `ppo_static`, `ppo_multicontext`, and `edge_only` warn but still run as heuristic proxies if no `--policy-path` is supplied.
   - For final experiments, do not report such outputs as RL results.
   - Recommended next code change: fail closed unless `--allow-heuristic-proxy` is explicitly provided.

3. **Policy provenance is incomplete.**
   - Result rows include `policy_backend` and `trained_policy_loaded`, but not policy SHA-256, training run ID, total timesteps, or config hash.
   - Add these before running final cloud experiments.

4. **Latency benchmark fairness still needs improvement.**
   - `src/latency.py` creates/resets the environment outside the per-method loop.
   - Recommended: reset/recreate env per `(context, method)` or use a fixed observation trace.

5. **Deployment score still needs normalization.**
   - Current score mixes episode energy, comfort rate, action instability, and latency directly.
   - Before paper claims, normalize units and run sensitivity analysis.

## Recommended cloud run sequence now

For cloud smoke only:

```bash
tar -xzf eaai_hvac_cloud_bundle.tar.gz
cd experiments_cloud
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
bash scripts/run_smoke.sh
```

For real HOT runs, first add the adapter:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
python -m src.experiment_runner --mode hot --methods static --seeds 1 --episode-steps 8 --run-name hot_adapter_smoke
```

Only after HOT smoke passes should PPO training and full matrix runs be started.

## Current package artifact

- `dist/eaai_hvac_cloud_bundle.tar.gz`
- Verified clean of local results/artifacts/caches after the packaging fix.
