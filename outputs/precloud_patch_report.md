# Pre-cloud Patch Report

Date: 2026-05-07  
Scope: safety/reproducibility fixes before uploading `experiments_cloud` to cloud/HPC.

## Changes applied

### 1. Fail closed for RL-named methods without checkpoints

Added `src/provenance.py` and integrated policy validation into:

- `src/experiment_runner.py`
- `src/latency.py`

The following methods now require `--policy-path` unless `--allow-heuristic-proxy` is explicitly passed:

- `ppo_static`
- `ppo_multicontext`
- `edge_only`

This prevents accidental paper-grade runs using heuristic stand-ins.

Smoke/debug runs can still use proxies with:

```bash
--allow-heuristic-proxy
```

The smoke scripts pass this flag only for dummy/debug use.

### 2. Policy and config provenance added to CSV rows

Control and latency outputs now include:

- `config_path`
- `config_sha256`
- `contexts_path`
- `contexts_sha256`
- `policy_backend`
- `trained_policy_loaded`
- `policy_path`
- `policy_sha256`
- `training_run_name`
- `policy_total_timesteps`

This makes it easier to audit which checkpoint and config produced each row.

### 3. Latency benchmark fairness improved

`src/latency.py` now recreates and resets the environment inside the per-method loop. This gives each method the same seeded initial trajectory instead of letting later methods inherit environment state advanced by earlier methods.

### 4. Scripts updated for compatibility

Updated scripts remain run-scoped and now pass heuristic-proxy override for dummy smoke/debug where needed:

- `scripts/run_smoke.sh`
- `scripts/run_baselines.sh`
- `scripts/run_latency.sh`
- `scripts/run_matrix.sh`

`run_baselines.sh`, `run_latency.sh`, and `run_matrix.sh` only add `--allow-heuristic-proxy` when `MODE=dummy`.

## Verification performed

| Check | Status |
|---|---:|
| Python compile | Passed |
| Shell syntax check | Passed |
| Guard: `ppo_static` without checkpoint fails | Passed |
| Guard: `edge_only` latency without checkpoint fails | Passed |
| Smoke script still runs | Passed |
| CSV headers include provenance columns | Passed |
| Tiny PPO checkpoint evaluation with provenance | Passed |
| Cloud package rebuilt cleanly | Passed |

Key command evidence:

```bash
cd experiments_cloud
.venv/bin/python -m compileall -q src
bash -n scripts/*.sh slurm/*.sbatch
.venv/bin/python -m src.experiment_runner --mode dummy --methods ppo_static ... # exits 1 without policy
.venv/bin/python -m src.latency --mode dummy --methods edge_only ... # exits 1 without policy
PYTHON=.venv/bin/python bash scripts/run_smoke.sh
bash scripts/package_for_cloud.sh
```

The verified package is:

```text
dist/eaai_hvac_cloud_bundle.tar.gz
```

The tarball includes `experiments_cloud/src/provenance.py` and excludes local `.venv`, caches, stale results, artifacts, and logs.

## Remaining before real paper experiments

1. Implement the real HOT/EnergyPlus adapter.
2. Train real PPO policies and pass their checkpoint paths into evaluation.
3. For method-specific PPO checkpoints, extend the runner to accept per-method policy paths instead of one global `--policy-path`.
4. Normalize deployment score before using it as a paper claim.
