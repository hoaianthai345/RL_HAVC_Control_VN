# Codebase Audit — `experiments_cloud` HVAC RL Experiment Package

Date: 2026-05-07  
Scope: local codebase under `experiments_cloud/` plus run scripts/configs.  
Focus: reproducibility, code/claim consistency, experiment validity, and execution risks.

## Executive summary

The package is usable as a **smoke-test scaffold** for HVAC/RL experiments: Python modules compile, dummy control/latency smoke tests run, and a tiny PPO train/evaluate cycle works when using the project virtual environment. However, I would **not use the current outputs as paper evidence** without fixes. The highest-risk problems are stale-result contamination in summaries, heuristic stand-ins labelled as PPO/edge-cloud methods unless a policy path is supplied, and HOT/EnergyPlus integration being only a template.

## Checks run

From `/Users/anhoaithai/Documents/AHT/1. PROJECTS/RL HVAC HPC`:

| Check | Result | Evidence |
|---|---:|---|
| File inventory | Run | `find . -maxdepth 4 -type f` |
| Python syntax compile | Pass | `cd experiments_cloud && python3 -m compileall -q src && echo compile_ok` |
| Dummy smoke pipeline | Pass | `cd experiments_cloud && bash scripts/run_smoke.sh` wrote `results/raw/smoke_control_control.csv`, `results/raw/smoke_latency.csv`, and summary CSVs |
| System Python PPO train | Fail | `ModuleNotFoundError: No module named 'stable_baselines3'` |
| Project venv dependency import | Pass | `.venv/bin/python` imported `numpy`, `pandas`, `yaml`, `gymnasium`, `torch`, `stable_baselines3` |
| Tiny PPO train + load/evaluate | Pass | `.venv/bin/python -m src.train_ppo --mode dummy --total-timesteps 16 ...` then `.venv/bin/python -m src.experiment_runner --policy-path ...` wrote `/tmp/audit_results/audit_policy_eval_control.csv` |

## Findings

### Critical / high-priority

1. **Summary CSVs aggregate every CSV in `results/raw`, including stale or unrelated runs.**
   - Evidence: `src/summarize_results.py:75-88` loads `sorted(input_dir.glob("*.csv"))` with no run-name, timestamp, mode, or manifest filter. In the current tree, `results/raw` contains `bootstrap_smoke_control.csv`, `smoke_control_control.csv`, and `smoke_latency.csv`, so a smoke summary includes bootstrap data.
   - Impact: reported means/CIs can silently mix old smoke, baseline, matrix, and latency runs. This is a reproducibility and paper-validity risk.
   - Fix: write each run to an isolated output directory or add `--run-name/--include-glob` filtering and store a manifest of input files in `results/summary/manifest.json`.

2. **Several named methods are heuristic stand-ins, not trained RL policies, unless `--policy-path` is supplied.**
   - Evidence: `src/controllers.py:143-156` maps `ppo_static`, `ppo_multicontext`, `edge_only`, and `edge_cloud_adaptive` to rule/proportional controllers by default. `src/controller_tags.py` does warn for missing policy path, but the CSV still contains method names that look like trained methods.
   - Impact: tables can be misread as PPO/edge-cloud RL results when they are heuristic proxies. The README correctly warns this, but the result schema relies on consumers noticing `policy_backend` and `trained_policy_loaded`.
   - Fix: rename fallback methods in outputs, fail closed for paper runs, or require `--allow-heuristic-proxy` for methods with RL names when no policy checkpoint is provided.

3. **HOT/EnergyPlus production path is not implemented in this repo.**
   - Evidence: `src/hot_adapter_template.py` raises `NotImplementedError`; `make_env(mode="hot")` requires `HOT_ENV_FACTORY` from user code.
   - Impact: cloud package cannot by itself reproduce real-building results. Dummy results are explicitly synthetic.
   - Fix: add a concrete HOT adapter, document dataset version/building files/weather mapping, and add a HOT smoke test that exercises one short episode.

4. **Deployment score mixes differently scaled quantities without normalization.**
   - Evidence: `src/deployment.py:31-47` computes `energy_kwh + 8*comfort_rate + 2*instability + latency_penalty`. Energy is episode-length-dependent while comfort is unitless; changing `episode_steps` changes score dominance.
   - Impact: policy ranking and acceptance gates may be arbitrary across horizons/contexts.
   - Fix: normalize energy per floor area or per step, define units, and calibrate weights with sensitivity analysis.

### Medium-priority

5. **Latency benchmark reuses one environment across methods within a context.**
   - Evidence: in `src/latency.py:52-55`, the env is created/reset before the method loop, not inside it. Later methods start after earlier methods have advanced the environment.
   - Impact: for controller-only latency this may be small, but it weakens fairness and complicates reproducibility for stateful controllers/adapters.
   - Fix: reset or recreate env for each `(context, method)` with the same seed and optionally precompute a fixed observation trace.

6. **Raw rollout traces are not saved, only aggregate rows.**
   - Evidence: `src/metrics.py:13-40` creates per-step `rows`, but `experiment_runner.py` writes only aggregate result rows.
   - Impact: debugging comfort excursions, instability, and controller behavior is hard; reviewers cannot inspect time-series behavior.
   - Fix: optionally write per-step traces under `results/traces/{run_name}/...csv`.

7. **The dummy environment samples exogenous weather/humidity multiple times per step.**
   - Evidence: `DummyHVACEnv.step()` samples outdoor/humidity at `src/envs.py:73-75`, then `_observation()` samples a new outdoor/humidity at `src/envs.py:143-146`.
   - Impact: the observation given to the controller may not correspond to the same exogenous conditions used to update physics and metrics.
   - Fix: generate one exogenous state per timestep and reuse it for transition, info, and next observation.

8. **Training script does not summarize/evaluate saved PPO checkpoints automatically.**
   - Evidence: `scripts/run_train_ppo.sh` only invokes `src.train_ppo`; it does not run evaluation with the new checkpoint.
   - Impact: easy to create a policy artifact but still publish heuristic fallback results.
   - Fix: after training, run `experiment_runner --policy-path artifacts/policies/<checkpoint>.zip` and record checkpoint hash in raw results.

### Lower-priority / hygiene

9. **No test suite is present.** Smoke scripts exist, but there are no unit/regression tests for metrics, score calculation, summaries, or controller mapping.
10. **Dependency versions are broad.** `requirements.txt` uses lower bounds only. This improves install flexibility but weakens reproducibility.
11. **Generated environment/cache files are present in the working tree.** `.venv/`, `__pycache__/`, and result files exist locally. Packaging excludes most of these, but repository hygiene would benefit from `.gitignore` enforcement.

## Recommended next fixes

1. Make summaries run-scoped: isolated `results/raw/<run_name>/` or manifest-based filtering.
2. Add a paper-mode guard: fail if RL-named methods run without real policy checkpoints unless explicitly overridden.
3. Add concrete HOT adapter or mark all HOT-dependent outputs as blocked until adapter validation passes.
4. Add minimal tests:
   - `deployment_score` units/latency merge test;
   - `summarize_results` ignores stale files or requires explicit inputs;
   - `build_controller` method-to-backend mapping;
   - dummy env deterministic reset and exogenous-state consistency.
5. Add checkpoint provenance: policy path, file hash, training config, total timesteps, seed, and environment mode in every result row.

## Status

- **Executable as a smoke-test scaffold:** yes, with `.venv/bin/python`.
- **Ready for manuscript quantitative claims:** no; current results should be treated as debug/proxy outputs unless run-scoping, policy provenance, and HOT validation are fixed.
