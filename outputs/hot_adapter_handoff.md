# HOT/EnergyPlus Adapter Handoff

Date: 2026-05-07  
Scope: file `experiments_cloud/my_hot_adapter.py` and how to finish it on cloud/HPC.

## What was added

A scaffold adapter file at:

```text
experiments_cloud/my_hot_adapter.py
```

It:

1. Validates two environment variables before doing anything else:
   - `HOT_DATASET_ROOT` — local path to the downloaded HOT dataset.
   - `ENERGYPLUS_PATH` — install root of the EnergyPlus version required by HOT.
2. Exposes `create_env(context, seed, episode_steps)` so the runner can call it via:

   ```bash
   export HOT_ENV_FACTORY="my_hot_adapter:create_env"
   ```
3. Provides `HotEnvWrapper`, a Gymnasium env with the action/observation/info schema the runner expects:
   - action `[heating_setpoint_C, cooling_setpoint_C]`
   - info keys: `energy_kwh`, `comfort_violation`, `temperature_deviation`, `action_instability`, `indoor_temp`, `outdoor_temp`, `occupancy`, `humidity`.
4. Marks the simulator-specific hooks with explicit `TODO` and raises `NotImplementedError` so unfinished code cannot silently produce results.

The file compiles and imports cleanly, but it intentionally fails at runtime until the simulator hooks are implemented.

## What you must implement on cloud

### 1. Download HOT dataset

The HOT dataset is referenced as:

- ACM BuildSys 2025: Berkes, Bengio, Rolnick, Vakalis. "A HOT Dataset: 150,000 Buildings for HVAC Operations Transfer Research." DOI: `10.1145/3736425.3770110`.
- Hugging Face: `https://huggingface.co/datasets/BuildingBench/HOT`

On cloud:

```bash
huggingface-cli login   # if required
huggingface-cli download BuildingBench/HOT --local-dir /data/HOT
export HOT_DATASET_ROOT=/data/HOT
```

I have not verified the exact dataset layout. After download, inspect the dataset README and update `_resolve_building_and_weather` to point to real `.idf`/`.epw` files or to whatever interface HOT publishes.

### 2. Install EnergyPlus

HOT relies on EnergyPlus for thermal simulation. Install the version specified by the HOT dataset documentation, then:

```bash
export ENERGYPLUS_PATH=/usr/local/EnergyPlus-X-Y-Z
```

If the cloud node is restricted, a Sinergym container is a known alternative for EnergyPlus + Gymnasium integration. Document whichever path you take.

### 3. Implement three hooks in `HotEnvWrapper`

The TODO blocks are:

- `_open_simulator(self)` — start a fresh EnergyPlus episode for `self.spec` (IDF + EPW + schedule). Save the handle on `self._sim_handle`.
- `_simulate_step(self, heat_sp, cool_sp)` — push the setpoints into EnergyPlus, advance one control interval, and return a dict with at least:
  - `indoor_temp`
  - `outdoor_temp`
  - `humidity`
  - `occupancy`
  - `energy_kwh`
- `_close_simulator(self)` — tear down the handle when the episode ends.

The reward calculation already lives in `step()` and matches what the runner expects.

### 4. Smoke-test the adapter on cloud

After implementing the hooks:

```bash
cd experiments_cloud
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
export HOT_DATASET_ROOT=/data/HOT
export ENERGYPLUS_PATH=/usr/local/EnergyPlus-X-Y-Z
python -m src.experiment_runner \
  --mode hot \
  --methods static \
  --seeds 1 \
  --episode-steps 8 \
  --output-dir results/raw/hot_adapter_smoke \
  --run-name hot_adapter_smoke
```

If this writes a CSV with non-zero `energy_kwh` and sensible temperatures, the adapter is wired up. Only then run training and full matrix.

## Verification done now

- `python -m compileall -q my_hot_adapter.py` passed.
- `import my_hot_adapter; hasattr(my_hot_adapter, 'create_env')` returned True.
- `bash scripts/package_for_cloud.sh` rebuilt the cloud bundle with the new adapter file included; `dist/eaai_hvac_cloud_bundle.tar.gz` now contains `my_hot_adapter.py` and `src/provenance.py`.

## Reminder before paper-grade runs

The packaging script also excludes a personal override at `my_hot_adapter_local.py`, so you can keep cluster-specific paths or credentials out of the cloud bundle if needed.

After finishing the adapter, follow the post-adapter sequence:

1. Train PPO with real HOT episodes.
2. Run the evaluation matrix with `--policy-path` and provenance flags.
3. Run latency benchmarks with the same policy paths.
4. Summarize the run-scoped output directories and fill the paper.
