"""HOT/EnergyPlus adapter scaffold for the experiments_cloud runner.

Usage on cloud/HPC after extracting the package:

    export PYTHONPATH="$PWD:$PYTHONPATH"
    export HOT_ENV_FACTORY="my_hot_adapter:create_env"

The runner calls ``create_env(context, seed, episode_steps)`` for every
(context, seed) pair. The returned object MUST be a Gymnasium-compatible
environment with:

    * ``reset(seed=...) -> (obs, info)``
    * ``step(action) -> (obs, reward, terminated, truncated, info)``
    * ``action_space`` and ``observation_space``
    * action layout ``[heating_setpoint_C, cooling_setpoint_C]`` so the
      existing rule-based and PPO controllers can drive it without changes
    * ``info`` containing per-step keys consumed by ``src/metrics.py``::

        energy_kwh, comfort_violation, temperature_deviation,
        action_instability, indoor_temp, outdoor_temp, occupancy

This file is a scaffold, not a runnable HOT adapter. Replace the stub
sections marked TODO before running paper-grade experiments. Do not report
results from the stub; it raises NotImplementedError on purpose.

References:
    * Berkes, Bengio, Rolnick, Vakalis. "A HOT Dataset: 150,000 Buildings for
      HVAC Operations Transfer Research." ACM BuildSys 2025.
      DOI: 10.1145/3736425.3770110.
      Dataset: https://huggingface.co/datasets/BuildingBench/HOT
    * Crawley et al. "EnergyPlus." Energy and Buildings, 2001.
      DOI: 10.1016/S0378-7788(00)00114-6.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "gymnasium is required for the HOT adapter. Install it on the cloud/HPC "
        "environment via 'pip install -r requirements.txt'."
    ) from exc


# ---------------------------------------------------------------------------
# Step 1. Resolve dataset / simulator path
# ---------------------------------------------------------------------------

HOT_DATASET_ROOT = Path(os.environ.get("HOT_DATASET_ROOT", "")).expanduser()
ENERGYPLUS_PATH = Path(os.environ.get("ENERGYPLUS_PATH", "")).expanduser()


def _resolve_paths() -> None:
    """Validate environment variables that point to dataset and simulator."""

    if not HOT_DATASET_ROOT or not HOT_DATASET_ROOT.exists():
        raise RuntimeError(
            "HOT_DATASET_ROOT is not set or does not exist. Download HOT from "
            "https://huggingface.co/datasets/BuildingBench/HOT and set "
            "HOT_DATASET_ROOT to the local path."
        )
    if not ENERGYPLUS_PATH or not ENERGYPLUS_PATH.exists():
        raise RuntimeError(
            "ENERGYPLUS_PATH is not set or does not exist. Install EnergyPlus "
            "(version required by HOT) and set ENERGYPLUS_PATH to its install root."
        )


# ---------------------------------------------------------------------------
# Step 2. Map experiments_cloud context -> HOT building/weather selection
# ---------------------------------------------------------------------------

def _resolve_building_and_weather(context: dict[str, Any]) -> dict[str, Any]:
    """Translate a contexts_min.yaml context into HOT-side identifiers.

    The default mapping below is illustrative. Replace it with the actual
    building IDs / IDF files / weather files supplied by the HOT dataset
    after inspecting the dataset README on Hugging Face.
    """

    building = str(context.get("building", "OfficeSmall"))
    climate_zone = str(context.get("climate_zone", "4C"))
    weather_type = str(context.get("weather_type", "TMY"))
    occupancy_schedule = str(context.get("occupancy_schedule", "standard"))

    # TODO: replace these stub paths with real HOT records.
    return {
        "building_id": f"{building}_{climate_zone}",
        "idf_path": HOT_DATASET_ROOT / "buildings" / f"{building}_{climate_zone}.idf",
        "weather_path": HOT_DATASET_ROOT / "weather" / f"{climate_zone}_{weather_type}.epw",
        "schedule": occupancy_schedule,
        "thermal_scenario": context.get("thermal_scenario", "default"),
        "shift_type": context.get("shift_type", "none"),
    }


# ---------------------------------------------------------------------------
# Step 3. Wrap simulator output into the runner's expected info schema
# ---------------------------------------------------------------------------

class HotEnvWrapper(gym.Env):
    """Adapter that exposes HOT/EnergyPlus as a Gymnasium env for the runner.

    Replace ``_simulate_step`` with the real simulator call before paper runs.
    """

    metadata = {"render_modes": []}

    def __init__(self, context: dict[str, Any], seed: int, episode_steps: int):
        _resolve_paths()
        self.context = context
        self.episode_steps = int(episode_steps)
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.spec = _resolve_building_and_weather(context)

        self.action_space = spaces.Box(
            low=np.array([18.0, 21.0], dtype=np.float32),
            high=np.array([23.0, 28.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, -40.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 100.0, 2.0, 50.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_index = 0
        self.indoor_temp = 22.0
        self.prev_action = np.array([20.0, 24.0], dtype=np.float32)
        self._sim_handle: Any = None

    # ------------------- Gymnasium API -------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        self.step_index = 0
        self.indoor_temp = float(self.rng.normal(22.0, 0.4))
        self.prev_action = np.array([20.0, 24.0], dtype=np.float32)
        self._open_simulator()
        return self._observation(energy=0.0), {}

    def step(self, action):
        heat_sp, cool_sp = self._sanitize_action(action)
        sim_out = self._simulate_step(heat_sp=heat_sp, cool_sp=cool_sp)

        self.indoor_temp = float(sim_out["indoor_temp"])
        outdoor = float(sim_out["outdoor_temp"])
        humidity = float(sim_out.get("humidity", 50.0))
        occupancy = float(sim_out.get("occupancy", 0.0))
        energy = float(sim_out["energy_kwh"])
        comfort_low, comfort_high = 20.0, 24.0
        deviation = max(comfort_low - self.indoor_temp, 0.0) + max(self.indoor_temp - comfort_high, 0.0)
        violation = 1.0 if deviation > 0.0 else 0.0
        instability = float(np.mean(np.abs(np.array([heat_sp, cool_sp]) - self.prev_action)))
        reward = -(0.12 * energy + 2.0 * deviation + 0.05 * instability)

        self.prev_action = np.array([heat_sp, cool_sp], dtype=np.float32)
        self.step_index += 1
        terminated = False
        truncated = self.step_index >= self.episode_steps
        info = {
            "energy_kwh": energy,
            "comfort_violation": violation,
            "temperature_deviation": deviation,
            "action_instability": instability,
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": outdoor,
            "occupancy": occupancy,
            "humidity": humidity,
        }
        return self._observation(energy=energy), float(reward), terminated, truncated, info

    def close(self):
        self._close_simulator()

    # ------------------- Simulator integration TODOs -------------------

    def _open_simulator(self) -> None:
        """Start a fresh HOT/EnergyPlus episode keyed by ``self.spec``."""

        # TODO: open EnergyPlus (e.g. via pyenergyplus, EnergyPlus API, or
        # Sinergym), load the IDF/EPW pointed to by self.spec, and store the
        # handle in self._sim_handle. This stub exists only to make the
        # NotImplementedError path obvious.
        raise NotImplementedError(
            "Implement _open_simulator() against the HOT dataset's expected "
            "EnergyPlus runtime before running paper experiments."
        )

    def _simulate_step(self, heat_sp: float, cool_sp: float) -> dict[str, float]:
        """Advance the simulator by one control interval and return outputs."""

        # TODO: send (heat_sp, cool_sp) to the simulator, advance one step,
        # and read indoor_temp, outdoor_temp, humidity, occupancy, and
        # HVAC energy in kWh from the simulator output stream.
        raise NotImplementedError("Implement _simulate_step() before running.")

    def _close_simulator(self) -> None:
        """Tear down the simulator handle for this episode."""

        # TODO: close the EnergyPlus process or release the simulator client.
        self._sim_handle = None

    # ------------------- helpers -------------------

    def _sanitize_action(self, action) -> tuple[float, float]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("Action must contain heating and cooling setpoints")
        heat_sp = float(np.clip(arr[0], 18.0, 23.0))
        cool_sp = float(np.clip(arr[1], 21.0, 28.0))
        cool_sp = max(cool_sp, heat_sp + 1.5)
        return heat_sp, cool_sp

    def _observation(self, energy: float):
        hour = (self.step_index % 96) / 4.0
        outdoor = 0.0
        humidity = 0.0
        occupancy = 0.0
        if self._sim_handle is not None:
            # TODO: read the most recent outdoor/humidity/occupancy values
            # from the simulator handle without advancing the simulation.
            pass
        return np.array(
            [
                self.indoor_temp,
                outdoor,
                humidity,
                occupancy,
                energy,
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Public factory consumed by experiments_cloud
# ---------------------------------------------------------------------------

def create_env(context: dict[str, Any], seed: int, episode_steps: int):
    """Factory used by experiments_cloud when ``HOT_ENV_FACTORY`` points here.

    Replace this body with logic that returns either:

    1. A direct ``HotEnvWrapper(context, seed, episode_steps)`` after
       implementing the simulator hooks above; or
    2. A Sinergym/EnergyPlus environment wrapped to expose the same
       observation/action/info schema that the runner expects.
    """

    return HotEnvWrapper(context=context, seed=seed, episode_steps=episode_steps)
