from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - allows syntax checks without gymnasium.
    gym = object
    spaces = None


@dataclass
class StepMetrics:
    energy_kwh: float
    comfort_violation: float
    temperature_deviation: float
    action_instability: float


class DummyHVACEnv(gym.Env if hasattr(gym, "Env") else object):
    """Small deterministic HVAC-like environment for cloud smoke tests.

    This environment is intentionally simple. It verifies the experiment
    pipeline but must not be used as paper evidence.

    The per-step ``energy_kwh`` field in ``info`` is a synthetic proxy for HVAC
    intensity, not a calibrated whole-building meter reading. Report it as a
    surrogate only until validated against HOT/EnergyPlus totals.
    """

    metadata = {"render_modes": []}

    def __init__(self, context: dict[str, Any], seed: int = 0, episode_steps: int = 288):
        self.context = context
        self.episode_steps = int(episode_steps)
        self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.indoor_temp = 22.0
        self.prev_action = np.array([20.0, 24.0], dtype=np.float32)
        dummy = context.get("dummy", {})
        self.climate_profile = dummy.get("climate_profile", "mild")
        self.envelope_factor = float(dummy.get("envelope_factor", 1.0))
        self.occupancy_scale = float(dummy.get("occupancy_scale", 1.0))
        if spaces is not None:
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.indoor_temp = float(self.rng.normal(22.0, 0.4))
        self.prev_action = np.array([20.0, 24.0], dtype=np.float32)
        return self._observation(energy=0.0), {}

    def step(self, action):
        heat_sp, cool_sp = self._sanitize_action(action)
        outdoor = self._outdoor_temperature()
        humidity = self._humidity()
        occupancy = self._occupancy()

        heat_need = max(0.0, heat_sp - self.indoor_temp)
        cool_need = max(0.0, self.indoor_temp - cool_sp)
        hvac_effect = 0.38 * heat_need - 0.34 * cool_need
        envelope_drift = 0.055 * self.envelope_factor * (outdoor - self.indoor_temp)
        internal_gain = 0.045 * occupancy
        noise = float(self.rng.normal(0.0, 0.03))
        self.indoor_temp += envelope_drift + hvac_effect + internal_gain + noise

        energy = 0.18 + 0.75 * heat_need + 0.85 * cool_need + 0.04 * occupancy
        comfort_low, comfort_high = 20.0, 24.0
        deviation = max(comfort_low - self.indoor_temp, 0.0) + max(self.indoor_temp - comfort_high, 0.0)
        violation = 1.0 if deviation > 0.0 else 0.0
        instability = float(np.mean(np.abs(np.array([heat_sp, cool_sp]) - self.prev_action)))
        reward = -(0.12 * energy + 2.0 * deviation + 0.05 * instability)

        self.prev_action = np.array([heat_sp, cool_sp], dtype=np.float32)
        self.step_index += 1
        terminated = False
        truncated = self.step_index >= self.episode_steps
        obs = self._observation(energy=energy)
        info = {
            "energy_kwh": float(energy),
            "comfort_violation": float(violation),
            "temperature_deviation": float(deviation),
            "action_instability": float(instability),
            "indoor_temp": float(self.indoor_temp),
            "outdoor_temp": float(outdoor),
            "occupancy": float(occupancy),
        }
        return obs, float(reward), terminated, truncated, info

    def _sanitize_action(self, action) -> tuple[float, float]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("Action must contain heating and cooling setpoints")
        heat_sp = float(np.clip(arr[0], 18.0, 23.0))
        cool_sp = float(np.clip(arr[1], 21.0, 28.0))
        cool_sp = max(cool_sp, heat_sp + 1.5)
        return heat_sp, cool_sp

    def _outdoor_temperature(self) -> float:
        hour = (self.step_index % 96) / 4.0
        day = self.step_index / 96.0
        daily = math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
        if self.climate_profile == "hot_humid":
            base, amp = 30.0, 6.0
        elif self.climate_profile == "cold":
            base, amp = 5.0, 7.0
        elif self.climate_profile == "mild_variable":
            base, amp = 16.0 + 2.0 * math.sin(day / 2.0), 8.0
        else:
            base, amp = 16.0, 5.0
        return float(base + amp * daily + self.rng.normal(0.0, 0.4))

    def _humidity(self) -> float:
        if self.climate_profile == "hot_humid":
            return float(np.clip(self.rng.normal(75.0, 5.0), 45.0, 95.0))
        return float(np.clip(self.rng.normal(55.0, 8.0), 20.0, 90.0))

    def _occupancy(self) -> float:
        hour = (self.step_index % 96) / 4.0
        occupied = 1.0 if 8.0 <= hour <= 18.0 else 0.15
        if "school" in str(self.context.get("occupancy_schedule", "")).lower():
            occupied = 1.0 if 7.0 <= hour <= 15.0 else 0.05
        return float(occupied * self.occupancy_scale)

    def _observation(self, energy: float):
        outdoor = self._outdoor_temperature()
        humidity = self._humidity()
        occupancy = self._occupancy()
        hour = (self.step_index % 96) / 4.0
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


def _load_hot_factory():
    factory_path = os.environ.get("HOT_ENV_FACTORY", "")
    if not factory_path or ":" not in factory_path:
        raise RuntimeError(
            "HOT mode requires HOT_ENV_FACTORY='module:function'. "
            "The function must return a Gymnasium-compatible HOT environment."
        )
    module_name, function_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def make_env(mode: str, context: dict[str, Any], seed: int, episode_steps: int):
    if mode == "dummy":
        return DummyHVACEnv(context=context, seed=seed, episode_steps=episode_steps)
    if mode == "hot":
        factory = _load_hot_factory()
        return factory(context=context, seed=seed, episode_steps=episode_steps)
    raise ValueError(f"Unsupported mode: {mode}")

