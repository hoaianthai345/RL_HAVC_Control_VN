from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .controller_tags import METHODS_USING_TRAINED_POLICY, warn_if_misconfigured


class Controller:
    name = "controller"

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        raise NotImplementedError


class StaticController(Controller):
    name = "static"

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        return np.array([20.0, 24.0], dtype=np.float32)


class AshraeController(Controller):
    name = "ashrae"

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        outdoor = float(obs[1])
        if outdoor < 15.0:
            return np.array([20.0, 23.0], dtype=np.float32)
        return np.array([22.0, 26.0], dtype=np.float32)


class ProportionalComfortController(Controller):
    name = "ppo_static"

    def __init__(self, heat_base: float = 20.0, cool_base: float = 24.5):
        self.heat_base = heat_base
        self.cool_base = cool_base

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        indoor = float(obs[0])
        outdoor = float(obs[1])
        heat = self.heat_base
        cool = self.cool_base
        if indoor < 21.0:
            heat += min(1.5, 0.4 * (21.0 - indoor))
        if indoor > 23.0:
            cool -= min(1.5, 0.4 * (indoor - 23.0))
        if outdoor > 30.0:
            cool -= 0.3
        if outdoor < 5.0:
            heat += 0.3
        return np.array([heat, max(cool, heat + 1.5)], dtype=np.float32)


class AdaptiveEdgeCloudController(Controller):
    name = "edge_cloud_adaptive"

    def __init__(self, update_interval_steps: int = 96):
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.step_index = 0
        self.heat_bias = 0.0
        self.cool_bias = 0.0
        self.recent_violations: list[float] = []
        self.base = ProportionalComfortController()

    def reset(self) -> None:
        self.step_index = 0
        self.heat_bias = 0.0
        self.cool_bias = 0.0
        self.recent_violations = []

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        if info:
            self.recent_violations.append(float(info.get("comfort_violation", 0.0)))
        if self.step_index > 0 and self.step_index % self.update_interval_steps == 0:
            rate = float(np.mean(self.recent_violations[-self.update_interval_steps :])) if self.recent_violations else 0.0
            if rate > 0.10:
                self.heat_bias = min(0.5, self.heat_bias + 0.1)
                self.cool_bias = max(-0.5, self.cool_bias - 0.1)
            else:
                self.heat_bias *= 0.8
                self.cool_bias *= 0.8
        self.step_index += 1
        action = self.base.act(obs)
        action[0] += self.heat_bias
        action[1] += self.cool_bias
        action[1] = max(action[1], action[0] + 1.5)
        return action.astype(np.float32)


class DelayedController(Controller):
    def __init__(self, inner: Controller, delay_ms: float, name: str):
        self.inner = inner
        self.delay_ms = float(delay_ms)
        self.name = name

    def reset(self) -> None:
        self.inner.reset()

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        return self.inner.act(obs, info)


class StableBaselinesPolicyController(Controller):
    def __init__(self, policy_path: str | Path, name: str = "ppo_policy"):
        from stable_baselines3 import PPO

        self.name = name
        self.model = PPO.load(str(policy_path))

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


@dataclass
class ControllerConfig:
    method: str
    cloud_delay_ms: float = 80.0
    update_interval_steps: int = 96
    policy_path: str | None = None


def build_controller(config: ControllerConfig) -> Controller:
    warn_if_misconfigured(config.method, config.policy_path)
    method = config.method
    if config.policy_path and method in METHODS_USING_TRAINED_POLICY:
        return StableBaselinesPolicyController(config.policy_path, name=method)
    if method == "static":
        return StaticController()
    if method == "ashrae":
        return AshraeController()
    if method == "ppo_static":
        return ProportionalComfortController(heat_base=20.0, cool_base=24.5)
    if method == "ppo_multicontext":
        return ProportionalComfortController(heat_base=20.2, cool_base=24.2)
    if method == "edge_only":
        return ProportionalComfortController(heat_base=20.2, cool_base=24.2)
    if method == "cloud_only":
        return DelayedController(
            ProportionalComfortController(heat_base=20.2, cool_base=24.2),
            delay_ms=config.cloud_delay_ms,
            name="cloud_only",
        )
    if method == "edge_cloud_adaptive":
        return AdaptiveEdgeCloudController(update_interval_steps=config.update_interval_steps)
    raise ValueError(f"Unsupported method: {method}")

