from __future__ import annotations

from typing import Any

import numpy as np

from .controllers import Controller


def rollout(env, controller: Controller, context: dict[str, Any], seed: int, episode_steps: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    controller.reset()
    rows = []
    last_info: dict[str, Any] = {}
    cumulative_reward = 0.0
    for step in range(episode_steps):
        action = controller.act(obs, last_info)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        rows.append(
            {
                "step": step,
                "reward": float(reward),
                "energy_kwh": float(info.get("energy_kwh", 0.0)),
                "comfort_violation": float(info.get("comfort_violation", 0.0)),
                "temperature_deviation": float(info.get("temperature_deviation", 0.0)),
                "action_instability": float(info.get("action_instability", 0.0)),
                "indoor_temp": float(info.get("indoor_temp", np.nan)),
                "outdoor_temp": float(info.get("outdoor_temp", np.nan)),
                "occupancy": float(info.get("occupancy", np.nan)),
            }
        )
        last_info = info
        if terminated or truncated:
            break

    energy = sum(row["energy_kwh"] for row in rows)
    comfort_rate = float(np.mean([row["comfort_violation"] for row in rows])) if rows else 0.0
    mean_deviation = float(np.mean([row["temperature_deviation"] for row in rows])) if rows else 0.0
    instability = float(np.mean([row["action_instability"] for row in rows])) if rows else 0.0
    return {
        "context_id": context["id"],
        "building": context.get("building", ""),
        "climate_zone": context.get("climate_zone", ""),
        "weather_type": context.get("weather_type", ""),
        "shift_type": context.get("shift_type", ""),
        "seed": seed,
        "steps": len(rows),
        "energy_kwh": float(energy),
        "comfort_violation_rate": comfort_rate,
        "mean_temperature_deviation": mean_deviation,
        "action_instability": instability,
        "cumulative_reward": float(cumulative_reward),
    }

