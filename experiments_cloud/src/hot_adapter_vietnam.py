"""
HOT adapter for Vietnam multi-city experiments.

Uses EnergyPlus 24.1 Python API directly (pyenergyplus).
No Sinergym required.

Setup (already done):
    EnergyPlus 24.1 installed at /Applications/EnergyPlus-24-1-0/
    pyenergyplus added to venv via energyplus.pth

Usage:
    export HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env"
    export PYTHONPATH="$PWD:$PYTHONPATH"

Backend selection (via HOT_BACKEND env var):
    eplus     — real EnergyPlus simulation (default when pyenergyplus available)
    epw_aware — lightweight fallback using real EPW weather but dummy physics
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = object
    spaces = None

# ── constants ─────────────────────────────────────────────────────────────────
COMFORT_LOW  = 20.0
COMFORT_HIGH = 24.0
TIMESTEP_MIN = 15
STEPS_PER_DAY = 96   # 24h × 4 steps/h

OBS_LOW  = np.array([0.0, -10.0, 0.0, 0.0,   0.0, -1.0, -1.0], dtype=np.float32)
OBS_HIGH = np.array([50.0, 50.0, 100.0, 2.0, 200.0,  1.0,  1.0], dtype=np.float32)
ACT_LOW  = np.array([16.0, 20.0], dtype=np.float32)
ACT_HIGH = np.array([24.0, 30.0], dtype=np.float32)

EPLUS_DIR = Path("/Applications/EnergyPlus-24-1-0")

# ── helpers ───────────────────────────────────────────────────────────────────
def _reward(energy: float, deviation: float, instability: float) -> float:
    return -(0.12 * energy + 2.0 * deviation + 0.05 * instability)

def _time_features(step: int) -> tuple[float, float]:
    hour = (step % STEPS_PER_DAY) * TIMESTEP_MIN / 60.0
    return math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)

def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    for base in [Path.cwd(), Path(__file__).parent.parent]:
        c = base / path
        if c.exists():
            return c
    return Path.cwd() / path


# ══════════════════════════════════════════════════════════════════════════════
# Backend A — EnergyPlusHOTEnv (real simulation)
# ══════════════════════════════════════════════════════════════════════════════

class EnergyPlusHOTEnv(gym.Env if hasattr(gym, "Env") else object):
    """
    Gymnasium environment wrapping EnergyPlus 24.1 via Python API.

    Control target: Core_ZN heating + cooling setpoints (Schedule:Constant).
    Observation:    [T_zone, T_out, RH, occupancy, energy_kWh, sin_h, cos_h]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        epjson_path: str,
        epw_path: str,
        context: dict[str, Any],
        seed: int = 0,
        episode_steps: int = 288,
    ):
        self.epjson_path  = _strip_ems(str(epjson_path))
        self.epw_path     = str(epw_path)
        self.context      = context
        self.episode_steps = int(episode_steps)

        if spaces is not None:
            self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
            self.action_space      = spaces.Box(ACT_LOW, ACT_HIGH, dtype=np.float32)

        # Inter-thread communication
        self._obs_q  = queue.Queue(maxsize=1)
        self._act_q  = queue.Queue(maxsize=1)

        # EnergyPlus state (reset each episode)
        self._ep_state   = None
        self._ep_thread  = None
        self._api        = None

        # Variable/actuator handles (resolved once warmup completes)
        self._h: dict[str, int] = {}

        self._step_idx   = 0
        self._prev_act   = np.array([20.0, 26.0], dtype=np.float32)
        self._episode_done = False
        self._outdir: str | None = None

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._stop_eplus()
        # Each EnergyPlus run writes 1–5 MB of .eio/.mdd/.csv output to its
        # output dir. Without cleanup this leaks ~9 GB per Hospital training.
        if self._outdir and Path(self._outdir).exists():
            shutil.rmtree(self._outdir, ignore_errors=True)
        self._outdir = None

        from pyenergyplus.api import EnergyPlusAPI
        self._api = EnergyPlusAPI()
        self._ep_state = self._api.state_manager.new_state()
        self._h.clear()
        self._step_idx  = 0
        self._prev_act  = np.array([20.0, 26.0], dtype=np.float32)
        self._episode_done = False

        # Suppress EnergyPlus console output
        self._api.runtime.set_console_output_status(self._ep_state, False)

        # Register per-timestep callback. Once EMS:Actuator declarations are
        # stripped from the epJSON (see _strip_ems above), all per-zone-timestep
        # callbacks deliver setpoint writes correctly; we keep the original
        # hook for backwards-compatibility with previously-trained policies.
        self._api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            self._ep_state, self._eplus_callback
        )

        # Output directory (temp, unique per episode)
        self._outdir = tempfile.mkdtemp(prefix="eplus_vn_")

        def _run():
            self._api.runtime.run_energyplus(self._ep_state, [
                "-w", self.epw_path,
                "-d", self._outdir,
                "--output-prefix", "ep",
                "-r",              # run period from IDF, not design day
                self.epjson_path,
            ])
            # Signal episode end to step()
            try:
                self._obs_q.put(None, timeout=5)
            except queue.Full:
                pass

        self._ep_thread = threading.Thread(target=_run, daemon=True)
        self._ep_thread.start()

        obs = self._obs_q.get(timeout=120)   # wait for first timestep
        if obs is None:
            raise RuntimeError("EnergyPlus exited before producing first observation.")
        return obs, {}

    def step(self, action: np.ndarray):
        if self._episode_done:
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

        # Sanitize and send action to EnergyPlus callback
        heat_sp = float(np.clip(action[0], ACT_LOW[0], ACT_HIGH[0]))
        cool_sp = float(np.clip(action[1], ACT_LOW[1], ACT_HIGH[1]))
        cool_sp = max(cool_sp, heat_sp + 1.5)
        self._act_q.put(np.array([heat_sp, cool_sp], dtype=np.float32))

        # Wait for next observation
        obs = self._obs_q.get(timeout=120)

        self._step_idx += 1
        truncated  = self._step_idx >= self.episode_steps
        terminated = False

        if obs is None:       # EnergyPlus finished year simulation
            obs = np.zeros(7, dtype=np.float32)
            terminated = True
            self._episode_done = True
            info = {"energy_kwh": 0.0, "comfort_violation": 0.0,
                    "temperature_deviation": 0.0, "action_instability": 0.0,
                    "indoor_temp": 22.0, "outdoor_temp": 25.0, "backend": "eplus"}
            return obs, 0.0, terminated, truncated, info

        t_in    = float(obs[0])
        t_out   = float(obs[1])
        energy  = float(obs[4])

        deviation   = max(COMFORT_LOW - t_in, 0.0) + max(t_in - COMFORT_HIGH, 0.0)
        violation   = 1.0 if deviation > 0.0 else 0.0
        instability = float(np.mean(np.abs(np.array([heat_sp, cool_sp]) - self._prev_act)))
        self._prev_act = np.array([heat_sp, cool_sp], dtype=np.float32)

        reward = _reward(energy, deviation, instability)
        info = {
            "energy_kwh":            energy,
            "comfort_violation":     violation,
            "temperature_deviation": deviation,
            "action_instability":    instability,
            "indoor_temp":           t_in,
            "outdoor_temp":          t_out,
            "occupancy":             float(obs[3]),
            "backend":               "eplus",
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_eplus()
        if self._outdir and Path(self._outdir).exists():
            shutil.rmtree(self._outdir, ignore_errors=True)
        self._outdir = None

    # ── EnergyPlus callback (runs in EnergyPlus thread) ───────────────────────

    def _eplus_callback(self, state: Any) -> None:
        api = self._api

        if not api.exchange.api_data_fully_ready(state):
            return

        # Skip EnergyPlus warmup days — meters are unreliable during warmup
        if api.exchange.warmup_flag(state):
            return

        # ── Resolve handles once ───────────────────────────────────────────────
        if not self._h:
            get_var = api.exchange.get_variable_handle
            get_act = api.exchange.get_actuator_handle
            get_met = api.exchange.get_meter_handle

            # Discover ALL zones with HEATING SETPOINT actuators. We must
            # actuate every zone, not just the primary one — otherwise the
            # untouched perimeter zones (typically 4× core area in DOE proto
            # buildings) dominate energy/comfort and the policy has no effect.
            zones = self._discover_all_zones(api, state)
            primary = zones[0] if zones else "CORE_ZN"
            self._zones = zones

            self._h["t_zone"]  = get_var(state, "Zone Mean Air Temperature", primary)
            self._h["t_out"]   = get_var(state, "Site Outdoor Air Drybulb Temperature", "Environment")
            self._h["rh"]      = get_var(state, "Site Outdoor Air Relative Humidity", "Environment")
            self._h["occ"]     = get_var(state, "Zone People Occupant Count", primary)
            self._h["hvac_j"]  = get_met(state, "Electricity:HVAC")

            # Per-zone setpoint actuator handles
            self._h["heat_sps"] = []
            self._h["cool_sps"] = []
            missing_zones = []
            for z in zones:
                hh = get_act(state, "Schedule:Constant", "Schedule Value", f"{z} HEATING SETPOINT")
                hc = get_act(state, "Schedule:Constant", "Schedule Value", f"{z} COOLING SETPOINT")
                if hh < 0 or hc < 0:
                    missing_zones.append(z)
                    continue
                self._h["heat_sps"].append(hh)
                self._h["cool_sps"].append(hc)

            scalar = {k: v for k, v in self._h.items()
                      if k not in ("heat_sps", "cool_sps") and isinstance(v, int) and v < 0}
            if scalar or missing_zones or not self._h["heat_sps"]:
                warnings.warn(
                    f"EnergyPlus handle resolution issues: scalar_invalid={scalar}, "
                    f"missing_zones={missing_zones}, zones_attempted={zones}"
                )

        # ── Read sensors ───────────────────────────────────────────────────────
        gv  = api.exchange.get_variable_value
        gm  = api.exchange.get_meter_value
        t_zone = gv(state, self._h["t_zone"])
        t_out  = gv(state, self._h["t_out"])
        rh     = gv(state, self._h["rh"])
        occ    = min(gv(state, self._h["occ"]), 2.0)

        # Meter returns J per timestep → convert to kWh
        hvac_j     = max(gm(state, self._h["hvac_j"]), 0.0)
        energy_kwh = hvac_j / 3_600_000.0

        sin_h, cos_h = _time_features(self._step_idx)
        obs = np.array([t_zone, t_out, rh, occ, energy_kwh, sin_h, cos_h],
                       dtype=np.float32)

        # ── Send observation to step() ─────────────────────────────────────────
        try:
            self._obs_q.put(obs, timeout=30)
        except queue.Full:
            return   # step() timed out, skip this timestep

        # ── Wait for action from step() ────────────────────────────────────────
        try:
            action = self._act_q.get(timeout=30)
        except queue.Empty:
            action = self._prev_act   # keep previous setpoints if no action received

        # ── Apply setpoints to all zones ───────────────────────────────────────
        sa = api.exchange.set_actuator_value
        h_act = float(action[0])
        c_act = float(action[1])
        for hh in self._h["heat_sps"]:
            sa(state, hh, h_act)
        for hc in self._h["cool_sps"]:
            sa(state, hc, c_act)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _discover_all_zones(self, api: Any, state: Any) -> list[str]:
        """
        Discover every zone that has a Schedule:Constant heating-setpoint
        actuator. Returns a list of zone names (e.g. ['CORE_ZN',
        'PERIMETER_ZN_1', ...]) so the policy actuates ALL zones, not just the
        first one. Order is stable (first-seen in the API's data CSV).
        """
        zones: list[str] = []
        try:
            csv = api.exchange.list_available_api_data_csv(state)
            lines = (csv.decode() if isinstance(csv, bytes) else str(csv)).split('\n')
            for line in lines:
                u = line.upper()
                if 'SCHEDULE:CONSTANT' not in u or 'HEATING SETPOINT' not in u:
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                name = parts[3].strip()  # e.g. "CORE_ZN HEATING SETPOINT"
                # Case-insensitive strip of trailing " HEATING SETPOINT"
                low = name.lower()
                if low.endswith(" heating setpoint"):
                    zone = name[: -len(" heating setpoint")].strip()
                else:
                    continue
                if zone and zone not in zones:
                    zones.append(zone)
        except Exception:
            pass
        return zones or ["CORE_ZN"]

    def _stop_eplus(self):
        # Signal EnergyPlus to stop, then wait for the run thread to exit BEFORE
        # deleting the state — otherwise the C++ runtime still dereferences state
        # we just freed and segfaults (esp. on the 55-zone Hospital model).
        if self._ep_state is not None and self._api is not None:
            try:
                self._api.runtime.stop_simulation(self._ep_state)
            except Exception:
                pass
            # Unblock the callback if it is waiting for an action so the run
            # thread can wind down promptly.
            try:
                self._act_q.put_nowait(self._prev_act)
            except queue.Full:
                pass
            # Drain any pending observation so the callback's obs_q.put doesn't block.
            try:
                self._obs_q.get_nowait()
            except queue.Empty:
                pass
            if self._ep_thread is not None and self._ep_thread.is_alive():
                self._ep_thread.join(timeout=10)
            try:
                self._api.state_manager.delete_state(self._ep_state)
            except Exception:
                pass
        self._ep_state  = None
        self._ep_thread = None
        self._api       = None
        # Final drain of any leftover items in queues.
        for q in (self._obs_q, self._act_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break


# ══════════════════════════════════════════════════════════════════════════════
# Backend B — EPWAwareEnv (fast fallback, no EnergyPlus needed)
# ══════════════════════════════════════════════════════════════════════════════

_STRIPPED_EPJSON_CACHE: dict[str, str] = {}


def _strip_ems(epjson_path: str) -> str:
    """
    Return a path to a copy of the model where ONLY the EMS:Actuator entries
    that target setpoint Schedule:Constants are removed. We keep everything
    else (EMS:Program, ProgramCallingManager, GlobalVariable, Sensor, etc.) so
    transformer/load-center logic that depends on EMS still works — stripping
    too aggressively crashes EnergyPlus (e.g. ElectricTransformer null deref
    in OfficeMedium). The targeted EMS:Actuators are what was claiming
    exclusive ownership of setpoint schedules and silently dropping our API
    writes. Cached per source path.
    """
    src = str(epjson_path)
    if src in _STRIPPED_EPJSON_CACHE:
        return _STRIPPED_EPJSON_CACHE[src]
    with open(src) as f:
        data = json.load(f)
    actuators = data.get("EnergyManagementSystem:Actuator", {})
    if not actuators:
        _STRIPPED_EPJSON_CACHE[src] = src
        return src
    keep = {}
    removed = 0
    for name, body in actuators.items():
        target = str(body.get("actuated_component_unique_name", "")).upper()
        ctype = str(body.get("actuated_component_type", "")).upper()
        if "SETPOINT" in target and ctype == "SCHEDULE:CONSTANT":
            removed += 1
            continue
        keep[name] = body
    if removed == 0:
        _STRIPPED_EPJSON_CACHE[src] = src
        return src
    if keep:
        data["EnergyManagementSystem:Actuator"] = keep
    else:
        data.pop("EnergyManagementSystem:Actuator", None)
    out_dir = Path(tempfile.gettempdir()) / "vn_eplus_models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(src).stem + "__sp_unblocked.epJSON")
    with open(out_path, "w") as f:
        json.dump(data, f)
    _STRIPPED_EPJSON_CACHE[src] = str(out_path)
    return str(out_path)


def _parse_epw(epw_path: str) -> np.ndarray:
    """Parse EPW → float32 array (8760, 3): [T_out, RH, GHI]."""
    rows = []
    with open(epw_path, encoding="utf-8", errors="replace") as f:
        for _ in range(8):
            f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 14:
                continue
            try:
                rows.append([float(parts[6]), float(parts[8]), float(parts[13])])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float32)


class EPWAwareEnv(gym.Env if hasattr(gym, "Env") else object):
    """Real EPW weather + DummyHVACEnv thermal model. For pipeline testing only."""

    metadata = {"render_modes": []}

    def __init__(self, epw_path: str, context: dict, seed: int = 0, episode_steps: int = 288):
        self._epw   = _parse_epw(epw_path)
        self.context = context
        self.episode_steps = int(episode_steps)
        self._rng   = np.random.default_rng(seed)
        cfg = context.get("dummy", {})
        self._env_f = float(cfg.get("envelope_factor", 1.0))
        self._occ_s = float(cfg.get("occupancy_scale", 1.0))

        if spaces is not None:
            self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
            self.action_space      = spaces.Box(ACT_LOW, ACT_HIGH, dtype=np.float32)

        self._step  = 0
        self._off   = 0
        self._t_in  = 25.0
        self._prev  = np.array([20.0, 26.0], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._off  = int(self._rng.integers(0, max(1, len(self._epw) - self.episode_steps // 4)))
        self._step = 0
        self._t_in = float(self._rng.normal(25.0, 1.0))
        self._prev = np.array([20.0, 26.0], dtype=np.float32)
        return self._obs(0.0), {}

    def step(self, action: np.ndarray):
        heat_sp = float(np.clip(action[0], ACT_LOW[0], ACT_HIGH[0]))
        cool_sp = float(np.clip(action[1], ACT_LOW[1], ACT_HIGH[1]))
        cool_sp = max(cool_sp, heat_sp + 1.5)

        t_out, rh, ghi = self._weather()
        occ  = self._occupancy()
        heat_need = max(0.0, heat_sp - self._t_in)
        cool_need = max(0.0, self._t_in - cool_sp)
        self._t_in += (0.055 * self._env_f * (t_out - self._t_in)
                       + 0.38 * heat_need - 0.34 * cool_need
                       + 0.045 * occ + 0.002 * ghi / 1000
                       + float(self._rng.normal(0, 0.03)))

        energy  = 0.18 + 0.75 * heat_need + 0.85 * cool_need + 0.04 * occ
        dev     = max(COMFORT_LOW - self._t_in, 0.0) + max(self._t_in - COMFORT_HIGH, 0.0)
        inst    = float(np.mean(np.abs(np.array([heat_sp, cool_sp]) - self._prev)))
        self._prev = np.array([heat_sp, cool_sp], dtype=np.float32)

        self._step += 1
        info = {"energy_kwh": energy, "comfort_violation": 1.0 if dev > 0 else 0.0,
                "temperature_deviation": dev, "action_instability": inst,
                "indoor_temp": self._t_in, "outdoor_temp": t_out, "backend": "epw_aware"}
        return self._obs(energy), _reward(energy, dev, inst), False, self._step >= self.episode_steps, info

    def close(self): pass

    def _weather(self):
        idx = (self._off + self._step // 4) % len(self._epw)
        r = self._epw[idx]
        return float(r[0]), float(r[1]), float(r[2])

    def _occupancy(self):
        hour = (self._step % STEPS_PER_DAY) * TIMESTEP_MIN / 60.0
        bldg = self.context.get("building", "")
        if "Hospital" in bldg:
            occ = 1.0
        elif "School" in bldg:
            occ = 1.0 if 7.0 <= hour <= 15.0 else 0.05
        else:
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.15
        return float(occ * self._occ_s)

    def _obs(self, energy: float) -> np.ndarray:
        t_out, rh, _ = self._weather()
        sin_h, cos_h = _time_features(self._step)
        return np.array([self._t_in, t_out, rh, self._occupancy(),
                         energy, sin_h, cos_h], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Factory — create_env
# ══════════════════════════════════════════════════════════════════════════════

def create_env(context: dict[str, Any], seed: int, episode_steps: int) -> Any:
    """
    HOT_ENV_FACTORY entry point.
    Selects backend: eplus (default) or epw_aware (fallback / HOT_BACKEND=epw_aware).
    """
    hot      = context.get("hot", {})
    epw_path  = _resolve(str(hot.get("epw_file", "")))
    model_path = _resolve(str(hot.get("model_file", "")))
    forced   = os.environ.get("HOT_BACKEND", "").lower()

    # ── EnergyPlus backend ────────────────────────────────────────────────────
    if forced != "epw_aware":
        try:
            from pyenergyplus.api import EnergyPlusAPI   # noqa: F401
            if not model_path.exists():
                raise FileNotFoundError(f"epJSON not found: {model_path}")
            if not epw_path.exists():
                raise FileNotFoundError(f"EPW not found: {epw_path}")
            return EnergyPlusHOTEnv(str(model_path), str(epw_path),
                                    context, seed, episode_steps)
        except ImportError:
            warnings.warn("pyenergyplus not found — falling back to EPWAwareEnv", stacklevel=2)
        except FileNotFoundError as e:
            warnings.warn(f"{e} — falling back to EPWAwareEnv", stacklevel=2)
        except Exception as e:
            warnings.warn(f"EnergyPlus init error: {e} — falling back to EPWAwareEnv", stacklevel=2)

    # ── EPWAwareEnv fallback ──────────────────────────────────────────────────
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    return EPWAwareEnv(str(epw_path), context, seed, episode_steps)
