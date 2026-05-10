"""Template for connecting the cloud runner to the real HOT environment.

Copy this file to `my_hot_adapter.py`, replace the placeholder implementation,
then run:

    export PYTHONPATH="$PWD:$PYTHONPATH"
    export HOT_ENV_FACTORY="my_hot_adapter:create_env"

The returned object should be Gymnasium-compatible.
"""

from __future__ import annotations

from typing import Any


def create_env(context: dict[str, Any], seed: int, episode_steps: int):
    """Create one HOT environment for a context from configs/contexts_min.yaml.

    Replace this body after installing the HOT toolkit and EnergyPlus on cloud.
    The adapter should map fields such as:

    - context["building"]
    - context["climate_zone"]
    - context["location"]
    - context["weather_type"]
    - context["occupancy_schedule"]
    - context["thermal_scenario"]

    to the concrete HOT building/weather files.
    """

    raise NotImplementedError(
        "Install HOT/EnergyPlus and implement create_env() in a copied adapter file. "
        "Do not edit this template directly for production runs."
    )

