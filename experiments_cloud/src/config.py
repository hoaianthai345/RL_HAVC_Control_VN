from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_contexts(path: str | Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    contexts = data.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"No contexts found in {path}")
    return contexts


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

