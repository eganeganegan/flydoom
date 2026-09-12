"""Small YAML composition layer supporting Hydra-style ``group=name`` arguments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _value(text: str) -> Any:
    return yaml.safe_load(text)


def load_config(arguments: list[str], config_root: str | Path = "configs") -> dict[str, Any]:
    """Compose group YAML files and apply dotted scalar overrides."""
    root = Path(config_root)
    config: dict[str, Any] = {}
    overrides: list[tuple[str, str]] = []
    for argument in arguments:
        if "=" not in argument:
            raise ValueError(f"Expected key=value argument, received {argument!r}")
        key, value = argument.split("=", 1)
        candidate = root / key / f"{value}.yaml"
        if "." not in key and candidate.exists():
            loaded = yaml.safe_load(candidate.read_text()) or {}
            config = _merge(config, loaded)
        else:
            overrides.append((key, value))
    for dotted_key, raw_value in overrides:
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _value(raw_value)
    return config
