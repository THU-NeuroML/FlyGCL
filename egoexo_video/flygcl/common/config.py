from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

import yaml


WORKSPACE = Path(__file__).resolve().parents[2]


def _merge(base: Dict, override: Dict) -> Dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("_base_", None)
    if parent:
        config = _merge(load_config(path.parent / parent), config)
    config["config_path"] = str(path)
    return config


def resolve_data_paths(config: Dict) -> Dict[str, Path]:
    data = config["data"]
    return {
        key: (WORKSPACE / value).resolve() if not Path(value).is_absolute() else Path(value)
        for key, value in data.items()
        if key.endswith("_root") or key.endswith("_pairs")
    }
