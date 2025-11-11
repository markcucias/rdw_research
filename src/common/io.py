from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigError(Exception):
    pass


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return a dict.
    Expands ~ and resolves relative paths.
    Raises ConfigError with a clear message on failure.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {p}:\n{e}") from e
    except OSError as e:
        raise ConfigError(f"Could not read config {p}: {e}") from e

    if data is None:
        # empty file is allowed, but normalize to {}
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML must be a mapping (dict). Got: {type(data).__name__}")
    return data