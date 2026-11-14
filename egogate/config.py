"""YAML config loading. Experiment configs overlay configs/default.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load default config, overlay an experiment config, then overlay CLI overrides.

    `overrides` uses dotted keys: {"matcher.threshold": 0.05}.
    """
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    if path is not None:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for key, val in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg
