"""Load and expose ``config.yaml`` to the server, pipeline, evals and agent."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

# Project root = parent of this file's package directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    """Parse config.yaml once and resolve output/sample paths to absolute."""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["outputs"] = str(ROOT / cfg["paths"]["outputs"])
    cfg["paths"]["samples"] = str(ROOT / cfg["paths"]["samples"])
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    return cfg


CONFIG = load_config()
