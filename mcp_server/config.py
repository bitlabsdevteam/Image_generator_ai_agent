"""Load and expose ``config.yaml`` to the server, pipeline, evals and agent."""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

# Project root = parent of this file's package directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Sets vars not already in the env.

    Skips blank lines, comments, and the placeholder token. Also mirrors HF_TOKEN to
    HUGGING_FACE_HUB_TOKEN so all huggingface_hub versions pick it up.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not value or value == "hf_replace_me_with_your_token":
            continue  # ignore the unset placeholder
        os.environ.setdefault(key, value)
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


_load_dotenv(ENV_PATH)


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
