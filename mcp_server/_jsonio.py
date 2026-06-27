"""Robust extraction of a JSON object from an LLM response.

Local models often wrap JSON in ``` fences or surround it with prose despite instructions.
This mirrors the defensive parsing first written in ``agent/interview.py`` and is shared by
every component that asks Ollama for structured output (scene planner, constraint verifier,
clarifying-question interview), so the fence/brace handling lives in exactly one place.
"""
from __future__ import annotations

import json
import re


def extract_json(raw: str) -> dict | None:
    """Best-effort parse of a single JSON object from ``raw``; None if nothing parses.

    Strips ```json ... ``` fences, then falls back to the first balanced-looking ``{...}``
    block if the model added surrounding prose.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
