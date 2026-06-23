"""LLM-backed prompt engineering used by the `enhance_prompt` MCP tool.

Uses the local Ollama model to expand a short user request into a rich Stable Diffusion
prompt, then merges in the style preset's tags/negatives. Degrades gracefully to a
template if Ollama is unavailable, so image generation never hard-fails on the LLM.
"""
from __future__ import annotations

import sys

from .config import CONFIG
from .styles import apply_style, get_style

# Subjects must stay short: the style preset (~20 tokens) is appended, and the combined
# prompt has to fit within CLIP's 77-token limit or the model silently drops the tail.
_MAX_SUBJECT_WORDS = 28

_SYS = (
    "You are a Stable Diffusion prompt engineer. Rewrite the user's request into ONE short, "
    "vivid, comma-separated image prompt (MAX 25 words) naming the subject, action, setting "
    "and mood. Be concise — no filler. Output ONLY the prompt text: no preamble, no quotes, "
    "no explanation. Do NOT include style keywords like '3d', 'anime' or 'render'; those are "
    "added separately."
)


def _cap_words(text: str, limit: int) -> str:
    """Trim an over-long subject at a comma boundary (or hard word limit) to protect the
    77-token budget, so the appended style tags are never truncated away."""
    words = text.split()
    if len(words) <= limit:
        return text
    clipped = " ".join(words[:limit])
    # Prefer to end on the last complete clause.
    if "," in clipped:
        clipped = clipped.rsplit(",", 1)[0]
    return clipped.rstrip(" ,")


def _ollama_expand(user_request: str) -> str | None:
    try:
        import ollama

        resp = ollama.chat(
            model=CONFIG["llm"]["model"],
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user_request},
            ],
            options={"temperature": CONFIG["llm"]["temperature"]},
        )
        text = resp["message"]["content"].strip()
        # Strip accidental wrapping quotes / labels.
        text = text.strip('"').strip()
        return text or None
    except Exception as exc:  # ollama not running, model missing, etc.
        print(f"[llm] enhance fallback ({exc})", file=sys.stderr, flush=True)
        return None


def enhance(user_request: str, style: str = "semi-3d-anime") -> dict:
    """Return {"prompt", "negative_prompt", "style"} for the given request."""
    expanded = _ollama_expand(user_request) or user_request
    expanded = _cap_words(expanded, _MAX_SUBJECT_WORDS)
    positive, negative = apply_style(expanded, style)
    return {
        "prompt": positive,
        "negative_prompt": negative,
        "style": style,
        "subject": expanded,
        "style_description": get_style(style)["description"],
    }
