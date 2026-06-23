"""LLM-backed prompt engineering used by the `enhance_prompt` MCP tool.

Uses the local Ollama model to expand a short user request into a rich Stable Diffusion
prompt, then merges in the style preset's tags/negatives. Degrades gracefully to a
template if Ollama is unavailable, so image generation never hard-fails on the LLM.
"""
from __future__ import annotations

from .config import CONFIG
from .styles import apply_style, get_style

_SYS = (
    "You are a Stable Diffusion prompt engineer. Rewrite the user's request into a single "
    "vivid, comma-separated image prompt describing subject, action, setting, mood, framing "
    "and lighting. Output ONLY the prompt text — no preamble, no quotes, no explanation. "
    "Do not include style keywords like '3d' or 'anime'; those are added separately."
)


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
        print(f"[llm] enhance fallback ({exc})", flush=True)
        return None


def enhance(user_request: str, style: str = "semi-3d-anime") -> dict:
    """Return {"prompt", "negative_prompt", "style"} for the given request."""
    expanded = _ollama_expand(user_request) or user_request
    positive, negative = apply_style(expanded, style)
    return {
        "prompt": positive,
        "negative_prompt": negative,
        "style": style,
        "subject": expanded,
        "style_description": get_style(style)["description"],
    }
