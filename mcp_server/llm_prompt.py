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
# ~34 words leaves room for the style suffix while preserving spatial/colour clauses that a
# tighter cap would force the LLM to drop.
_MAX_SUBJECT_WORDS = 34

# The enhancer returns TWO things, because in CLIP-guided SD the negative prompt is the most
# reliable compositional steering tool — yet a generic style-only negative can't suppress the
# failure modes a *specific* scene invites (e.g. a "massive hamburger" tiling into a burger
# wall, or "hands up" collapsing to arms-down). So the LLM also names this scene's likely
# failure modes, which we merge into the negative prompt.
_SYS = (
    "You are an expert SDXL prompt engineer. From the user's request output EXACTLY two lines "
    "and nothing else:\n"
    "SUBJECT: <one vivid comma-separated prompt, MAX 32 words. Put the MAIN SUBJECT and its "
    "KEY POSE/ACTION FIRST and state the pose explicitly (e.g. 'both arms raised high overhead, "
    "cheering'). PRESERVE spatial relationships verbatim ('standing on top of', 'inside', "
    "'behind' — never reduce 'on top of a X' to 'X top'). Force countable scene objects to the "
    "SINGULAR the request implies ('a single giant hamburger', not 'hamburgers'). PRESERVE "
    "named colour palettes, brands and counts ('McDonald's yellow-red palette', 'twin tails'). "
    "NO style keywords like 3d/anime/render — those are added later.>\n"
    "AVOID: <comma-separated list of likely failure modes to EXCLUDE for THIS scene: wrong "
    "object counts, wrong pose, wrong layout, wrong colours (e.g. for a girl atop one giant "
    "burger cheering: 'multiple hamburgers, rows of burgers, burger wall, arms down, hands at "
    "sides, lowered arms, neutral pose').>"
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


def _parse_subject_avoid(text: str) -> tuple[str, str]:
    """Pull the SUBJECT / AVOID lines out of the LLM response. Tolerates missing labels by
    falling back to the whole text as the subject and an empty scene-negative."""
    subject, avoid = "", ""
    for line in text.splitlines():
        low = line.strip()
        if low.upper().startswith("SUBJECT:"):
            subject = low.split(":", 1)[1].strip().strip('"')
        elif low.upper().startswith("AVOID:"):
            avoid = low.split(":", 1)[1].strip().strip('"')
    if not subject:
        subject = text.strip().strip('"')
    return subject, avoid


def _ollama_expand(user_request: str) -> tuple[str, str] | None:
    """Return (subject, scene_negative) from the LLM, or None if Ollama is unavailable."""
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
        subject, avoid = _parse_subject_avoid(text)
        return (subject, avoid) if subject else None
    except Exception as exc:  # ollama not running, model missing, etc.
        print(f"[llm] enhance fallback ({exc})", file=sys.stderr, flush=True)
        return None


def _merge_negative(scene_negative: str, style_negative: str) -> str:
    """Prepend the scene-specific failure modes to the style negatives (scene first so the
    most important suppressions aren't truncated if the negative ever runs long), de-duped."""
    seen: set[str] = set()
    parts: list[str] = []
    for chunk in (scene_negative, style_negative):
        for term in (t.strip() for t in chunk.split(",")):
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                parts.append(term)
    return ", ".join(parts)


def enhance(user_request: str, style: str = "semi-3d-anime") -> dict:
    """Return {"prompt", "negative_prompt", "style", ...} for the given request.

    The positive prompt is the LLM-built subject (pose-first, singular-forced) plus the style
    tags; the negative prompt is this scene's LLM-named failure modes merged with the style
    preset's generic quality negatives.
    """
    expanded = _ollama_expand(user_request)
    if expanded:
        subject, scene_negative = expanded
    else:
        subject, scene_negative = user_request, ""
    subject = _cap_words(subject, _MAX_SUBJECT_WORDS)
    positive, style_negative = apply_style(subject, style)
    negative = _merge_negative(scene_negative, style_negative)
    return {
        "prompt": positive,
        "negative_prompt": negative,
        "style": style,
        "subject": subject,
        "scene_negative": scene_negative,
        "style_description": get_style(style)["description"],
    }
