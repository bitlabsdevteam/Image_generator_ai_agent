"""Thin Ollama wrapper for the agent's reasoning and prompt-refinement."""
from __future__ import annotations

import ollama

from mcp_server.config import CONFIG


def chat(messages: list[dict], temperature: float | None = None) -> str:
    """Run a chat completion against the configured local model."""
    resp = ollama.chat(
        model=CONFIG["llm"]["model"],
        messages=messages,
        options={"temperature": temperature if temperature is not None else CONFIG["llm"]["temperature"]},
    )
    return resp["message"]["content"].strip()


def refine_prompt(subject: str, prev_prompt: str, eval_result: dict) -> str:
    """Ask the LLM to rewrite the SD prompt to fix the weakest eval dimension.

    Returns a bare subject prompt (style tags are re-applied downstream).
    """
    weak = []
    if eval_result["clip_score"] < eval_result["thresholds"]["clip"]:
        weak.append("the image does not match the request well (low prompt alignment) — "
                    "make the subject, action and setting more explicit and prominent")
    if eval_result["style_sim"] < eval_result["thresholds"]["style"]:
        weak.append("the style drifted from semi-3D anime — describe a glossy stylized 3D "
                    "character render with big expressive eyes and soft studio lighting")

    guidance = " Also, ".join(weak) or "improve overall composition and clarity"
    msgs = [
        {"role": "system", "content": (
            "You rewrite Stable Diffusion prompts. Output ONLY the improved comma-separated "
            "prompt text for the SUBJECT (no style keywords like 3d/anime, no quotes, no notes)."
        )},
        {"role": "user", "content": (
            f"Original request: {subject}\n"
            f"Previous prompt: {prev_prompt}\n"
            f"Problem to fix: {guidance}\n"
            "Rewrite the subject prompt to address this."
        )},
    ]
    out = chat(msgs).strip().strip('"')
    return out or subject
