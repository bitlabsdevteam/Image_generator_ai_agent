"""Claude-Code-style clarifying questions before image generation.

Given a short user request, ask the local Ollama LLM to generate 0-4 targeted
multiple-choice questions (creative *and* technical), render them as numbered CLI menus,
and fold the answers back into an enriched request + style + generation params.

Degrades gracefully: if Ollama is down or returns unparseable output, `generate_questions`
returns `[]` and the agent proceeds exactly as if no interview happened — mirroring the
fallback contract in `mcp_server/llm_prompt.py:_ollama_expand`.
"""
from __future__ import annotations

import json
import re
import sys

from agent import llm

# Cap to avoid question fatigue (matches the "dynamic 0-4" design decision).
_MAX_QUESTIONS = 4
_VALID_KINDS = {"creative", "style", "model", "quality"}
# Quality preset -> SD sampling steps. "balanced" matches config.yaml's default (25).
_QUALITY_STEPS = {"fast": 15, "balanced": 25, "quality": 40}

_SYS = (
    "You are a creative director for a semi-3D-anime image generator. Given a short image "
    "request, decide whether a FEW clarifying questions would meaningfully improve the result. "
    "Ask between 0 and 4 questions — fewer (or none) if the request is already specific, more "
    "if it is vague. Each question MUST offer 2-4 concrete options.\n"
    "Question kinds:\n"
    " - 'creative': mood, setting, subject details, motifs (free-form option text).\n"
    " - 'style': art-style preset. Options MUST be exact names from the provided styles.\n"
    " - 'model': image model. Options MUST be exact names from the provided models.\n"
    " - 'quality': speed/quality tradeoff. Options MUST be from: fast, balanced, quality.\n"
    "Prefer creative questions; include at most one of each technical kind, and only when it "
    "would genuinely help.\n"
    "Output ONLY strict JSON, no prose, no code fences, in exactly this shape:\n"
    '{"questions": [{"id": "mood", "kind": "creative", "question": "...", '
    '"options": ["...", "..."]}]}'
)


def generate_questions(user_request: str, styles: list, models: dict) -> list[dict]:
    """Ask the LLM for 0-4 clarifying questions. Returns [] on any failure (graceful skip)."""
    style_names = [s.get("name", "") for s in styles if isinstance(s, dict)]
    model_names = list((models or {}).get("models", {}).keys())
    user = (
        f"Image request: {user_request}\n"
        f"Available styles: {', '.join(style_names) or 'semi-3d-anime'}\n"
        f"Available models: {', '.join(model_names) or 'dreamshaper-xl'}\n"
        "Generate the clarifying questions as JSON."
    )
    try:
        raw = llm.chat([
            {"role": "system", "content": _SYS},
            {"role": "user", "content": user},
        ])
        return _parse_questions(raw)
    except Exception as exc:  # ollama down, model missing, etc.
        print(f"[interview] skipped — question generation failed ({exc})",
              file=sys.stderr, flush=True)
        return []


def _parse_questions(raw: str) -> list[dict]:
    """Defensively extract a validated question list from an LLM response."""
    text = raw.strip()
    # Strip ```json ... ``` fences if the model added them despite instructions.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    # Fall back to the first {...} block if there's surrounding prose.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    questions: list[dict] = []
    for i, q in enumerate(data.get("questions", [])):
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        kind = str(q.get("kind", "creative")).strip().lower()
        options = [str(o).strip() for o in q.get("options", []) if str(o).strip()]
        if not question or kind not in _VALID_KINDS or len(options) < 2:
            continue
        qid = str(q.get("id") or f"q{i}").strip()
        questions.append({"id": qid, "kind": kind, "question": question,
                          "options": options[:4]})
        if len(questions) >= _MAX_QUESTIONS:
            break
    return questions


def ask(questions: list[dict]) -> dict[str, str]:
    """Render each question as a numbered menu; collect answers from stdin.

    Returns {question_id: answer} for answered questions only (skipped ones are omitted).
    """
    answers: dict[str, str] = {}
    if not questions:
        return answers

    print("\n[agent] A few quick questions to shape your image "
          "(pick a number, type your own, or press Enter to skip):\n", flush=True)
    for n, q in enumerate(questions, 1):
        print(f"Q{n}. {q['question']}", flush=True)
        for j, opt in enumerate(q["options"], 1):
            print(f"  [{j}] {opt}", flush=True)
        try:
            choice = input("  > ").strip()
        except EOFError:
            break
        if not choice:
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(q["options"]):
            answers[q["id"]] = q["options"][int(choice) - 1]
        else:
            answers[q["id"]] = choice
    print(flush=True)
    return answers


def compose(user_request: str, questions: list[dict], answers: dict[str, str],
            styles: list, models: dict) -> tuple[str, str, dict]:
    """Fold answers into (enriched_request, style, gen_params)."""
    by_id = {q["id"]: q for q in questions}
    style_names = {s.get("name") for s in styles if isinstance(s, dict)}
    model_names = set((models or {}).get("models", {}).keys())

    style = "semi-3d-anime"
    gen_params: dict = {}
    creative_bits: list[str] = []

    for qid, answer in answers.items():
        kind = by_id.get(qid, {}).get("kind", "creative")
        if kind == "style" and answer in style_names:
            style = answer
        elif kind == "model" and answer in model_names:
            gen_params["model"] = answer
        elif kind == "quality" and answer.lower() in _QUALITY_STEPS:
            gen_params["steps"] = _QUALITY_STEPS[answer.lower()]
        else:  # creative (or an unrecognized technical answer treated as a hint)
            creative_bits.append(answer)

    enriched = user_request
    if creative_bits:
        enriched = f"{user_request}. {', '.join(creative_bits)}"
    return enriched, style, gen_params
