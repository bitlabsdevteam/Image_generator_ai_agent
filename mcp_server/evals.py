"""Evaluation utilities.

- clip_score: CLIP cosine similarity between the image and the text prompt (prompt match).
- style_sim:  mean CLIP cosine similarity between the image and the reference samples in
              ``images/`` (semi-3D-anime style fidelity).
- aesthetic:  lightweight, normalized proxy derived from CLIP image-feature norm — no extra
              weights to download; good enough as a relative quality signal for the loop.
- vision_judge (optional): natural-language grade from a multimodal Ollama model.

The CLIP model loads once and is reused across calls.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch
from PIL import Image

from .config import CONFIG

# ViT-L/14 (Large, 14px patches) — more discriminating image-text scoring than ViT-B/32,
# and better aligned with how SDXL renders. Used only for evaluation/ranking, not generation.
_CLIP_NAME = "ViT-L-14"
_CLIP_PRETRAINED = "laion2b_s32b_b82k"


@functools.lru_cache(maxsize=1)
def _load_clip():
    import open_clip

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        _CLIP_NAME, pretrained=_CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(_CLIP_NAME)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


@functools.lru_cache(maxsize=1)
def _sample_features() -> torch.Tensor | None:
    """CLIP image features for each reference sample in the samples dir (normalized)."""
    model, preprocess, _, device = _load_clip()
    sample_dir = Path(CONFIG["paths"]["samples"])
    paths = [p for p in sample_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not paths:
        return None
    feats = []
    with torch.no_grad():
        for p in paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(img)
            feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats, dim=0)


def _image_features(image_path: str) -> torch.Tensor:
    model, preprocess, _, device = _load_clip()
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(img)
    return f / f.norm(dim=-1, keepdim=True)


def clip_score(image_path: str, prompt: str) -> float:
    """Cosine similarity between image and prompt embeddings, mapped to 0..1."""
    model, _, tokenizer, device = _load_clip()
    img_f = _image_features(image_path)
    with torch.no_grad():
        txt = tokenizer([prompt]).to(device)
        txt_f = model.encode_text(txt)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    cos = float((img_f @ txt_f.T).squeeze().item())
    return round(cos, 4)


def style_sim(image_path: str) -> float:
    """Mean cosine similarity to the reference samples (0..1). 0.0 if no samples."""
    samples = _sample_features()
    if samples is None:
        return 0.0
    img_f = _image_features(image_path)
    cos = float((img_f @ samples.T).mean().item())
    return round(cos, 4)


def aesthetic(image_path: str) -> float:
    """Lightweight aesthetic proxy in ~0..10 from CLIP feature magnitude.

    Not a trained aesthetic predictor (avoids an extra download); use it as a relative
    signal, not an absolute quality grade.
    """
    model, preprocess, _, device = _load_clip()
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(img)
    norm = float(f.norm(dim=-1).item())
    # Map a typical CLIP norm range (~8..16) onto ~3..9.
    score = max(0.0, min(10.0, (norm - 8.0) / 8.0 * 6.0 + 3.0))
    return round(score, 2)


def evaluate(image_path: str, prompt: str) -> dict:
    """Run the loop-gating evals and return a verdict against config thresholds."""
    cfg = CONFIG["evals"]
    cs = clip_score(image_path, prompt)
    ss = style_sim(image_path)
    aes = aesthetic(image_path)
    passed = cs >= cfg["clip_threshold"] and ss >= cfg["style_threshold"]
    # Composite score used to pick the best image across loop iterations.
    composite = round(cs + ss + aes / 20.0, 4)
    return {
        "clip_score": cs,
        "style_sim": ss,
        "aesthetic": aes,
        "composite": composite,
        "passed": passed,
        "thresholds": {"clip": cfg["clip_threshold"], "style": cfg["style_threshold"]},
    }


def verify_constraints(image_path: str, scene_plan: dict) -> dict:
    """Have a multimodal model grade each scene-plan constraint (pose, placement, relation).

    This is the structured spatial gate CLIP cannot provide: it checks *layout*, not concept
    presence. Returns {"available", "all_ok", "constraints":[{id, ok, reason}], "failed":[...]}.
    Degrades gracefully (available=False) if the VLM is down or returns unparseable output, so
    the loop can fall back to the style/quality gate rather than block.
    """
    from . import scene_planner
    from ._jsonio import extract_json

    checklist = scene_planner.constraints_from_plan(scene_plan)
    if not checklist:
        return {"available": True, "all_ok": True, "constraints": [], "failed": []}

    model = CONFIG.get("verify", {}).get("model") or CONFIG["llm"]["vision_model"]
    lines = "\n".join(f'- id="{c["id"]}": {c["text"]}' for c in checklist)
    msg = (
        "You are verifying whether an AI-generated image satisfies specific composition "
        "constraints. Judge ONLY what is visible. For EACH constraint answer strictly.\n"
        f"Constraints:\n{lines}\n\n"
        'Output ONLY JSON: {"constraints":[{"id":"<id>","ok":true|false,'
        '"reason":"<short>"}]}. ok=true only if clearly satisfied.'
    )
    try:
        import ollama

        resp = ollama.chat(model=model,
                           messages=[{"role": "user", "content": msg, "images": [image_path]}])
        data = extract_json(resp["message"]["content"])
    except Exception:
        data = None

    if not isinstance(data, dict) or not isinstance(data.get("constraints"), list):
        return {"available": False, "all_ok": None, "constraints": [], "failed": []}

    graded = {str(c.get("id")): c for c in data["constraints"] if isinstance(c, dict)}
    results = []
    for c in checklist:
        g = graded.get(c["id"], {})
        ok = bool(g.get("ok", False))
        results.append({"id": c["id"], "text": c["text"], "ok": ok,
                        "reason": str(g.get("reason", ""))})
    failed = [r for r in results if not r["ok"]]
    return {"available": True, "all_ok": not failed, "constraints": results, "failed": failed,
            "judge_model": model}


def vision_judge(image_path: str, prompt: str) -> dict:
    """Optional qualitative grade from a multimodal Ollama model. Slow; off the loop path."""
    import ollama

    model = CONFIG["llm"]["vision_model"]
    msg = (
        f"You are grading an AI-generated image for the request: '{prompt}'. "
        "The target style is semi-3D anime (glossy stylized 3D render, big expressive eyes, "
        "vibrant colors). In 2-3 sentences, judge relevance, style fidelity, and quality, "
        "then give a score out of 10."
    )
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": msg, "images": [image_path]}],
    )
    return {"judge_model": model, "notes": resp["message"]["content"]}
