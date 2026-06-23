"""Stable Diffusion pipeline wrapper (diffusers on Apple-Silicon MPS).

Loads the model once (module-level singleton) and renders images in fp16 on the MPS
(Metal) backend. Falls back to CPU if MPS is unavailable. Tuned for an M1 Pro / 32GB.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch

from .config import CONFIG
from .styles import apply_style

# Cache of loaded pipelines keyed by model name, so repeated calls are cheap.
_PIPELINES: dict[str, object] = {}


def _resolve_device(requested: str) -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype(name: str):
    return {"float16": torch.float16, "float32": torch.float32}.get(name, torch.float16)


def get_pipeline(model_name: str | None = None):
    """Lazily load and cache a diffusers pipeline for the given registry model."""
    img = CONFIG["image"]
    model_name = model_name or img["default_model"]
    if model_name in _PIPELINES:
        return _PIPELINES[model_name]

    spec = img["models"][model_name]
    device = _resolve_device(img["device"])
    # On CPU, fp16 is unsupported/slow — use fp32 there.
    dtype = _dtype(img["dtype"]) if device != "cpu" else torch.float32

    if spec["arch"] == "sdxl":
        from diffusers import StableDiffusionXLPipeline as Pipe
    else:
        from diffusers import StableDiffusionPipeline as Pipe

    print(f"[sd] loading {spec['repo']} ({spec['arch']}) on {device}/{dtype} ...", flush=True)
    pipe = Pipe.from_pretrained(spec["repo"], torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    # Memory-friendly on unified-memory Macs.
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)

    _PIPELINES[model_name] = pipe
    return pipe


def generate(
    prompt: str,
    negative_prompt: str | None = None,
    style: str | None = "semi-3d-anime",
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    model: str | None = None,
) -> dict:
    """Render one image and write it (plus sidecar metadata) to the outputs dir.

    If ``negative_prompt`` is None the style preset supplies one; the style tags are
    appended to ``prompt`` so callers can pass either a bare subject or a full prompt.
    """
    img = CONFIG["image"]
    pipe = get_pipeline(model)
    device = _resolve_device(img["device"])

    full_prompt, preset_negative = apply_style(prompt, style)
    negative = negative_prompt or preset_negative

    steps = steps or img["steps"]
    width = width or img["width"]
    height = height or img["height"]
    guidance_scale = guidance_scale if guidance_scale is not None else img["guidance_scale"]
    seed = img["seed"] if seed is None else seed
    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % (2**32)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.time()
    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    elapsed = round(time.time() - t0, 1)
    image = result.images[0]

    out_dir = Path(CONFIG["paths"]["outputs"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = out_dir / f"img_{stamp}_{seed}.png"
    image.save(png_path)

    meta = {
        "path": str(png_path),
        "prompt": full_prompt,
        "negative_prompt": negative,
        "style": style,
        "model": model or img["default_model"],
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "device": device,
        "elapsed_sec": elapsed,
    }
    with open(png_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[sd] wrote {png_path} in {elapsed}s", flush=True)
    return meta
