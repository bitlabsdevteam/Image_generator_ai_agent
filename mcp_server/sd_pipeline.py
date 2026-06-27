"""Stable Diffusion pipeline wrapper (diffusers on Apple-Silicon MPS).

Loads the model once (module-level singleton) and renders images in fp16 on the MPS
(Metal) backend. Falls back to CPU if MPS is unavailable. Tuned for an M1 Pro / 32GB.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Stop-words and style tokens to drop when building a readable filename from the prompt.
_SLUG_SKIP = {
    "a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for", "is", "are",
    "semi", "3d", "anime", "style", "render", "glossy", "stylized", "character", "art",
    "detailed", "highly", "vibrant", "colors", "cinematic", "lighting", "soft", "big",
    "expressive", "eyes", "smooth", "shading",
}


def _slugify_prompt(prompt: str, max_words: int = 6) -> str:
    """Build a short, readable filename slug from the subject words of a prompt.

    Drops style keywords and stop-words so the name reflects the scene, e.g.
    'American teenagers, laughing and dancing at a party, semi-3D anime style' ->
    'american-teenagers-laughing-dancing-party'.
    """
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())
    kept = [w for w in words if w not in _SLUG_SKIP]
    slug = "-".join((kept or words)[:max_words])
    return slug or "image"

from .config import CONFIG
from .styles import apply_style
from . import pose_control

# Cache of loaded pipelines keyed by model name, so repeated calls are cheap.
_PIPELINES: dict[str, object] = {}
# ControlNet pipelines are heavier; cache them separately keyed by (model, controlnet_repo).
_CONTROLNET_PIPELINES: dict[tuple[str, str], object] = {}


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

    if spec["arch"] == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        # SDXL on MPS must run fp32: the UNet overflows to NaN (pure-black images) in fp16
        # on Metal — even with an fp16-fix VAE (the VAE fix only covers VAE decode). fp32 is
        # reliable (~2-3 min/image at 1024px on M1). Use fp16 on CUDA where it's stable.
        dtype = torch.float16 if device == "cuda" else torch.float32
        # NOTE: logs go to stderr — stdout is the MCP JSON-RPC transport and must stay clean.
        print(f"[sd] loading {spec['repo']} (sdxl) on {device}/{dtype} ...",
              file=sys.stderr, flush=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            spec["repo"], torch_dtype=dtype, use_safetensors=True
        )
    else:
        from diffusers import StableDiffusionPipeline

        # SD1.5: honor config dtype (fp32 on MPS avoids its own fp16 VAE NaN bug).
        dtype = _dtype(img["dtype"]) if device != "cpu" else torch.float32
        print(f"[sd] loading {spec['repo']} (sd15) on {device}/{dtype} ...",
              file=sys.stderr, flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            spec["repo"], torch_dtype=dtype, safety_checker=None
        )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # memory-friendly on unified-memory Macs
    pipe.set_progress_bar_config(disable=False)

    _PIPELINES[model_name] = pipe
    return pipe


def get_controlnet_pipeline(controls: tuple[str, ...], model_name: str | None = None):
    """Lazily load an SDXL multi-ControlNet pipeline for the given ordered ``controls``.

    ``controls`` are config keys under ``image.controlnet`` (e.g. ("openpose",) or
    ("openpose","depth")). Each ControlNet adds ~2.5GB, so pipelines are built and cached only
    when conditioning is actually requested, keyed by (model, controls).
    """
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    img = CONFIG["image"]
    model_name = model_name or img["default_model"]
    spec = img["models"][model_name]
    if spec["arch"] != "sdxl":
        raise ValueError(f"ControlNet path requires an SDXL model, got {model_name} ({spec['arch']})")

    key = (model_name, controls)
    if key in _CONTROLNET_PIPELINES:
        return _CONTROLNET_PIPELINES[key]

    device = _resolve_device(img["device"])
    # Same MPS rule as the base pipeline: fp32 on Metal (fp16 UNet -> NaN/black), fp16 on CUDA.
    dtype = torch.float16 if device == "cuda" else torch.float32
    cn_cfg = img["controlnet"]
    nets = [
        ControlNetModel.from_pretrained(cn_cfg[name]["repo"], torch_dtype=dtype, use_safetensors=True)
        for name in controls
    ]
    print(f"[sd] loading ControlNet {list(controls)} + {spec['repo']} on {device}/{dtype} ...",
          file=sys.stderr, flush=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        spec["repo"], controlnet=(nets if len(nets) > 1 else nets[0]),
        torch_dtype=dtype, use_safetensors=True,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)

    _CONTROLNET_PIPELINES[key] = pipe
    return pipe


def _layout_map(plan: dict, width: int, height: int):
    """A coarse layout image (filled silhouettes per entity) for a canny/depth ControlNet.

    Nearer entities (later in the plan, or the subject of an 'on top of'/'in front of'
    relation) are drawn brighter so a depth ControlNet reads figure-over-object ordering.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    entities = plan.get("entities", [])
    nearer = {r["subject"] for r in plan.get("relations", [])
              if r["predicate"] in ("on top of", "on", "standing on", "in front of", "above")}
    for i, e in enumerate(entities):
        x0, y0, x1, y1 = (e["box"][0] * width, e["box"][1] * height,
                          e["box"][2] * width, e["box"][3] * height)
        depth = 200 if e["id"] in nearer else 90 + int(60 * i / max(1, len(entities)))
        draw.ellipse([x0, y0, x1, y1], fill=(depth, depth, depth))
    return img


def _build_control_images(plan: dict, controls: tuple[str, ...], width: int, height: int):
    """Build the control image for each requested control from the scene plan.

    Returns (active_controls, images) keeping only controls whose image could be built — so a
    plan with no humans drops 'openpose' and the pipeline is sized to what actually applies.
    """
    active: list[str] = []
    images: list = []
    for name in controls:
        if name == "openpose":
            sk = pose_control.skeletons_from_plan(plan, width, height)
            if sk is None:
                continue
            img = sk
        elif name in ("depth", "canny"):
            img = _layout_map(plan, width, height)
        else:
            continue
        active.append(name)
        images.append(img)
    return tuple(active), images


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
    scene_plan: dict | None = None,
) -> dict:
    """Render one image and write it (plus sidecar metadata) to the outputs dir.

    If ``negative_prompt`` is None the style preset supplies one; the style tags are
    appended to ``prompt`` so callers can pass either a bare subject or a full prompt.

    If ``scene_plan`` is given (see ``scene_planner``), its entity boxes/poses are turned into
    ControlNet conditioning so spatial relations and poses ("standing on top of X", "arms up")
    are enforced structurally — what CLIP text guidance cannot do. Plans with no usable control
    (e.g. no humans, controls disabled) fall back to plain text-to-image.
    """
    img = CONFIG["image"]
    device = _resolve_device(img["device"])

    steps = steps or img["steps"]
    width = width or img["width"]
    height = height or img["height"]

    # Decide on conditioning from the plan + configured controls.
    cn_cfg = img.get("controlnet", {})
    wanted = tuple(cn_cfg.get("controls", []) or ())
    active, control_images = (
        _build_control_images(scene_plan, wanted, width, height) if (scene_plan and wanted)
        else ((), [])
    )
    use_control = bool(active)
    pipe = get_controlnet_pipeline(active, model) if use_control else get_pipeline(model)

    # Apply the style preset ONLY when a style is given. Callers that already styled the
    # prompt (e.g. via enhance_prompt) must pass style=None to avoid double-application,
    # which would duplicate the style suffix and overflow CLIP's 77-token limit.
    if style:
        full_prompt, preset_negative = apply_style(prompt, style)
        negative = negative_prompt or preset_negative
    else:
        full_prompt = prompt
        negative = negative_prompt or ""

    guidance_scale = guidance_scale if guidance_scale is not None else img["guidance_scale"]
    seed = img["seed"] if seed is None else seed
    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % (2**32)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # ControlNet kwargs only on the control path; the plain pipeline call is untouched otherwise.
    extra: dict = {}
    if use_control:
        scales = [float(cn_cfg[name]["conditioning_scale"]) for name in active]
        extra = {
            "image": control_images if len(control_images) > 1 else control_images[0],
            "controlnet_conditioning_scale": scales if len(scales) > 1 else scales[0],
        }

    t0 = time.time()
    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        **extra,
    )
    elapsed = round(time.time() - t0, 1)
    image = result.images[0]

    out_dir = Path(CONFIG["paths"]["outputs"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Readable, descriptive filename: "<subject-slug>_<timestamp>.png" (timestamp keeps it
    # unique). The seed is retained in the sidecar metadata for reproducibility.
    slug = _slugify_prompt(full_prompt)
    png_path = out_dir / f"{slug}_{stamp}.png"
    image.save(png_path)
    # Persist the control map(s) next to the render for debugging/reproducibility.
    for name, cimg in zip(active, control_images):
        cimg.save(out_dir / f"{slug}_{stamp}_control_{name}.png")

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
        "controls": list(active),
        "controlnet_conditioning_scale": extra.get("controlnet_conditioning_scale"),
        "elapsed_sec": elapsed,
    }
    with open(png_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[sd] wrote {png_path} in {elapsed}s", file=sys.stderr, flush=True)
    return meta
