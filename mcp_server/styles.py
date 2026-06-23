"""Style presets for the image generator.

Each preset contributes positive style tags and a negative prompt that steer Stable
Diffusion toward a target aesthetic. The "semi-3d-anime" preset is tuned to match the
reference banners in ``images/`` — glossy Pixar-meets-anime mobile-game character renders.
"""
from __future__ import annotations

STYLES: dict[str, dict] = {
    "semi-3d-anime": {
        "description": (
            "Glossy semi-3D anime / mobile-game character render: smooth subsurface "
            "shading, big expressive eyes, clean stylized features, soft studio lighting."
        ),
        # Appended to the subject prompt to lock in the look. Kept short (~20 tokens) so the
        # subject + style stays within CLIP's 77-token limit and the style is never truncated.
        "positive": (
            "semi-3D anime style, glossy stylized 3D render, big expressive eyes, "
            "vibrant colors, soft cinematic lighting, highly detailed"
        ),
        "negative": (
            "flat 2d, line art, sketch, lowres, blurry, bad anatomy, deformed, "
            "extra limbs, extra fingers, mutated hands, watermark, text, signature, "
            "jpeg artifacts, ugly, grainy, photorealistic, realistic photo, dull colors"
        ),
    },
    "anime-2d": {
        "description": "Flat cel-shaded 2D anime illustration.",
        "positive": (
            "2D anime illustration, cel shading, clean line art, vibrant colors, "
            "detailed, sharp focus"
        ),
        "negative": "3d render, photorealistic, lowres, blurry, bad anatomy, watermark, text",
    },
}

DEFAULT_STYLE = "semi-3d-anime"


def get_style(name: str | None) -> dict:
    """Return the style preset, falling back to the default."""
    return STYLES.get(name or DEFAULT_STYLE, STYLES[DEFAULT_STYLE])


def apply_style(subject_prompt: str, style: str | None) -> tuple[str, str]:
    """Combine a subject prompt with a style preset.

    Returns ``(positive_prompt, negative_prompt)``.
    """
    preset = get_style(style)
    positive = f"{subject_prompt.strip()}, {preset['positive']}"
    return positive, preset["negative"]


def list_styles() -> list[dict]:
    """Introspection helper for MCP clients."""
    return [{"name": n, "description": s["description"]} for n, s in STYLES.items()]
