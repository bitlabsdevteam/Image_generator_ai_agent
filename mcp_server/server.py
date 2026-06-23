"""MCP server exposing the semi-3D-anime image-generation toolset.

Runs over stdio so any MCP client (the bundled agent, Claude Desktop, an inspector, ...)
can reuse these tools. Launch standalone with:  python -m mcp_server.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import CONFIG
from . import evals as _evals
from . import sd_pipeline
from .styles import list_styles as _list_styles
from .llm_prompt import enhance as _enhance

mcp = FastMCP("semi3d-image-gen")


@mcp.tool()
def list_styles() -> list[dict]:
    """List available art-style presets (name + description)."""
    return _list_styles()


@mcp.tool()
def list_models() -> dict:
    """List the registered Stable Diffusion models and the active default."""
    img = CONFIG["image"]
    return {"default": img["default_model"], "models": img["models"], "device": img["device"]}


@mcp.tool()
def enhance_prompt(user_request: str, style: str = "semi-3d-anime") -> dict:
    """Expand a short request into a rich SD prompt + negative prompt using the local LLM.

    Returns {"prompt", "negative_prompt", "style"}.
    """
    return _enhance(user_request, style)


@mcp.tool()
def generate_image(
    prompt: str,
    negative_prompt: str | None = None,
    style: str = "semi-3d-anime",
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    model: str | None = None,
) -> dict:
    """Render an image with Stable Diffusion (diffusers on MPS) and save it.

    Returns metadata including the output PNG path. Style tags are auto-appended to the
    prompt; pass negative_prompt to override the style's default negatives.
    """
    return sd_pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        style=style,
        steps=steps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        seed=seed,
        model=model,
    )


@mcp.tool()
def evaluate_image(image_path: str, prompt: str) -> dict:
    """Score an image: clip_score, style_sim, aesthetic, composite, and pass/fail."""
    return _evals.evaluate(image_path, prompt)


@mcp.tool()
def vision_judge(image_path: str, prompt: str) -> dict:
    """Optional qualitative grade from a multimodal Ollama model (slow)."""
    return _evals.vision_judge(image_path, prompt)


if __name__ == "__main__":
    mcp.run()
