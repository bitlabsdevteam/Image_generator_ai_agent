"""MCP server exposing the semi-3D-anime image-generation toolset.

Runs over stdio so any MCP client (the bundled agent, Claude Desktop, an inspector, ...)
can reuse these tools. Launch standalone with:  python -m mcp_server.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import CONFIG
from . import evals as _evals
from . import sd_pipeline
from . import scene_planner as _planner
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
def plan_scene(user_request: str) -> dict:
    """Parse a request into a structured scene plan (entities, spatial relations, boxes, poses).

    The plan drives ControlNet conditioning and the verifier's checklist, so spatial
    relationships and poses are enforced structurally rather than left to CLIP text guidance.
    """
    return _planner.plan_scene(user_request)


@mcp.tool()
def generate_image(
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
    """Render an image with Stable Diffusion (diffusers on MPS) and save it.

    Returns metadata including the output PNG path. Style tags are auto-appended to the
    prompt; pass negative_prompt to override the style's default negatives. Pass a
    scene_plan (from plan_scene) to enforce poses/placement via ControlNet — the reliable
    fix when text alone can't pin down composition.
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
        scene_plan=scene_plan,
    )


@mcp.tool()
def verify_image(image_path: str, scene_plan: dict) -> dict:
    """Check an image against a scene plan's constraints with the local vision model.

    Returns per-constraint pass/fail (pose, placement, relation) — the structured spatial gate
    that CLIP scoring cannot provide.
    """
    return _evals.verify_constraints(image_path, scene_plan)


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
