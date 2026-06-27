"""Multi-agent harness for the semi-3D-anime image pipeline.

The loop is split into three single-responsibility agents that pass a shared ``state``
dict between them over one MCP session (the bundled FastMCP server, spawned by
``agent.agent``). This mirrors a small Claude-Code-style agentic system: each agent owns
one job, declares the tools/skills it uses, and the orchestrator (``agent.agent.run``)
routes feedback between them.

    PromptAgent      analyse the request -> best SD prompt + scene plan;
                     on eval feedback, regenerate a better prompt (or replan structure).
        tools (MCP): enhance_prompt, plan_scene
        skills:      agent.llm.refine_prompt, agent.llm.replan_scene (local Ollama)

    GenerationAgent  render the prompt with the UNet (Stable Diffusion / SDXL + ControlNet).
        tools (MCP): generate_image

    EvalAgent        score the render with OpenCLIP, and (optionally) verify spatial
                     constraints with the local VLM.
        tools (MCP): evaluate_image (OpenCLIP), verify_image (VLM)

``cleanup_outputs`` enforces "only keep the correct version": every render but the kept
one (its PNG + sidecar JSON) is deleted at the end of a run.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp_server.config import CONFIG
from agent import llm


def parse_tool_result(result) -> dict:
    """Extract a dict payload from an MCP CallToolResult."""
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        # FastMCP wraps non-dict returns under "result"; dicts pass through.
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    for block in result.content:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return {"text": block.text}
    return {}


class PromptAgent:
    """Agent 1 — analyses the request into the best prompt and (re)generates it on feedback.

    Owns all prompt/structure reasoning: it expands the request (``enhance_prompt``), plans
    the scene (``plan_scene``), and—when the EvalAgent reports a failure—decides whether to
    fix *structure* (replan boxes/poses) or *wording* (refine the subject), since CLIP text
    guidance cannot fix spatial errors.
    """

    def __init__(self, session, style: str = "semi-3d-anime"):
        self.session = session
        self.style = style

    async def _enhance(self, request: str) -> dict:
        return parse_tool_result(await self.session.call_tool(
            "enhance_prompt", {"user_request": request, "style": self.style}))

    async def analyse(self, user_request: str) -> dict:
        """Produce the initial prompt + scene plan. Returns the shared loop ``state``."""
        enh = await self._enhance(user_request)
        state = {
            "subject": enh.get("subject", user_request),
            "prompt": enh["prompt"],
            "negative": enh["negative_prompt"],
            "scene_plan": None,
        }
        if CONFIG.get("scene_planning", {}).get("enabled", True):
            state["scene_plan"] = parse_tool_result(await self.session.call_tool(
                "plan_scene", {"user_request": user_request}))
        return state

    async def revise(self, state: dict, eval_result: dict, verify_result: dict,
                     user_request: str) -> dict:
        """Regenerate a better prompt from the eval/verify feedback.

        Structure-first: if the verifier flagged failed spatial constraints, replan the
        scene geometry. Otherwise rewrite the subject to fix the weak CLIP/style dimension
        and re-enhance it back into a full styled prompt.
        """
        if state.get("scene_plan") and verify_result.get("failed"):
            state["scene_plan"] = llm.replan_scene(
                state["scene_plan"], verify_result["failed"], user_request)
            state["last_action"] = "replan"
        else:
            state["subject"] = llm.refine_prompt(state["subject"], state["prompt"], eval_result)
            reenh = await self._enhance(state["subject"])
            state["prompt"], state["negative"] = reenh["prompt"], reenh["negative_prompt"]
            state["last_action"] = "refine"
        return state


class GenerationAgent:
    """Agent 2 — renders the prompt with the UNet (Stable Diffusion / SDXL + ControlNet)."""

    def __init__(self, session):
        self.session = session

    async def generate(self, state: dict, gen_params: dict | None = None) -> dict:
        # style=None: the PromptAgent already applied the style preset; re-applying would
        # duplicate tags and overflow CLIP's 77-token limit.
        return parse_tool_result(await self.session.call_tool("generate_image", {
            "prompt": state["prompt"], "negative_prompt": state["negative"], "style": None,
            "scene_plan": state.get("scene_plan"), **(gen_params or {}),
        }))


class EvalAgent:
    """Agent 3 — judges the render with OpenCLIP and (optionally) the VLM constraint verifier.

    Returns a verdict the orchestrator uses to stop, rank, or trigger PromptAgent feedback.
    """

    def __init__(self, session, verify_on: bool = True):
        self.session = session
        self.verify_on = verify_on

    async def evaluate(self, image_path: str, user_request: str,
                       scene_plan: dict | None) -> dict:
        # OpenCLIP scoring against the ORIGINAL request (clip_score, style_sim, aesthetic).
        ev = parse_tool_result(await self.session.call_tool(
            "evaluate_image", {"image_path": image_path, "prompt": user_request}))

        vr = {"available": False, "all_ok": None, "failed": [], "constraints": []}
        if self.verify_on and scene_plan:
            vr = parse_tool_result(await self.session.call_tool(
                "verify_image", {"image_path": image_path, "scene_plan": scene_plan}))

        # A render is "correct" only if on-style AND every spatial constraint holds (when the
        # verifier is available). Composite weights satisfied constraints heavily so a
        # spatially-correct image outranks a prettier wrong one.
        style_ok = ev["style_sim"] >= ev["thresholds"]["style"]
        constraints_ok = (vr.get("all_ok") is True) or (not vr.get("available"))
        n_ok = sum(1 for c in vr.get("constraints", []) if c.get("ok"))
        return {
            "eval": ev,
            "verify": vr,
            "style_ok": style_ok,
            "constraints_ok": constraints_ok,
            "passed": style_ok and constraints_ok,
            "composite": round(ev["composite"] + n_ok, 4),
        }


def cleanup_outputs(produced: list[str], keep: str | None) -> list[str]:
    """Delete every produced render except ``keep`` (its PNG + sidecar JSON).

    Enforces "only store the correct version": across a multi-iteration run the loop writes
    one PNG per attempt; this removes the rejected ones so ``outputs/`` holds only the kept
    best/passing image. Returns the list of removed PNG paths.
    """
    keep_resolved = Path(keep).resolve() if keep else None
    removed: list[str] = []
    for path in produced:
        p = Path(path).resolve()
        if keep_resolved is not None and p == keep_resolved:
            continue
        for f in (p, p.with_suffix(".json")):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        removed.append(str(p))
    return removed
