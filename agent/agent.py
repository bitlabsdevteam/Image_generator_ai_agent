"""Agent harness: orchestrates the semi-3D-anime image generation loop.

This is an MCP *client*. It launches the bundled MCP server over stdio and drives its
tools (enhance_prompt -> generate_image -> evaluate_image) in a refine loop, using the
local Ollama LLM (via agent.llm) to rewrite prompts when evals fall short. Context is
kept bounded by agent.context.ContextManager.

Usage:
    python -m agent.agent "American teenagers having fun at a party"
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server.config import CONFIG, ROOT
from agent import llm
from agent import interview
from agent.context import ContextManager

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()


def _parse(result) -> dict:
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


async def run(user_request: str, *, interactive: bool = False) -> dict:
    ctx = ContextManager(system_prompt=SYSTEM_PROMPT, max_tokens=3000)
    ctx.add("user", f"Request: {user_request}")

    server = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT)
    )

    max_iters = CONFIG["loop"]["max_iters"]
    best: dict | None = None

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            print(f"[agent] connected to MCP server; tools: {tools}\n", flush=True)

            # 0) Optional clarifying interview: the LLM analyses the request and asks the
            #    user 0-4 multiple-choice questions, then folds answers into the request,
            #    chosen style, and generation params. Skipped (no-op) when non-interactive.
            style = "semi-3d-anime"
            gen_params: dict = {}
            if interactive:
                styles = _parse(await session.call_tool("list_styles", {}))
                styles = styles.get("result", styles) if isinstance(styles, dict) else styles
                models = _parse(await session.call_tool("list_models", {}))
                qs = interview.generate_questions(user_request, styles, models)
                answers = interview.ask(qs)
                user_request, style, gen_params = interview.compose(
                    user_request, qs, answers, styles, models)
                if qs:
                    ctx.add("user", f"Clarified request: {user_request}")
                    print(f"[agent] clarified request: {user_request}\n"
                          f"[agent] style={style} params={gen_params or '{}'}\n", flush=True)

            # 1) Enhance the short request into a rich SD prompt (LLM-backed tool).
            enh = _parse(await session.call_tool(
                "enhance_prompt", {"user_request": user_request, "style": style}
            ))
            subject = enh.get("subject", user_request)
            prompt = enh["prompt"]
            negative = enh["negative_prompt"]
            ctx.add("assistant", f"Enhanced prompt: {prompt}")
            print(f"[agent] enhanced prompt:\n  {prompt}\n", flush=True)

            # 2) Refine loop: generate -> evaluate -> (refine) until thresholds pass.
            for i in range(1, max_iters + 1):
                print(f"[agent] === iteration {i}/{max_iters} ===", flush=True)
                # style=None: enhance_prompt already applied the style preset to `prompt`;
                # re-applying here would duplicate tags and overflow CLIP's 77-token limit.
                gen = _parse(await session.call_tool("generate_image", {
                    "prompt": prompt, "negative_prompt": negative, "style": None,
                    **gen_params,
                }))
                ctx.add("tool", json.dumps(gen), kind="tool_result")

                ev = _parse(await session.call_tool("evaluate_image", {
                    "image_path": gen["path"], "prompt": user_request,
                }))
                ctx.add("tool", json.dumps(ev), kind="tool_result")
                print(f"[agent] eval: clip={ev['clip_score']} style={ev['style_sim']} "
                      f"aesthetic={ev['aesthetic']} passed={ev['passed']}", flush=True)

                record = {"iteration": i, "image": gen["path"], "prompt": prompt, "eval": ev}
                if best is None or ev["composite"] > best["eval"]["composite"]:
                    best = record

                if ev["passed"]:
                    print("[agent] thresholds met — stopping.\n", flush=True)
                    break
                if i == max_iters:
                    print("[agent] max iterations reached.\n", flush=True)
                    break

                # 3) Ask the LLM to fix the weakest dimension, then re-apply style.
                subject = llm.refine_prompt(subject, prompt, ev)
                reenh = _parse(await session.call_tool(
                    "enhance_prompt", {"user_request": subject, "style": style}
                ))
                prompt, negative = reenh["prompt"], reenh["negative_prompt"]
                ctx.add("assistant", f"Refined subject -> {subject}")
                print(f"[agent] refined prompt:\n  {prompt}\n", flush=True)

    print(f"[agent] context: {ctx.stats()}", flush=True)
    return {"request": user_request, "best": best}


def main() -> None:
    flags = {"-i", "--interactive", "-y", "--no-interactive"}
    args = [a for a in sys.argv[1:] if a not in flags]
    request = " ".join(args).strip() or "American teenagers having fun at a party"
    force_on = any(a in ("-i", "--interactive") for a in sys.argv[1:])
    force_off = any(a in ("-y", "--no-interactive") for a in sys.argv[1:])
    # Ask clarifying questions by default on a real terminal; `-y` / non-TTY skips them
    # (keeps scripts and evals/run_evals.py one-shot). `-i` forces them on.
    interactive = force_on or (not force_off and sys.stdin.isatty())
    result = asyncio.run(run(request, interactive=interactive))
    best = result["best"]
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Request : {result['request']}")
    print(f"Image   : {best['image']}")
    print(f"Scores  : clip={best['eval']['clip_score']} "
          f"style={best['eval']['style_sim']} aesthetic={best['eval']['aesthetic']} "
          f"(iteration {best['iteration']})")


if __name__ == "__main__":
    main()
