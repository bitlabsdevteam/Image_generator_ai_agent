"""Agent harness: orchestrates the semi-3D-anime image generation loop.

This is an MCP *client*. It launches the bundled MCP server over stdio and drives a small
multi-agent pipeline (see agent.agents) over its tools:

    PromptAgent (analyse) -> GenerationAgent (UNet/SD) -> EvalAgent (OpenCLIP) --fail--> back

The orchestrator below runs that loop up to loop.max_iters times, feeds each failed eval
back to the PromptAgent to regenerate a better prompt, keeps the best render, and finally
deletes every rejected render so outputs/ holds only the correct version. Context is kept
bounded by agent.context.ContextManager.

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
from agent import interview
from agent.agents import (
    PromptAgent, GenerationAgent, EvalAgent, cleanup_outputs, parse_tool_result as _parse,
)
from agent.context import ContextManager

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()


async def run(user_request: str, *, interactive: bool = False) -> dict:
    ctx = ContextManager(system_prompt=SYSTEM_PROMPT, max_tokens=3000)
    ctx.add("user", f"Request: {user_request}")

    server = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT)
    )

    max_iters = CONFIG["loop"]["max_iters"]
    keep_only_best = CONFIG["loop"].get("keep_only_best", True)
    best: dict | None = None
    produced: list[str] = []  # every render written this run, for end-of-run cleanup

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

            # Wire the three single-responsibility agents over this MCP session.
            verify_on = CONFIG.get("verify", {}).get("enabled", True)
            prompt_agent = PromptAgent(session, style=style)
            gen_agent = GenerationAgent(session)
            eval_agent = EvalAgent(session, verify_on=verify_on)

            # 1) PROMPT AGENT — analyse the request into the best prompt + scene plan.
            state = await prompt_agent.analyse(user_request)
            ctx.add("assistant", f"Enhanced prompt: {state['prompt']}")
            print(f"[agent:prompt] enhanced prompt:\n  {state['prompt']}\n", flush=True)
            if state.get("scene_plan"):
                rels = ", ".join(f"{r['subject']} {r['predicate']} {r['object']}"
                                 for r in state["scene_plan"].get("relations", [])) or "none"
                ctx.add("assistant", f"Scene plan relations: {rels}")
                print(f"[agent:prompt] scene plan: "
                      f"{len(state['scene_plan'].get('entities', []))} entities; "
                      f"relations: {rels}\n", flush=True)

            # 2) Loop: GENERATION AGENT -> EVAL AGENT -> (on fail) feed back to PROMPT AGENT.
            for i in range(1, max_iters + 1):
                print(f"[agent] === iteration {i}/{max_iters} ===", flush=True)

                gen = await gen_agent.generate(state, gen_params)
                produced.append(gen["path"])
                ctx.add("tool", json.dumps(gen), kind="tool_result")
                print(f"[agent:gen] rendered {gen['path']}", flush=True)

                verdict = await eval_agent.evaluate(
                    gen["path"], user_request, state.get("scene_plan"))
                ev, vr = verdict["eval"], verdict["verify"]
                ctx.add("tool", json.dumps(ev), kind="tool_result")
                if vr.get("available"):
                    oks = sum(1 for c in vr.get("constraints", []) if c.get("ok"))
                    print(f"[agent:eval] verify: {oks}/{len(vr.get('constraints', []))} "
                          f"constraints ok", flush=True)
                    for c in vr.get("failed", []):
                        print(f"[agent:eval]   FAIL {c['id']}: {c.get('reason', '')}", flush=True)
                print(f"[agent:eval] clip={ev['clip_score']} style={ev['style_sim']} "
                      f"style_ok={verdict['style_ok']} "
                      f"constraints_ok={verdict['constraints_ok']} "
                      f"passed={verdict['passed']}", flush=True)

                record = {"iteration": i, "image": gen["path"], "prompt": state["prompt"],
                          "eval": ev, "verify": vr, "score": verdict["composite"],
                          "passed": verdict["passed"]}
                if best is None or verdict["composite"] > best["score"]:
                    best = record

                if verdict["passed"]:
                    print("[agent] thresholds + constraints met — stopping.\n", flush=True)
                    break
                if i == max_iters:
                    print("[agent] max iterations reached.\n", flush=True)
                    break

                # 3) FEEDBACK — hand the failed eval back to the PromptAgent, which regenerates
                #    a better prompt (or replans scene structure) before the next render.
                state = await prompt_agent.revise(state, ev, vr, user_request)
                if state.get("last_action") == "replan":
                    print("[agent:prompt] replanned scene structure from failed constraints\n",
                          flush=True)
                else:
                    ctx.add("assistant", f"Refined subject -> {state['subject']}")
                    print(f"[agent:prompt] refined prompt:\n  {state['prompt']}\n", flush=True)

    # 4) Keep only the correct version: delete every rejected render (PNG + sidecar JSON).
    if keep_only_best and best is not None:
        removed = cleanup_outputs(produced, best["image"])
        if removed:
            print(f"[agent] cleanup: removed {len(removed)} rejected render(s); "
                  f"kept {best['image']}", flush=True)

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
