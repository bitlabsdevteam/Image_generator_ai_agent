# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI agent that turns a short text request into a polished **semi-3D anime** image
(glossy Pixar-meets-anime mobile-game character renders). Image generation runs locally via
Stable Diffusion (`diffusers` on Apple-Silicon MPS); prompt work runs on a local LLM via
Ollama. The toolset is split into a reusable **MCP server** and an **agent harness** that
drives it.

## Commands

All Python must run inside the project venv (created by `setup.sh`). The repo pins **Python
3.12** because the system default (3.14) has no PyTorch wheels.

```bash
bash setup.sh                                              # venv + deps + download DreamShaper-8 + check Ollama
./.venv/bin/python -m agent.agent "your request here"     # run the full generate→evaluate→refine loop
./.venv/bin/python -m mcp_server.server                   # run the MCP server standalone (stdio)
./.venv/bin/python evals/run_evals.py --steps 20          # batch regression evals over evals/eval_dataset.json
ollama pull qwen3.5:4b                                     # pull the default LLM if missing
```

There is no test suite, linter, or build step. `evals/run_evals.py` is the closest thing to
a regression check. To validate a single case, run `agent.agent` with one request, or add a
prompt to `evals/eval_dataset.json` and run the batch.

## Architecture

Two halves communicate over MCP (stdio), not in-process:

- **`mcp_server/`** — a `FastMCP` server (`server.py`) exposing the image toolset:
  `enhance_prompt`, `generate_image`, `evaluate_image`, `vision_judge`, `list_styles`,
  `list_models`, plus `plan_scene` and `verify_image` for spatial control. Each tool is a thin
  wrapper over a module (`llm_prompt`, `sd_pipeline`, `evals`, `styles`, `scene_planner`). This
  server is reusable by any MCP client (e.g. Claude Desktop).
- **`agent/agent.py`** — an MCP *client* that **spawns the server as a subprocess** over
  stdio and drives the loop: `enhance_prompt` + `plan_scene` → (`generate_image` (ControlNet-
  conditioned by the plan) → `evaluate_image` (style) + `verify_image` (constraints) →
  `replan_scene` / `refine_prompt`) up to `loop.max_iters` times, keeping the best image by a
  composite that weights satisfied constraints heavily.

### Spatial control (why prompts alone weren't enough)

CLIP-guided SDXL has no spatial operator — "girl standing **on top of** a burger, arms up"
collapses to "girl + burger" with the strongest noun prior winning, and `clip_score` is blind
to layout so it can't catch the error. The fix is structural, all derived from the prompt at
runtime (no pose presets / keyword tables):
- **`scene_planner.py`** turns any request into a JSON plan: entities, **spatial relations**,
  normalized boxes, and a **parametric pose** (joint angles) per human. `normalize_layout`
  snaps boxes to satisfy the relations geometrically. It disables Qwen "thinking" (`/no_think`,
  low temp) or planning takes minutes.
- **`pose_control.py`** builds the OpenPose skeleton **parametrically via forward kinematics**
  from the plan's joint angles (a neutral rig + angles; nothing pose-specific hardcoded).
- **`sd_pipeline.py`** turns the plan into ControlNet conditioning (multi-ControlNet, config
  `image.controlnet.controls`, default `[openpose]`).
- **`evals.verify_constraints`** has the local VLM (`gemma4:12b`) grade each plan constraint
  (pose/placement) yes/no — the gate CLIP can't be. Failures drive `scene_planner.replan`,
  which fixes **structure** (boxes/poses), since rewriting text can't fix spatial errors.

### Things that aren't obvious from a single file

- **`config.yaml` is the single source of truth**, loaded once via `mcp_server/config.py`
  (`CONFIG`, an `lru_cache`d singleton that also resolves `paths.*` to absolute and creates
  `outputs/`). Both server and agent import it. Change behavior here, not in code.
- **dtype is `float32`, deliberately.** fp16 SD1.5 VAE produces NaNs/black images on MPS.
  The config comment is authoritative; `sd_pipeline.py`'s module docstring still says "fp16"
  but the code reads `dtype` from config and falls back to fp32 on CPU. Don't switch to
  fp16 to "fix" speed.
- **Two separate LLM modules, both hitting Ollama:** `mcp_server/llm_prompt.py` (`enhance`,
  server-side, expands a request into a full prompt) and `agent/llm.py` (`refine_prompt`,
  client-side, rewrites the subject to fix the weakest eval dimension). Both degrade
  gracefully if Ollama is down — generation never hard-fails on the LLM.
- **Style tags are applied downstream, not by the LLM.** Prompts flow as a bare *subject*;
  `styles.apply_style()` appends the preset's positive tags and supplies the negative prompt.
  The LLM system prompts explicitly forbid emitting style keywords (3d/anime), so the loop
  can re-style a refined subject cleanly. `enhance` returns both `subject` and the full
  `prompt`.
- **Two distinct eval gates with different thresholds.** The in-loop `evals.evaluate()`
  gates on `config.yaml` `evals.clip_threshold`/`style_threshold` (0.21 / 0.50). The batch
  harness uses the stricter thresholds in `evals/eval_dataset.json` (0.25 / 0.55). The CLIP
  score in the loop is computed against the **original user request**, not the enhanced
  prompt.
- **`aesthetic` is a proxy**, not a trained predictor — derived from CLIP feature-norm to
  avoid an extra model download. Treat it as a relative signal only. `composite =
  clip + style + aesthetic/20` is used to pick the best image across iterations; `passed`
  (clip & style thresholds) decides when to stop early.
- **`images/`** holds the reference samples used by `style_sim` (mean CLIP cosine to those
  images). They define the target look — replacing them changes the style-similarity metric.
- **`outputs/`** gets a PNG plus a sidecar `.json` of full generation metadata per render.
- The optional `vision_judge` (multimodal Ollama) is **off the loop path** (`evals.enable_vision_judge: false`) because it's slow.

## Models

- Image: `dreamshaper-8` (SD1.5, default, fast) or `dreamshaper-xl` (1024px, slower) — both
  HuggingFace-hosted, loaded by `diffusers.from_pretrained`. Image gen uses `diffusers`, not
  Ollama; Ollama runs the *LLM only* and cannot run Stable Diffusion.
- LLM: `llm.model` (default `qwen3.5:4b`) for enhance/refine; `llm.vision_model` for the
  optional judge.
