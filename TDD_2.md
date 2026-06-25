# TDD_2 — Models & End-to-End Pipeline

**Project:** LimitBreak — semi-3D-anime image generation agent (MCP + harness)
**Scope of this doc:** every model in the system and the complete technical path from a
text prompt to the final PNG.
**Hardware target:** MacBook M1 Pro, 32 GB unified memory, MPS (Metal) backend.

---

## 1. How many models?

It depends on granularity. The Stable Diffusion "model" is itself a pipeline of several
networks, and CLIP has two towers, so both a top-level and a network-level count are given.

### 1.1 Top-level — 3 mandatory model systems (+1 optional)

| # | Role | Model | Format / runtime | Where in code |
|---|------|-------|------------------|---------------|
| 1 | LLM brain (prompt writer / refiner) | **qwen3.5:4b** | GGUF, Ollama | `mcp_server/llm_prompt.py`, `agent/llm.py` |
| 2 | Image generator | **DreamShaper-XL** (`Lykon/dreamshaper-xl-1-0`, SDXL) | safetensors fp32, diffusers/MPS | `mcp_server/sd_pipeline.py` |
| 3 | Evaluator / scorer | **open_clip ViT-L/14** (`laion2b_s32b_b82k`) | PyTorch fp32, MPS | `mcp_server/evals.py` |
| 4 | *(optional)* vision judge | **gemma4:12b** (multimodal) | GGUF, Ollama | `mcp_server/evals.vision_judge` (off by default) |

### 1.2 Network-level — ~7 distinct neural networks

The SDXL system (#2) is internally **four** networks, and CLIP models have two towers each.

- **qwen3.5:4b** — 1 transformer
- **SDXL internally = 4 networks:**
  1. Text encoder 1 — **OpenAI CLIP ViT-L/14** (text tower, hidden dim 768)
  2. Text encoder 2 — **OpenCLIP ViT-bigG/14** (text tower, hidden 1280 + pooled 1280)
  3. **UNet** — the denoiser (~2.6 B params, the heavy compute)
  4. **VAE** — encoder/decoder (only the decoder is used at inference)
- **Eval CLIP ViT-L/14** — image tower + text tower (1 model, 2 towers)
- *(optional)* **gemma4:12b**

> **Subtle but important:** CLIP ViT-L/14 appears **twice with different weights**.
> SDXL's text-encoder-1 is *OpenAI* CLIP-L and **steers** the pixels; the evaluator is a
> *LAION-trained* open_clip CLIP-L and only **scores** them. Same architecture, different
> checkpoints, different jobs. The diffusion **scheduler** (DPM/Euler) is an *algorithm*,
> not a model.

---

## 2. End-to-end pipeline: prompt → image

```
"American teenagers having fun at a party"
        │
        ▼  agent/agent.py  (MCP client, the harness)
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1  Harness boot                                        │
│  • load system_prompt.txt into ContextManager(max_tokens=3000)│
│  • spawn mcp_server.server as a subprocess over stdio         │
│  • JSON-RPC handshake (initialize, list_tools)               │
└─────────────────────────────────────────────────────────────┘
        │ tool call: enhance_prompt
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2  Prompt engineering   (MODEL 1: qwen3.5:4b)          │
│  llm_prompt.enhance():                                        │
│   • Ollama chat, system="write ONE concise ≤25-word SD prompt,│
│     no style keywords"                                        │
│   • _cap_words() → hard 28-word cap (protect 77-token budget) │
│   • apply_style() → append semi-3d-anime tags + negative      │
│  → {prompt, negative_prompt, subject}                         │
└─────────────────────────────────────────────────────────────┘
        │ loop (max_iters = 2)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3a  Generation   (MODEL 2: SDXL DreamShaper-XL)        │
│  sd_pipeline.generate(style=None):   ── inside diffusers ──   │
│   1 TOKENIZE   prompt → tokens (2 tokenizers, 77 max)        │
│   2 TEXT ENCODE  CLIP-L (768d) ⊕ bigG (1280d) → 2048d/token  │
│                  + pooled bigG embedding; same for negative  │
│   3 NOISE      seeded Gaussian latent  4×96×96  (768/8)       │
│   4 DENOISE ×25  UNet predicts noise per timestep,           │
│                  conditioned on text emb + time + size conds; │
│                  classifier-free guidance (cond vs uncond,    │
│                  scale 6.5); scheduler steps the latent       │
│   5 VAE DECODE  latent → 768×768×3 RGB pixels (fp32)         │
│  → save PNG + sidecar JSON (descriptive slug name)           │
└─────────────────────────────────────────────────────────────┘
        │ tool call: evaluate_image
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3b  Scoring   (MODEL 3: open_clip ViT-L/14)           │
│  evals.evaluate():                                            │
│   • clip_score = cos(img_emb, prompt_emb)                     │
│   • style_sim  = mean cos(img_emb, images/ sample embs)       │
│   • aesthetic  = proxy from image-feature norm                │
│   • passed = clip≥0.18 AND style≥0.40 ; composite = ranking   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼  decision
   passed?  ── yes ──► STOP, return best
        │
        no & iters left
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3c  Refine   (MODEL 1 again: qwen3.5:4b)              │
│  agent.llm.refine_prompt(): rewrite the WEAKEST dimension    │
│   (low clip → make subject/scene explicit;                   │
│    low style → push 3D/glossy/anime tags)                    │
│  → re-enhance → back to STAGE 3a                             │
└─────────────────────────────────────────────────────────────┘
        │ (every tool result appended to ContextManager;
        │  bulky payloads shrunk / old turns summarized if >budget)
        ▼
   FINAL: best-by-composite image path + scores
```

---

## 3. What the data *is* at each hop

| Hop | Representation | Shape / type | Notes |
|-----|----------------|--------------|-------|
| 1 | Raw prompt | `str` | user input |
| 2 | Enriched prompt | `str` | qwen expands intent; style preset adds "look" vocabulary |
| 3 | Tokens | `int[]`, ≤77 | two CLIP tokenizers |
| 4 | Text embeddings | `[77 × 2048]` + pooled `[1280]` | the **only** thing SDXL "understands" — never the raw words |
| 5 | Latent tensor | `[4 × 96 × 96]` floats | starts as pure noise; UNet denoises over 25 steps |
| 6 | Pixels | `[768 × 768 × 3]` | VAE decodes the cleaned latent |
| 7 | Eval embeddings | vectors | eval CLIP re-encodes the image to *measure* it |

`768 / 8 = 96` because the VAE downsamples by a factor of 8; the latent has 4 channels.

---

## 4. The two control loops (the "agent" part)

- **Inner loop = diffusion** (25 UNet steps): pure math, no LLM — denoises one image.
- **Outer loop = agentic refine** (≤2 iterations): generate → **evaluate** → if the CLIP
  gate fails, the **LLM rewrites the prompt** and tries again, always keeping the
  best-`composite` image.

This outer loop is what makes the system an *agent* rather than a one-shot generator: one
model (CLIP) judges another model's output (SDXL), and a third model (qwen) acts on that
judgment to improve the next attempt — orchestrated as **MCP tool calls** over a stdio
JSON-RPC transport, with the transcript kept bounded by context compaction
(`agent/context.py`).

---

## 5. Key architectural separations

- **Generation vs. evaluation are different CLIPs.** SDXL's internal CLIP-L *steers*
  pixels; the eval CLIP-L only *scores* them and never feeds back into generation except
  indirectly, through the LLM's prompt rewrite.
- **LLM vs. diffusion are different runtimes.** qwen/gemma run in **Ollama (GGUF)**;
  SDXL/CLIP run in **PyTorch on MPS (fp32)**. Ollama cannot run Stable Diffusion — that is
  why image generation uses `diffusers`, not GGUF.
- **The MCP server is the reusable core.** All four model capabilities are exposed as tools
  (`enhance_prompt`, `generate_image`, `evaluate_image`, `vision_judge`), so any MCP client
  (the bundled agent, or Claude Desktop) can drive the same models.

---

## 6. Apple-Silicon constraints that shaped the design

These are non-obvious and directly affected model/precision/resolution choices.

- **fp16 → NaN / black images on MPS.** SD1.5's VAE *and* SDXL's UNet overflow in fp16 on
  Metal (an fp16-fix VAE alone is insufficient for SDXL). Everything runs **fp32** on MPS.
  A black ~1–3 KB PNG is the signature of this bug.
- **SDXL fp32 resolution vs. RAM.** At **768 px** the pipeline fits 32 GB (~56 s/image). At
  **1024 px**, SDXL (~14 GB) plus the CLIP eval model exceeds RAM → macOS swaps → ~30
  min/image (~75 s/step). Default is 768; raise to 1024 only with other apps closed.
- **System Python is 3.14 → no PyTorch wheels.** The ML stack runs in a dedicated **3.12
  venv** (`/opt/homebrew/bin/python3.12 -m venv .venv`); `setup.sh` handles this.

---

## 7. Eval score reference (open_clip ViT-L/14)

Measured ranges used to calibrate the loop's gate (`config.yaml`):

| Signal | On-target | Off-target | Threshold |
|--------|-----------|-----------|-----------|
| `clip_score` (image ↔ prompt) | ~0.18 (stylized, on-prompt) | negative, e.g. −0.08 (wrong prompt) | **0.18** |
| `style_sim` (image ↔ samples) | ~0.37 wide scenes … ~0.62 close-ups | low for off-style drift | **0.40** |
| `aesthetic` | saturates ~10 with ViT-L norms | — | not a gate |

`composite = clip_score + style_sim + aesthetic/20` is used only to pick the best image
across iterations; the pass/fail gate is `clip_score ≥ 0.18 AND style_sim ≥ 0.40`.

---

## 8. File → responsibility map

| File | Responsibility |
|------|----------------|
| `agent/agent.py` | MCP client; boots server, runs the refine loop, picks best |
| `agent/llm.py` | Ollama chat + `refine_prompt` (weakest-dimension rewrite) |
| `agent/context.py` | `ContextManager` — token budget, payload shrinking, summarization |
| `agent/system_prompt.txt` | Orchestrator role + semi-3D-anime style rubric |
| `mcp_server/server.py` | FastMCP server exposing the tools (stdio JSON-RPC) |
| `mcp_server/sd_pipeline.py` | diffusers wrapper — DreamShaper-XL on MPS (fp32), slug naming |
| `mcp_server/evals.py` | open_clip ViT-L/14 scoring; optional gemma4 vision judge |
| `mcp_server/llm_prompt.py` | LLM-backed `enhance_prompt` + `_cap_words` token guard |
| `mcp_server/styles.py` | `semi-3d-anime` style preset (positive tags + negatives) |
| `mcp_server/config.py` | Loads `config.yaml` + `.env` (HF_TOKEN) |
| `config.yaml` | Model registry, device/dtype, steps/resolution, thresholds, loop iters |
