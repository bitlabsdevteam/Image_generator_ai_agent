"""Batch eval harness — a regression check distinct from the in-loop scoring.

Generates one image per prompt in eval_dataset.json, scores each with the CLIP-based
evals, and prints a pass/fail table against the dataset thresholds.

    python evals/run_evals.py [--steps 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server import evals, sd_pipeline
from mcp_server.llm_prompt import enhance

DATASET = Path(__file__).parent / "eval_dataset.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="override inference steps (faster)")
    args = ap.parse_args()

    data = json.loads(DATASET.read_text())
    style = data["style"]
    rows = []

    print(f"Running {len(data['prompts'])} eval prompts (style={style})\n")
    for prompt in data["prompts"]:
        enh = enhance(prompt, style)
        meta = sd_pipeline.generate(
            prompt=enh["subject"], negative_prompt=enh["negative_prompt"],
            style=style, steps=args.steps,
        )
        ev = evals.evaluate(meta["path"], prompt)
        rows.append((prompt, ev, meta["path"]))
        print(f"  clip={ev['clip_score']:.3f} style={ev['style_sim']:.3f} "
              f"aes={ev['aesthetic']:.1f} {'PASS' if ev['passed'] else 'fail'}  | {prompt}")

    passed = sum(1 for _, ev, _ in rows if ev["passed"])
    print(f"\n=== {passed}/{len(rows)} passed thresholds "
          f"(clip>={data['thresholds']['clip_score']}, style>={data['thresholds']['style_sim']}) ===")
    print("Images written to outputs/.")


if __name__ == "__main__":
    main()
