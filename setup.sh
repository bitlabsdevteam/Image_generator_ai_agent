#!/usr/bin/env bash
# Setup for the semi-3D-anime image generation agent on Apple Silicon.
# Creates a Python 3.12 venv (system 3.14 has no PyTorch wheels), installs deps,
# pre-downloads the default Stable Diffusion model, and checks the local LLM.
set -euo pipefail
cd "$(dirname "$0")"

PY312="${PY312:-/opt/homebrew/bin/python3.12}"
VENV=".venv"

echo "==> Using Python: $PY312"
"$PY312" --version

if [ ! -d "$VENV" ]; then
  echo "==> Creating venv at $VENV"
  "$PY312" -m venv "$VENV"
fi

echo "==> Installing Python dependencies"
"$VENV/bin/python" -m pip install --upgrade pip -q
"$VENV/bin/python" -m pip install -q -r requirements.txt

echo "==> Verifying torch + MPS"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
PY

echo "==> Pre-downloading default Stable Diffusion model (DreamShaper-8, ~2GB)"
"$VENV/bin/python" - <<'PY'
from mcp_server.sd_pipeline import get_pipeline
get_pipeline()  # downloads + caches the default model
print("model ready")
PY

echo "==> Checking Ollama"
if command -v ollama >/dev/null 2>&1; then
  ollama list || true
  echo "    (configure the model in config.yaml -> llm.model; default: qwen3.5:4b)"
else
  echo "    WARNING: ollama not found. enhance/refine will fall back to a template."
fi

echo ""
echo "Setup complete. Try:"
echo "  ./.venv/bin/python -m agent.agent \"American teenagers having fun at a party\""
