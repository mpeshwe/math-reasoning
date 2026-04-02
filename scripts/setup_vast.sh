#!/bin/bash
# Setup script for Vast.ai inference server
# Vast only runs the model server — code and results stay local
#
# Usage on Vast instance:
#   curl -s https://raw.githubusercontent.com/mpeshwe/math-reasoning/main/scripts/setup_vast.sh | bash
#
# Then on local machine:
#   ssh -p SSH_PORT -L 8000:localhost:8000 root@SSH_HOST
#   python scripts/run_baseline.py --config configs/single_gpu.yaml
#
# Caveats:
#   - Use PyTorch image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#   - Host driver must be >= 575 (filter: driver_version>=575.00.00)
#   - Avoid Blackwell GPUs (RTX 50xx) — not supported by PyTorch 2.4
#   - Use Qwen (ungated) not Llama (gated)
#   - Minimum 30GB disk

set -e

echo "[setup_vast] Installing sglang..."
pip install "sglang[all]"

echo "[setup_vast] Starting sglang server..."
echo "[setup_vast] Model will be downloaded on first run (~6GB for Qwen2.5-3B)"
python3 -m sglang.launch_server --model Qwen/Qwen2.5-3B-Instruct --port 8000
