#!/bin/bash
# Setup script for Vast.ai instances
# Usage: ssh into instance, then:
#   git clone https://github.com/mpeshwe/math-reasoning.git
#   cd math-reasoning && bash scripts/setup_vast.sh

set -e

echo "[setup_vast] Installing dependencies..."
pip install vllm datasets pyyaml requests

echo "[setup_vast] Starting vLLM server in background..."
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 &
VLLM_PID=$!

echo "[setup_vast] Waiting for vLLM server to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
done
echo "[setup_vast] vLLM server is ready (PID: $VLLM_PID)"

echo "[setup_vast] Running baseline evaluation..."
python scripts/run_baseline.py --config configs/single_gpu.yaml

echo "[setup_vast] Done! Results saved to results/"
