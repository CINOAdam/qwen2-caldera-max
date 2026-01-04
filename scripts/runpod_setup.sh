#!/bin/bash
# RunPod setup script for 72B compression
# Run this after cloning the repo

set -e

echo "=== Installing dependencies ==="
pip install -e .
pip install pyyaml datasets transformers accelerate safetensors tqdm

echo "=== Creating directories ==="
mkdir -p artifacts/qwen2-72b logs

echo "=== Pre-downloading model (this takes a while) ==="
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2-72B-Instruct')
print('Downloading model weights...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-72B-Instruct', torch_dtype='auto', device_map='auto')
print('Download complete!')
"

echo "=== Setup complete ==="
echo "Run: PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase all"
