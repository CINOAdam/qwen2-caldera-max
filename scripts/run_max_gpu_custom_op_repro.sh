#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
ART_DIR="$ROOT_DIR/artifacts"
mkdir -p "$ART_DIR"

export MODULAR_MAX_DEBUG=True
export MODULAR_ENABLE_PROFILING=1
export MODULAR_ENABLE_GPU_PROFILING=1
export PYTHONFAULTHANDLER=1

source "$ROOT_DIR/.venv/bin/activate"

python "$ROOT_DIR/scripts/max_gpu_custom_op_repro.py" --device cuda > "$ART_DIR/max_gpu_custom_op_gpu.log" 2>&1
