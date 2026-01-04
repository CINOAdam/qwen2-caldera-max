#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT_DIR/artifacts/mojo"
mkdir -p "$OUT_DIR"

mojo build "$ROOT_DIR/kernels/q_lr_fused_gpu_lib.mojo" \
  -o "$OUT_DIR/libq_lr_fused_gpu.so" \
  --emit shared-lib

echo "Built $OUT_DIR/libq_lr_fused_gpu.so"
