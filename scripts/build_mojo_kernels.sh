#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT_DIR/artifacts/mojo"
mkdir -p "$OUT_DIR"

mojo build "$ROOT_DIR/kernels/mixkvq_streaming_gpu_fused_lib.mojo" \
  -o "$OUT_DIR/libmixkvq_streaming_gpu_fused.so" \
  --emit shared-lib

echo "Built $OUT_DIR/libmixkvq_streaming_gpu_fused.so"
