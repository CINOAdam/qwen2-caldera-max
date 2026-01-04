# Qwen2-72B CALDERA + Mojo/MAX Kernels

Research repo for CALDERA-style low-precision + low-rank compression of Qwen2-72B,
plus a Mojo/MAX fused kernel for fast inference of the form:

  y = Q x + L (R x)

The goal is to keep CALDERA's accuracy benefits while closing the throughput gap
caused by low-rank dequant and extra matmuls.

## Goals

- Reproduce CALDERA-style compression on Qwen2-72B with calibrated data.
- Export compressed weights that can run on a single RTX 3090.
- Build a Mojo/MAX kernel that fuses dequant + matmul for Q + LR.
- Benchmark quality and throughput vs common baselines.

## Repository layout

- `docs/` research notes, experiment matrix, and write-up material
- `scripts/` runnable helpers for compression, export, and evaluation
- `configs/` model- and run-specific configuration files
- `src/` Python modules for calibration, decomposition, and eval
- `kernels/` Mojo/MAX kernel sources and build scripts
- `benchmarks/` throughput and quality benchmark harnesses
- `artifacts/` local outputs (ignored by git)

## Quickstart (high-level)

1) Create `.env` in repo root with any needed credentials or tokens.
2) Run compression on RunPod (A100/H100 recommended for Qwen2-72B).
3) Export artifacts to `artifacts/` and transfer to local 3090.
4) Run local inference + kernel benchmarks.

Detailed steps live in `docs/RUNBOOK.md`.

## Status

- Calibration + CALDERA-style compression pipeline working on RunPod.
- Reference runtime (`src/caldera_runtime.py`) + eval script (`scripts/eval.py`) added.
- Selective loader (`src/caldera_loader.py`) for skipping linear weights when artifacts cover all modules.
- Artifact packer (`scripts/pack_artifacts.py`) for 2/4-bit storage.
- SipIt-style fidelity scorer (`scripts/layer_fidelity.py`) for selective compression ranking.
- Per-layer overrides in `configs` via `caldera.layer_overrides` / `caldera.pattern_overrides`.
- MixKVQ KV-cache prototype (`src/mixkvq.py`) with a small-model smoke test (`scripts/mixkvq_generate.py`).
- Mojo/MAX kernel remains prototype.
