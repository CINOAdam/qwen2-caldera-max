# Research Summary: CALDERA Compression + SipIt-Guided Selectivity

## Overview
This project explores how to run very large models (Qwen2-72B) on consumer GPUs by combining:
- CALDERA-style low-bit, low-rank decomposition (Q + L(Rx))
- Selective compression informed by SipIt-style inversion fidelity
- Custom runtime and kernel paths (Mojo/MAX) to reduce latency and memory

The short-term goal is a practical 72B-class model that fits a single RTX 3090 without falling off a quality cliff. The longer-term goal is a research-grade compression runtime that can be published, reproduced, and extended by others.

## Why It Matters
Large language models remain inaccessible to most researchers due to memory cost. If we can compress 70B+ models into the 20-22GB range while preserving useful quality, we unlock:
- Local inference and experiments on consumer hardware
- Faster iteration and lower cost for research
- Easier deployment for labs and startups without data center budgets

## Core Ideas

### 1) CALDERA-Style Decomposition
Each linear layer is approximated as:

  W ≈ Q + L R

- Q is a groupwise-quantized weight matrix (low-bit)
- L and R are low-rank factors (also quantized)
- The residual between W and Q is captured by the low-rank term

This preserves more structure than pure quantization and can reduce error per parameter.

### 2) Selective Compression (SipIt-Guided)
Uniform compression across all layers tanks quality. Instead, we use SipIt-style inversion fidelity to rank layer sensitivity:
- Measure how well hidden states reconstruct tokens
- Track per-layer margin/fidelity under compression
- Keep the most sensitive layers higher rank / higher precision

This yields a practical compression budget that targets quality where it matters most.

### 3) Runtime + Kernels
Dequant + low-rank multiplies are expensive if done naively. This repo includes a custom runtime and a Mojo/MAX kernel path to fuse common operations and close the throughput gap.

## Current Status (Local 3090)
- Full CALDERA compression can fit Qwen2-72B in ~20-21GB VRAM with embedding/lm_head offload.
- Quality is very poor at that extreme compression (PPL in the 1e5 range on wikitext).
- A SipIt-guided layer sensitivity run has been completed to inform selective compression.
- Selective compression is now running on the pod with rank upgrades for the most sensitive layers.

## Planned Improvements
- Increase rank and/or bitwidth for the most sensitive layers under a 22GB VRAM cap.
- Expand calibration data with a mix of general corpora and domain-focused slices.
- Evaluate with more stable PPL samples and additional benchmarks.

## Possible Impact
If successful, this work shows that:
- 70B-class models can be made usable on consumer GPUs
- Selective compression guided by intrinsic signals can preserve quality better than uniform compression
- Kernel-level optimization can bring compressed inference closer to baseline speed

This would enable smaller teams to experiment with frontier-scale models without renting large multi-GPU clusters.

## Use Cases
- Local inference and fine-tuning on a single 24GB GPU
- Rapid research iteration on large models without cloud spend
- Deployable edge or on-prem inference where VRAM is limited
- Compression research benchmarking (quality vs. size vs. speed)
- Interpretability research (SipIt-style inversion signals on compressed models)

## Open Questions
- How far can selective compression push quality before VRAM exceeds 22GB?
- What calibration mixture gives the best generalization after compression?
- Can SipIt fidelity be turned into a formal per-layer budget allocator?
- How much speed can fused kernels recover relative to baseline FP16?

## Next Milestones
- Finish selective compression run and evaluate PPL vs VRAM.
- Generate a ranked layer override plan based on SipIt margins.
- Release a write-up + reproducible scripts for the compression pipeline.

