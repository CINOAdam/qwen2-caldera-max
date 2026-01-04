# Research Plan

## Objective

Compress Qwen2-72B with a CALDERA-style low-precision + low-rank decomposition
and deliver a Mojo/MAX fused kernel that improves throughput for Q + LR inference.

Define clear success criteria for:
- Quality: perplexity delta vs baseline (target to be set after baseline run).
- Footprint: fit within <= 22 GB VRAM with minimal offload.
- Throughput: tokens/sec vs fp16 baseline at batch=1, seq=1 and seq=128.

## Core questions

- What bit/rank budgets hit a 3090-friendly footprint while preserving acceptable quality?
- How much throughput is lost by unfused Q + LR, and how much can a fused kernel recover?
- Does CALDERA-style quantized low-rank outperform common PTQ baselines at similar bit budgets?

## Phases

### Phase 1: Baselines and scaffolding

- Reproduce a single PTQ baseline on Qwen2-72B (pick AWQ or GPTQ first, pin version + settings).
- Establish a quality + throughput harness with fixed datasets, seeds, and prompt sets.
- Verify local inference on 3090 with a compressed baseline.

### Phase 2: CALDERA-style compression

- Implement or adapt a CALDERA-style decomposition pipeline.
- Stage 1 sweep: BQ in {2}, BL/BR in {4,16}, rank in {128,256}.
- Stage 2 sweep: refine around the best combos and introduce SipIt-guided overrides early.
- Export compressed weights for local inference.
  - Initial implementation uses calibration-aware low-rank on activations with groupwise quantization.

### Phase 3: Mojo/MAX kernel

- Implement fused Q + LR inference kernel with on-the-fly dequant.
- Compare fused vs unfused throughput at batch=1, seq=1 and seq=128.
- Validate numerical parity vs unfused reference (define max/mean abs error targets).

### Phase 4: Write-up

- Document the kernel design, quant choices, and results.
- Publish artifacts and a reproducible runbook.

## Experiment matrix (initial)

- Model: Qwen2-72B
- BQ: 2
- BL/BR: 4, 16
- Rank: 64, 128, 256
- Calibration size: 1k, 4k, 8k sequences
- Metrics: perplexity (C4/WikiText2), quick zero-shot tasks, throughput (tok/s)

## Risks

- E8 lattice quantization may be too complex for a first kernel pass.
- Calibrated decomposition may require significant compute on 72B.
- Local runtime integration can delay kernel testing.
- Dequant overhead and per-step KV expansion can dominate long-context latency.
- Fused kernel may need a two-stage path (Rx then Qx+Lrx) if full fusion regresses.
