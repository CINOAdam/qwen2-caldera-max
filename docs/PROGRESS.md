# Current Progress (MixKVQ)

This file captures the current state of the MixKVQ work so it can be resumed after a reboot.

## Where we are

- Repo root: `/home/adam/Projects/qwen2-caldera-max`
- Current working directory at last check: `/home/adam`

## What was implemented

- MixKVQ KV-cache prototype with:
  - Mixed-precision keys (BF16 + INT4/INT2) and INT2/INT4 value quantization.
  - Buffered quantization (`buffer_size`) to reduce chunk count.
  - Streaming attention path to avoid materializing full KV every step.
  - Optional dequant cache to speed up long-context runs (`cache_dequant`).
- Qwen2 attention patch to use MixKVQ cache and streaming attention.
- Sampling added to the generation script.
- Logit-compare script for correctness checks.

Key files changed/added:
- `src/mixkvq.py`
- `scripts/mixkvq_generate.py`
- `scripts/mixkvq_compare.py`

## Known results (summary)

- Logit compare (Qwen2-0.5B, default prompt):
  - prefill max_abs ~2.8e-02, mean_abs ~4.3e-03
  - incremental max_abs ~2.3e-02, mean_abs ~3.6e-03
- 512 tokens with sampling (temp=0.7, top_p=0.9):
  - MixKVQ output coherent, less degenerate than greedy.
  - Baseline output comparable.
- 2048 tokens with sampling:
  - MixKVQ: alloc ~2.92GiB, reserved ~3.12GiB
  - Baseline: alloc ~2.98GiB, reserved ~3.14GiB
  - Small memory win, MixKVQ slightly more drift-y.
- 4096 tokens:
  - Too slow with current streaming attention; bottleneck is per-step KV repeat and dequant.

## How to resume (commands)

### Compare logits (sanity check)

```
PYTHONPATH=. python scripts/mixkvq_compare.py \
  --model-id Qwen/Qwen2-0.5B-Instruct \
  --mode both
```

### MixKVQ generation (sampling)

```
PYTHONPATH=. python scripts/mixkvq_generate.py \
  --model-id Qwen/Qwen2-1.5B-Instruct \
  --prompt "Explain low-rank compression in one paragraph." \
  --max-new-tokens 1024 \
  --ratio-bf16 0.5 \
  --ratio-int4 0.25 \
  --key-bits-low 4 \
  --key-bits-mid 4 \
  --value-bits 8 \
  --update-interval 16 \
  --buffer-size 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --report-memory
```

### Baseline (no MixKVQ)

```
PYTHONPATH=. python scripts/mixkvq_generate.py \
  --model-id Qwen/Qwen2-1.5B-Instruct \
  --prompt "Explain low-rank compression in one paragraph." \
  --max-new-tokens 1024 \
  --no-mixkvq \
  --temperature 0.7 \
  --top-p 0.9 \
  --report-memory
```

## Current bottleneck

- Long context (4096+) is too slow because streaming attention repeats KV heads for every
  cached chunk (`_repeat_kv`) and dequantizes many chunks per step.

## Suggested next steps

1) Performance path:
   - Avoid per-step `_repeat_kv` by storing KV already expanded to num_attention_heads, or
     implement a fused attention path that handles grouped KV.
2) Memory measurement:
   - Run a baseline 4096 token test only and compare to the 2048 MixKVQ delta.
3) Quality tuning:
   - Adjust `buffer_size` (e.g., 128/256) or precision to find the best quality/memory tradeoff.

## Updates Log

- 2025-12-30: Added Mattermost helper script (`scripts/mattermost_post.py`) with create/post/fetch
  commands, created channel `qwen2-caldera-max-updates` (id `c8eb4ydejjfgppppyf4fcpnarr`),
  and posted an initial status update. Updated `docs/RESEARCH_PLAN.md` with success criteria,
  staged sweeps, kernel validation targets, and additional risks.
- 2025-12-30: Reworked MixKVQ GQA path to avoid per-step `_repeat_kv` expansion by streaming
  attention over grouped query heads. Added `_streaming_attention_gqa` and routed grouped cases
  through it. Sanity check: `scripts/mixkvq_compare.py --model-id Qwen/Qwen2-0.5B-Instruct --mode both`.
- 2025-12-30: Added Mattermost polling helper (`scripts/mattermost_poll.py`) that tracks new posts
  using a persisted timestamp and appends to `artifacts/mattermost_poll.log`.
- 2025-12-30: Started Mattermost poller in background (pid 85137) writing stdout to
  `artifacts/mattermost_poll.out` and appending messages to `artifacts/mattermost_poll.log`.
- 2025-12-30: Restarted poller with 5s interval + auto-ack enabled (pid 86495).
- 2025-12-30: Updated poller to skip auto-acking posts from the same user id, restarted
  (pid 90716) to apply the change.
- 2025-12-30: MixKVQ 2048-token run (Qwen2-0.5B) after GQA change completed in ~353s
  (real time; logged in `artifacts/mixkvq_logs/mixkvq_2048_0p5b.txt`). Memory after generate:
  alloc=0.95GiB, reserved=1.01GiB.
- 2025-12-30: Added `--log-timing` to `scripts/mixkvq_generate.py` and confirmed baseline
  timing at 256 tokens (Qwen2-0.5B, no MixKVQ): ~84 tok/s, generate_s=3.046. Log in
  `artifacts/mixkvq_logs/baseline_256_0p5b_timing.txt`.
- 2026-01-03: Baseline 2048 tokens (Qwen2-0.5B, no MixKVQ) completed with timing:
  ~89 tok/s, generate_s=22.999, memory after generate alloc=0.96GiB reserved=1.02GiB.
  Log in `artifacts/mixkvq_logs/baseline_2048_0p5b_timing.txt`.
- 2026-01-03: MixKVQ 2048 tokens (Qwen2-0.5B, bf16=0.5, int4=0.25) with timing:
  ~5.94 tok/s, generate_s=344.56, memory after generate alloc=0.95GiB reserved=1.01GiB.
  Log in `artifacts/mixkvq_logs/mixkvq_2048_0p5b_timing.txt`.
- 2026-01-03: Added Mojo CPU prototype for MixKVQ streaming attention with GQA and chunked
  softmax (`kernels/mixkvq_streaming_cpu.mojo`). It validates streaming vs naive output
  (max abs diff ~1.9e-09).
- 2026-01-03: Added Mojo GPU prototype for MixKVQ streaming attention with GQA and two
  chunks (`kernels/mixkvq_streaming_gpu.mojo`). It matches the CPU reference (max abs
  diff ~1.2e-07) on small inputs.
- 2026-01-03: Extended the Mojo GPU prototype to support N chunks via concatenated
  K/V buffers plus chunk offsets/lengths. Verified against CPU naive reference
  (max abs diff ~1.2e-07) on small inputs.
- 2026-01-03: Added Mojo GPU fused-dequant prototype (`kernels/mixkvq_streaming_gpu_fused.mojo`)
  for int4-style quantized K/V with per-chunk K scales and per-token V scales. Verified against
  CPU naive reference (max abs diff ~3.0e-08) on small inputs.
- 2026-01-03: Added Python Mojo shim (`src/mixkvq_mojo.py`) and build script
  (`scripts/build_mojo_kernels.sh`) to call the fused GPU kernel from Python (host-side
  quantization for prototype runs). Added `use_mojo_fused` config and CLI flag. Added
  a shared-lib build target `kernels/mixkvq_streaming_gpu_fused_lib.mojo`.
- 2026-01-03: Added mixed-precision fused prototype (`kernels/mixkvq_streaming_gpu_fused_mixed.mojo`)
  supporting bf16/int4/int2 key dims via a mask. Verified against CPU reference
  (max abs diff ~2.2e-08) on small inputs.
- 2026-01-03: Added basic kernel timing loop to `kernels/mixkvq_streaming_gpu_fused.mojo`
  (prints per-iter time for small inputs).
- 2025-12-30: Extended the poller to optionally auto-ack new messages (`--auto-ack`) while ignoring
  posts prefixed with `[codex]` or `[codex-auto-ack]` to avoid loops.
- 2026-01-03: Fixed Mojo C-ABI export to avoid `raises` by wrapping GPU setup in a try/except,
  added causal masking to GPU kernels (fused/mixed/base), and updated the Mojo test to compare
  transposed outputs. `scripts/test_mojo_fused_mixkvq.py` now reports max_abs ~9.8e-04,
  mean_abs ~8.9e-05 against the Python MixKVQ path.
- 2026-01-03: Added attention-mask support to the Mojo fused kernel (and batch-aware K scales),
  updated the Python shim to pass masks and handle batch>1, and added GPU-side int4 quantization
  kernels plus a new `mixkvq_streaming_fused_host_fp16` export used by `streaming_attention_fused`.
  Verified with batch>1 and masked test snippets; MixKVQ generation still works with `--use-mojo-fused`.
- 2026-01-03: Mojo fused 256-token run (Qwen2-0.5B, int4-only) with timing/memory:
  ~9.52 tok/s, generate_s=26.90, after_generate alloc=0.94GiB reserved=0.99GiB.
- 2026-01-03: Baseline 256-token run (Qwen2-0.5B, no MixKVQ) with timing/memory:
  ~84.81 tok/s, generate_s=3.02, after_generate alloc=0.94GiB reserved=0.99GiB.
- 2026-01-03: Added Mojo int4 segment caching to avoid re-quantizing existing chunks,
  introduced `streaming_attention_fused_from_layer` to reuse cached segments and only
  quantize the FP buffer when present, and added early-exit checks in the GPU kernels
  for chunks beyond the causal horizon.
- 2026-01-03: Added a MAX custom op prototype (`kernels/mixkvq_max_op/`) and
  Python wrapper (`src/mixkvq_max.py`) for device-pointer execution, plus a `use_max_op`
  toggle in MixKVQ and CLI support (`--use-max-op`).
- 2026-01-03: Updated the MAX custom op packaging to a Mojo source directory and
  adjusted the kernel to use Input/OutputTensor loops (avoids GPU enqueue API issues);
  CPU custom-op smoke test passes, while GPU invocation still crashes (likely CPU-only
  MAX wheel or missing GPU runtime support).
- 2026-01-03: Reinstalled `modular` from the official Modular index; GPU devices are
  still detected (`accelerator_count=1`), but GPU custom-op execution crashes even for
  a minimal copy op. CPU custom ops continue to work.
- 2026-01-03: Upgraded to Modular nightly (`MAX 26.1.0.dev2026010305`) per docs; GPU
  custom-op execution still crashes without a Python traceback.
- 2026-01-03: Re-ran the minimal GPU custom-op test with debug env vars
  (`MODULAR_MAX_DEBUG`, `MODULAR_ENABLE_PROFILING`, `MODULAR_ENABLE_GPU_PROFILING`);
  it now reports a segmentation fault inside `max/engine/api.py:_Model_execute`.
- 2026-01-03: Added `scripts/max_gpu_custom_op_repro.py` and captured crash logs
  in `artifacts/max_gpu_custom_op_gpu.log` plus a gdb backtrace in
  `artifacts/max_gpu_custom_op_gdb.txt` (segfault in a MAX worker thread).
- 2026-01-03: Added `scripts/run_max_gpu_custom_op_repro.sh` to run the repro with
  debug env vars; generated a 30GiB core file at `artifacts/max_gpu_custom_op.core`
  and captured a gdb dump in `artifacts/max_gpu_custom_op_gdb_core.txt`.
- 2026-01-03: Added a GPU shared-lib kernel for fused Q + LR (`kernels/q_lr_fused_gpu_lib.mojo`),
  a build script (`scripts/build_q_lr_kernels.sh`), a Python ctypes wrapper
  (`src/q_lr_mojo.py`), and a parity test (`scripts/test_q_lr_mojo.py`) against the
  NumPy reference. The test reports max_abs=1.79e-07, mean_abs=4.30e-08.
- 2026-01-03: Updated `benchmarks/run_kernel_bench.py` to run reference vs Mojo fused
  Q + LR (with warmup/validation options). Quick sanity runs at 64x64x8 report
  reference avg_time_s=1.96e-04 and mojo avg_time_s=4.67e-01.
- 2026-01-03: Added preset support and a timing summary line (with approximate GFLOPs)
  to `benchmarks/run_kernel_bench.py`, including a "large" 4096x4096x128 preset.
- 2026-01-03: Added `--log-csv` to `benchmarks/run_kernel_bench.py` to append benchmark
  results (including validation stats) to a CSV in `artifacts/`.
- 2026-01-03: Ran `benchmarks/run_kernel_bench.py` with `--preset large --backend both --validate`.
  Reference avg_time_s=4.195e-02 (0.85 GFLOPs), Mojo avg_time_s=3.332e-02 (1.07 GFLOPs),
  validation max_abs=3.296e-03 mean_abs=7.332e-04; appended to `artifacts/kernel_bench.csv`.
- 2026-01-03: Added `--tag` support to `benchmarks/run_kernel_bench.py` and a sweep helper
  (`scripts/run_kernel_bench_sweep.py`) to iterate shapes/bits and log to CSV.
- 2026-01-03: Ran `scripts/run_kernel_bench_sweep.py --validate --tag sweep-jan03` for
  shapes 512/1024/2048 and bits {2,4} x {4}. Reference stays ~1.1-1.3 GFLOPs, Mojo
  ranges ~0.12-1.08 GFLOPs; validation max_abs up to ~1.83e-03, mean_abs ~3.8e-04.
- 2026-01-03: Added tiled Q/LR accumulation kernels plus a batched Mojo entrypoint that
  keeps weights on device across iterations (`q_lr_fused_host_tiled`,
  `q_lr_fused_host_batched`). Updated `src/q_lr_mojo.py` and `benchmarks/run_kernel_bench.py`
  with a `mojo_cached` backend, and expanded sweep defaults to include 4096/8192 shapes.
- 2026-01-03: Ran a large cached sweep (`scripts/run_kernel_bench_sweep.py` with 4096/8192
  shapes, bits 2/4, iters=2). Mojo cached avg_time_s=2.01e-02 (1.77 GFLOPs) at 4096 and
  avg_time_s=8.89e-02 (1.60 GFLOPs) at 8192; appended to `artifacts/kernel_bench.csv`.
- 2026-01-03: Investigated MAX device-pointer custom ops: `CustomOpLibrary` on CUDA
  still segfaults with a trivial `copy_first` kernel, while `graph_op` without
  custom kernels works on CUDA. `graph_op` + `ops.custom` (kernel_library) also
  segfaults. Drafted a bug report in `docs/BUG_REPORT_MAX_GPU_CUSTOM_OP.md`.
- 2026-01-03: Filed GitHub issue for MAX GPU custom op segfault:
  https://github.com/modular/modular/issues/5737 (logs attached via secret gist).
- 2026-01-03: Optimized Q/LR GPU kernels to iterate per-quant group (fewer scale
  loads/divides) and verified parity (max_abs=1.79e-07). Quick mojo_cached bench
  at 2048x2048x128 reports avg_time_s=6.14e-03 (~1.54 GFLOPs).
- 2026-01-03: Re-ran large mojo_cached sweep after group-loop optimization:
  4096 avg_time_s=1.46e-02 (~2.44 GFLOPs) and 8192 avg_time_s=6.12e-02 (~2.33 GFLOPs),
  tagged `sweep-large-v2` in `artifacts/kernel_bench.csv`.
- 2026-01-03: Tried tile_size=64 and re-ran the large mojo_cached sweep:
  4096 avg_time_s=2.56e-02 (~1.39 GFLOPs) and 8192 avg_time_s=6.26e-02 (~2.28 GFLOPs),
  tagged `sweep-large-v3` in `artifacts/kernel_bench.csv`.
- 2026-01-03: Reverted tile_size to 128 and re-ran the large mojo_cached sweep:
  4096 avg_time_s=1.58e-02 (~2.26 GFLOPs) and 8192 avg_time_s=6.30e-02 (~2.27 GFLOPs),
  tagged `sweep-large-v4` in `artifacts/kernel_bench.csv`.
- 2026-01-03: Optimized `CalderaLinear` to reuse cached R dequantization and avoid
  double-dequant in the non-chunk path; added a small smoke test. This reduces
  per-step overhead when `cache_dequant=True`.
- 2026-01-03: Added a `caldera_cached` backend to `benchmarks/run_kernel_bench.py`
  to benchmark `CalderaLinear` with optional dequant caching.
- 2026-01-03: Added Caldera cache modes (`none`, `r`, `lr`, `qlr`) and a `--device`
  flag to `benchmarks/run_kernel_bench.py`; `CalderaLinear` now supports selective
  dequant caching for R-only or L+R.
- 2026-01-03: Attempted GPU Caldera cache sweep; torch CUDA init failed with
  `cudaGetDeviceCount` error 304 and `Can't initialize NVML`, so GPU benches are
  blocked until CUDA is visible to torch.
- 2026-01-03: CalderaLinear cache A/B at 2048x2048x128: cache on avg_time_s=1.14e-04
  (~82.8 GFLOPs), cache off avg_time_s=2.34e-03 (~4.04 GFLOPs). Logged to
  `artifacts/kernel_bench.csv` with tags `caldera-cache-on`/`caldera-cache-off`.
- 2026-01-03: Mojo cached benchmark at 2048x2048x128: avg_time_s=3.74e-03 (~2.53 GFLOPs),
  tagged `mojo-cache-2048` in `artifacts/kernel_bench.csv`.
- 2026-01-03: 4096x4096x128 cache A/B: Caldera cache on avg_time_s=2.51e-04
  (~142.1 GFLOPs), cache off avg_time_s=2.29e-02 (~1.56 GFLOPs), Mojo cached
  avg_time_s=1.53e-02 (~2.34 GFLOPs). Logged to `artifacts/kernel_bench.csv`
  with tags `caldera-cache-on-4096`, `caldera-cache-off-4096`, `mojo-cache-4096`.
- 2026-01-04: Started 72B compression on RunPod (2x A100 80GB). Baseline results:
  PPL=8.6955, MMLU=73.3% (abstract_algebra 74%, high_school_math 62%, computer_security 84%).
  3bit-uniform compression in progress. Fidelity queued to run after compression.
- 2026-01-04: CPU Caldera cache-mode sweep at 2048x2048x128: none avg_time_s=4.344e-03
  (~2.17 GFLOPs), r avg_time_s=3.714e-03 (~2.54 GFLOPs), lr avg_time_s=4.592e-03
  (~2.06 GFLOPs), qlr avg_time_s=1.22e-04 (~77.2 GFLOPs). Logged with tags
  `caldera-cache-none-2048-cpu`, `caldera-cache-r-2048-cpu`, `caldera-cache-lr-2048-cpu`,
  `caldera-cache-qlr-2048-cpu`.
- 2026-01-04: CPU Caldera cache-mode sweep at 4096x4096x128: none avg_time_s=2.374e-02
  (~1.50 GFLOPs), r avg_time_s=2.595e-02 (~1.37 GFLOPs), lr avg_time_s=2.287e-02
  (~1.56 GFLOPs), qlr avg_time_s=1.29e-04 (~276 GFLOPs). Logged with tags
  `caldera-cache-none-4096-cpu`, `caldera-cache-r-4096-cpu`, `caldera-cache-lr-4096-cpu`,
  `caldera-cache-qlr-4096-cpu`.
- 2026-01-04: Completed 72B compression + evaluation on RunPod (2x A100 80GB).
  Fixed multi-GPU CALDERA support (added `_get_device_for_module()` for hf_device_map
  lookup at layer level, freed original Linear weights before loading replacements).
  Final results:
  * PPL: baseline=8.70, 3bit-uniform=12.65, 3bit-ultra=12.48 (ultra 1.3% better)
  * MMLU: baseline=73.3%, 3bit-uniform=67.3%, 3bit-ultra=70.0% (ultra +2.7pp)
  * computer_security: uniform=84%, ultra=90% (+6pp from protecting late layers)
  Updated `docs/phase0_validation_report.md` with 72B results section.
