# Runbook

This is a living checklist for running remote compression and local validation.

## 1) Remote setup (RunPod)

- Use A100 80GB or H100 if possible.
- Create a fresh environment with CUDA and Python 3.10+.
- Clone this repo and add `.env` if needed for model access.

### Create a pod (REST API)

```
PYTHONPATH=. python scripts/runpod_manage.py create \
  --config configs/runpod.yaml \
  --state artifacts/runpod/pod.json
```

### List pods

```
PYTHONPATH=. python scripts/runpod_manage.py list --output artifacts/runpod/pods.json
```

### Stop or delete a pod

```
PYTHONPATH=. python scripts/runpod_manage.py stop <pod_id>
PYTHONPATH=. python scripts/runpod_manage.py delete <pod_id>
```

### Update pod image/env (triggers reset)

```
PYTHONPATH=. python scripts/runpod_manage.py update <pod_id> \
  --config configs/runpod.yaml
```

## 2) Compression (remote)

Planned command (placeholder):

```
python scripts/compress.py --config configs/qwen2_72b_caldera.yaml
```

Expected outputs (placeholder):

- `artifacts/qwen2-72b/caldera/` with compressed weights
- `artifacts/qwen2-72b/metrics.json`

## 3) Transfer artifacts (remote -> local)

- Use `rsync` or `rclone` to copy `artifacts/` to local machine.

## 4) Local validation (RTX 3090)

Quick perplexity check:

```
PYTHONPATH=. python scripts/eval.py \
  --model-id Qwen/Qwen2-7B-Instruct \
  --caldera-dir artifacts/qwen2-72b/caldera \
  --samples 16 \
  --max-length 512
```

GPU memory report:

```
PYTHONPATH=. python scripts/eval.py \
  --model-id Qwen/Qwen2-72B-Instruct \
  --samples 2 \
  --max-length 128 \
  --report-memory
```

Pack artifacts to 2/4-bit:

```
PYTHONPATH=. python scripts/pack_artifacts.py \
  --input-dir artifacts/qwen2-72b/caldera \
  --output-dir artifacts/qwen2-72b/caldera-packed
```

Skip loading linear weights (requires full CALDERA artifacts with `meta.module_name`):

```
PYTHONPATH=. python scripts/eval.py \
  --model-id Qwen/Qwen2-72B-Instruct \
  --caldera-dir artifacts/qwen2-72b/caldera \
  --skip-linear-weights \
  --chunk-size 512
```

Offload embeddings/lm_head to CPU:

```
PYTHONPATH=. python scripts/eval.py \
  --model-id Qwen/Qwen2-72B-Instruct \
  --caldera-dir artifacts/qwen2-72b/caldera \
  --skip-linear-weights \
  --chunk-size 512 \
  --offload-embeddings \
  --offload-lm-head
```

## 4b) Selective compression (SipIt fidelity)

Rank blocks by approximate inversion fidelity (lower = more sensitive):

```
PYTHONPATH=. python scripts/layer_fidelity.py \
  --model-id Qwen/Qwen2-72B-Instruct \
  --caldera-dir artifacts/qwen2-72b/caldera-tuned-packed \
  --skip-linear-weights \
  --samples 8 \
  --max-length 256 \
  --samples-per-seq 6 \
  --negatives 2048 \
  --offload-embeddings \
  --offload-lm-head \
  --output artifacts/qwen2-72b/layer_fidelity.json
```

Use the lowest-fidelity blocks to decide which layers to keep higher rank/bits.

Example overrides in a compression config:

```
caldera:
  rank: 256
  bq: 4
  bl: 4
  pattern_overrides:
    "model.layers.76.*": {rank: 384, bq: 6, bl: 6}
    "model.layers.77.*": {rank: 384, bq: 6, bl: 6}
    "model.layers.78.*": {rank: 512, bq: 8, bl: 8}
  skip_layers:
    - "model.layers.0.self_attn.q_proj"
```

## 5) Kernel benchmarks

Planned command (placeholder):

```
python benchmarks/run_kernel_bench.py --model artifacts/qwen2-72b/caldera
```

## 6) MixKVQ KV-cache test (small model)

Quick smoke test on a small Qwen2 model:

```
PYTHONPATH=. python scripts/mixkvq_generate.py \
  --model-id Qwen/Qwen2-0.5B-Instruct \
  --prompt "Explain low-rank compression in one paragraph." \
  --max-new-tokens 64 \
  --ratio-bf16 0.1 \
  --ratio-int4 0.2
```
