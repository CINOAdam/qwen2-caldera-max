# RunPod Deployment Guide for 72B Compression

## Requirements

- **GPU**: 2x A100 80GB (or 4x A100 40GB)
- **Disk**: 500GB+ (model weights + compressed artifacts)
- **Estimated cost**: ~$60-80 for full pipeline
- **Estimated time**: ~15-20 hours total

## Quick Start

### 1. Launch RunPod Instance

- Template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- GPU: 2x A100 80GB (Community Cloud: ~$3.98/hr)
- Disk: 500GB container + 100GB volume

### 2. Setup Environment

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/qwen2-caldera-max.git
cd qwen2-caldera-max

# Install dependencies
pip install -e .
pip install pyyaml datasets transformers accelerate safetensors

# Login to HuggingFace (for Qwen2-72B access)
huggingface-cli login
```

### 3. Run Pipeline

**Option A: Full pipeline (recommended)**
```bash
# Runs: baseline → fidelity → compress (both) → eval (both)
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase all 2>&1 | tee logs/72b_pipeline.log
```

**Option B: Step by step**
```bash
# 1. Baseline evaluation (~2hr)
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase baseline

# 2. Fidelity measurement (~2hr)
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase fidelity

# 3. Compress uniform (~6hr)
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase compress --config 3bit-uniform

# 4. Compress ultra (~6hr)
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase compress --config 3bit-ultra

# 5. Evaluate both (~2hr each)
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase eval --config 3bit-uniform
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase eval --config 3bit-ultra

# 6. Summary
PYTHONPATH=. python scripts/runpod_72b_pipeline.py --phase summary
```

### 4. Download Results

```bash
# Key artifacts to download
tar -czvf results.tar.gz \
    artifacts/qwen2-72b/layer_fidelity.json \
    artifacts/qwen2-72b/caldera-3bit-uniform/compression_stats.json \
    artifacts/qwen2-72b/caldera-3bit-uniform/mmlu_results.json \
    artifacts/qwen2-72b/caldera-3bit-ultra/compression_stats.json \
    artifacts/qwen2-72b/caldera-3bit-ultra/mmlu_results.json \
    artifacts/qwen2-72b/baseline_mmlu_results.json

# Use runpodctl or scp to download
```

## Cost Breakdown

| Phase | GPU Time | Cost (2xA100) |
|-------|----------|---------------|
| Baseline eval | 2hr | $8 |
| Fidelity | 2hr | $8 |
| Compress uniform | 6hr | $24 |
| Compress ultra | 6hr | $24 |
| Eval (both) | 4hr | $16 |
| **Total** | **20hr** | **~$80** |

## Troubleshooting

### OOM during compression
- Reduce calibration samples: edit config `calibration.samples: 256`
- Use CPU offloading: add `--device-map auto` to compress.py

### Network errors downloading model
- Pre-download model: `huggingface-cli download Qwen/Qwen2-72B-Instruct`
- Use offline mode after download: `HF_HUB_OFFLINE=1`

### Compression interrupted
- Pipeline has checkpoint detection - rerun same command to resume
- Check `artifacts/qwen2-72b/*/layers/` for partially completed compression

## Expected Results

Based on 7B validation:

| Config | Expected PPL Improvement | Expected MMLU |
|--------|-------------------------|---------------|
| 3bit-uniform | baseline | ~X% |
| 3bit-ultra | **+30-50% vs uniform** | TBD |

The key hypothesis: sensitivity-guided allocation should provide even larger gains at 72B scale due to increased error compounding over 80 layers.

## Files Created

- `configs/qwen2_72b_caldera_3bit_uniform.yaml`
- `configs/qwen2_72b_caldera_3bit_ultra.yaml`
- `configs/qwen2_72b_caldera_4bit_uniform.yaml`
- `scripts/runpod_72b_pipeline.py`
