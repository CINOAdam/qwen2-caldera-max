# N=20 Goodhart Gap Experiment Manifest

**Date**: 2025-01-07
**Git Commit**: 9ad9bab7b010ada229fdfe2f29894fe4b563309a

## Model Configuration
- **Base Model**: Qwen/Qwen2-72B-Instruct
- **Hardware**: 2x A100 80GB (RunPod)
- **Precision**: bf16

## Compression Configurations

### Baseline
- No compression
- File: `goodhart_72b_baseline_n20_v2.json`

### 4-bit Uniform
- Config: `configs/qwen2_72b_caldera_4bit_uniform.yaml`
- All layers compressed to 4-bit
- File: `goodhart_4bit_uniform_n20.json`

### 4-bit Selective
- Config: `configs/qwen2_72b_caldera_4bit_selective.yaml`
- Protects layers 0-8 (early) and 72-79 (late)
- File: `goodhart_4bit_selective_n20_v2.json`

## Test Suite
20 multi-step reasoning tasks from `scripts/eval_goodhart_gap.py`:
- 4 multi_step_math
- 3 time
- 4 financial
- 1 recipe
- 3 units
- 3 logic
- 2 scheduling

## Known Issues
- `explanation_length` field in v1 files is truncated at 200 chars (bug fixed post-run)
- Understanding rubric uses keyword matching (50% threshold)

## Results Summary

| Condition | Understands | Executes | Gaps | DBDK Rate |
|-----------|-------------|----------|------|-----------|
| Baseline | 75% (15/20) | 65% (13/20) | 6 | 31% (4/13) |
| 4-bit Uniform | 75% (15/20) | 65% (13/20) | 6 | 31% (4/13) |
| 4-bit Selective | 60% (12/20) | 65% (13/20) | 5 | 46% (6/13) |

## Key Finding
Selective compression preserves execution but degrades understanding scores,
resulting in elevated DBDK (Does-But-Doesn't-Know) rate.
