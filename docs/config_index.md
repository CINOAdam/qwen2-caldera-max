# Configuration Index

## 3-bit Configs (Recommended)

| Config | PPL | Description | Status |
|--------|-----|-------------|--------|
| `qwen2_1.5b_caldera_3bit_ultra.yaml` | 42.25 | **BEST** - Extreme focus on layers 0, 27, 1, 25 | Working |
| `qwen2_1.5b_caldera_3bit_tiered.yaml` | 42.60 | 3-tier allocation (critical/high/low) | Working |
| `qwen2_1.5b_caldera_3bit_aggressive.yaml` | 42.84 | 10 high-fidelity layers at r=100 | Working |
| `qwen2_1.5b_caldera_3bit_fidelity.yaml` | 43.15 | Moderate allocation (r=76/54) | Working |
| `qwen2_1.5b_caldera_3bit_nocal.yaml` | 43.61 | Uniform baseline (r=64) | Working |
| `qwen2_1.5b_caldera_3bit_extreme.yaml` | 44.07 | Lower budget test | Working |
| `qwen2_1.5b_caldera_3bit_optimal.yaml` | - | Late-layer bias (wrong heuristic) | Deprecated |
| `qwen2_1.5b_caldera_3bit_proportional.yaml` | - | Continuous allocation | Broken (rank bug) |

## 4-bit Configs

| Config | PPL | Description | Status |
|--------|-----|-------------|--------|
| `qwen2_1.5b_caldera_uniform.yaml` | 23.97 | 4-bit uniform (r=64) | Working |
| `qwen2_1.5b_caldera_manual_optimal.yaml` | 23.99 | 4-bit with allocation | Working |

**Note:** At 4-bit, allocation makes negligible difference (<0.1%) because quantization error is already very low.

## 2-bit Configs (Non-Functional)

| Config | Description | Status |
|--------|-------------|--------|
| `qwen2_1.5b_caldera_2bit_uniform.yaml` | 2-bit test | Broken (PPL=millions) |
| `qwen2_1.5b_caldera_2bit_nocal.yaml` | 2-bit without calibration | Broken |

**Root cause:** 2-bit quantization has 60% reconstruction error, which compounds to random noise over 28 layers.

## Key Parameters

```yaml
caldera:
  bq: 3          # Bits for Q (quantized base weight)
  bl: 4          # Bits for L factor
  br: 4          # Bits for R factor
  rank: 64       # Low-rank residual dimension
  group_size: 128
  use_calibration: false  # Calibration has numerical issues
  ridge_lambda: 1.0       # Required for stability
```

## Rank Constraints

Due to a bug in `quantize_groupwise()`, some rank values cause reshape errors. Safe values:
- Multiples of 128: 128, 256, 384, 512
- Tested working: 32, 40, 44, 48, 52, 54, 64, 72, 76, 80, 88, 100, 106, 128, 256

Avoid: 164, 192 (and likely other values near multiples of 128)
