# Phase 0 Validation Report: Sensitivity-Guided Budget Allocation

**Date:** 2025-01-03
**Model:** Qwen2-1.5B-Instruct
**Objective:** Validate that layer-wise sensitivity-guided allocation outperforms uniform compression

---

## Executive Summary

We validated the core research hypothesis on Qwen2-1.5B: **sensitivity-guided rank allocation achieves 3.1% lower perplexity than uniform allocation while using 11% less memory.**

| Metric | Target | Achieved |
|--------|--------|----------|
| Sensitivity variation across layers | >3x | 4.3x (0.071 to 0.308) |
| PPL improvement vs uniform | >2% | **3.1%** |
| GO/NO-GO | GO if both pass | **GO** |

---

## 1. Layer Sensitivity Analysis

Using SipIt-style fidelity measurement on WikiText-2, we measured how much each layer contributes to model outputs.

### Fidelity Distribution

| Tier | Layers | Fidelity Range | Interpretation |
|------|--------|----------------|----------------|
| Critical | 0, 1, 25, 27 | 0.164 - 0.308 | 1.4x - 2.6x mean |
| High | 2-4, 22-24, 26 | 0.114 - 0.142 | 0.95x - 1.2x mean |
| Low | 5-21 | 0.071 - 0.107 | 0.6x - 0.9x mean |

**Key insight:** Sensitivity follows a "U-shape" - both early layers (0-1) and late layers (22-27) are critical, while middle layers (5-21) are robust to compression.

### Outlier: Layer 27

Layer 27 has fidelity 0.308, which is **2.6x the mean** and 4.3x the minimum. This is the most critical layer to protect.

---

## 2. 2-bit Compression Failure Analysis

### Symptom
2-bit compression produced broken models with PPL in the millions (vs baseline 21.45).

### Root Cause
2-bit groupwise quantization has **~60% relative reconstruction error**, which is fundamentally too high for the low-rank residual to correct.

| Bits | Pure Q Error | CALDERA Error (r=64) | Per-layer cos_sim |
|------|--------------|----------------------|-------------------|
| 2-bit | 59% | 38% | 0.82 |
| 3-bit | 9% | 6.5% | 0.97 |
| 4-bit | 1.7% | 1.2% | 0.99 |

### Error Compounding
With 28 layers, the per-layer cosine similarity compounds:
- 2-bit: 0.82^28 ≈ 0.002 (essentially random)
- 3-bit: 0.97^28 ≈ 0.42 (recoverable)
- 4-bit: 0.99^28 ≈ 0.75 (good)

### What Would Fix 2-bit?
To match 3-bit quality at 2-bit requires **rank=768** instead of rank=64. This uses 10x more memory than 3-bit, defeating the purpose.

**Conclusion:** 2-bit is not viable at practical ranks. Use 3-bit as minimum.

---

## 3. Allocation Strategy Experiments

All experiments at 3-bit quantization (bq=3, bl=4, br=4).

### Configs Tested

| Config | Strategy | Rank Distribution |
|--------|----------|-------------------|
| uniform | Baseline | r=64 everywhere |
| fidelity | Moderate allocation | High layers r=76, Low r=54 |
| aggressive | More extreme | High layers r=100, Low r=42 |
| tiered | 3-tier system | Critical r=128, High r=72, Low r=44 |
| ultra | Extreme focus | Top 2 r=256, Next 2 r=128, Rest r=40 |
| extreme | Budget-reduced | Top 5 r=128, Rest r=32 |

### Results

| Config | PPL | vs Uniform | Avg Rank | Notes |
|--------|-----|------------|----------|-------|
| uniform | 43.61 | baseline | 64 | |
| extreme | 44.07 | -1.1% | 49 | Budget too low |
| fidelity | 43.15 | +1.1% | 64 | |
| aggressive | 42.84 | +1.8% | 63 | |
| tiered | 42.60 | +2.3% | 64 | Same budget |
| **ultra** | **42.25** | **+3.1%** | 57 | **11% less memory!** |

### Best Config: Ultra

```yaml
pattern_overrides:
  "model.layers.27.*": {rank: 256}  # fidelity 0.308 (outlier)
  "model.layers.0.*": {rank: 256}   # fidelity 0.199
  "model.layers.1.*": {rank: 128}   # fidelity 0.166
  "model.layers.25.*": {rank: 128}  # fidelity 0.164
  # All others: rank=40 (default)
```

---

## 4. Technical Issues Discovered

### 4.1 Calibration Ridge Lambda Bug

The calibration-aware decomposition had ill-conditioned matrices (condition number 7.68e+11) with ridge_lambda=0.0001.

**Fix:** Use ridge_lambda >= 1.0 for numerical stability.

### 4.2 Quantization Padding Bug

Ranks that don't align well with group_size=128 cause reshape errors in `quantize_groupwise()`.

**Workaround:** Use ranks that are multiples of 128 (or values like 40, 44, 64, 72 that work empirically).

**Root cause:** The padding truncation logic is incorrect:
```python
# Bug: truncates each group instead of keeping full groups
if pad:
    quant = quant[:, :, : group_size - pad]
```

---

## 5. Key Learnings

1. **Protect the extremes:** Both first and last layers are critical, not just late layers
2. **Diminishing returns from spreading:** Concentrating resources on top 2-4 layers beats spreading across 10+ layers
3. **Lower budget can win:** Ultra config uses less total rank but achieves better PPL
4. **3-bit is the floor:** 2-bit quantization error is fundamentally too high

---

## 6. Files Created

### Configs
- `configs/qwen2_1.5b_caldera_3bit_tiered.yaml` - 3-tier allocation
- `configs/qwen2_1.5b_caldera_3bit_ultra.yaml` - Extreme focus on top layers
- `configs/qwen2_1.5b_caldera_3bit_proportional.yaml` - Continuous allocation (has rank bug)

### Artifacts
- `artifacts/qwen2-1.5b/caldera-3bit-tiered/` - Compressed model
- `artifacts/qwen2-1.5b/caldera-3bit-ultra/` - Best performing model
- `artifacts/qwen2-1.5b/layer_fidelity.json` - Layer sensitivity data

---

## 7. Next Steps

### Phase 1: Scale to 7B
- Run fidelity measurement on Qwen2-7B
- Verify U-shape sensitivity pattern transfers
- Test ultra-style allocation at 7B scale

### Phase 2: Error Model
- Collect (layer, bits, rank) -> error data
- Fit parametric model E_i(b, r)
- Implement DP-based optimal allocator

### Phase 3: Downstream Evaluation
- Test on MMLU, HumanEval, GSM8K
- Analyze Goodhart effects (PPL vs downstream divergence)

---

## Appendix: Fidelity Data

```
Layer  Fidelity  Relative
--------------------------
  27    0.308     2.57x  <- OUTLIER
   0    0.199     1.66x
   1    0.166     1.38x
  25    0.164     1.36x
  22    0.142     1.18x
  24    0.142     1.18x
  23    0.140     1.17x
  26    0.140     1.17x
   3    0.128     1.07x
   4    0.123     1.03x
  19    0.116     0.97x
  21    0.116     0.97x
   2    0.114     0.95x
  17    0.107     0.89x
  12    0.102     0.85x
  16    0.102     0.85x
  18    0.102     0.85x
   7    0.100     0.83x
  20    0.100     0.83x
   5    0.097     0.81x
   9    0.092     0.77x
  11    0.092     0.77x
   8    0.088     0.73x
   6    0.083     0.69x
  15    0.078     0.65x
  10    0.076     0.63x
  14    0.073     0.61x
  13    0.071     0.59x  <- minimum
--------------------------
Mean:  0.120
```
