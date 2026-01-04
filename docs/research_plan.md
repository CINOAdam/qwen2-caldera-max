# Research Plan: Sensitivity-Guided Budget Allocation for Model Compression

## Overview

**Research Goal**: Demonstrate that layer-wise sensitivity-guided bit/rank allocation outperforms uniform compression at equivalent memory budgets.

**Secondary Goal**: Analyze Goodhart effects - when perplexity improvements don't transfer to downstream tasks.

**Constraint**: Local-only execution (3090) until approach is validated. No cloud spend until we have evidence it works.

---

## Validation Philosophy

Each phase has a **GO/NO-GO gate**. We don't proceed to expensive steps without evidence.

| Phase | Gate Criteria | Fallback |
|-------|--------------|----------|
| 1 (Sweep) | Error varies >10x across configs | Pivot to different research direction |
| 2 (Fit) | R² > 0.7 for parametric model | Use non-parametric interpolation |
| 3 (Allocator) | Optimal ≠ uniform allocation | Paper becomes "uniform is optimal" |
| 4 (Small model) | >5% PPL improvement | Re-examine error model |
| 5 (Full model) | Results hold at scale | Paper on smaller model only |

---

## Phase 0: Small Model Validation (Local 3090, ~4 hours)

**Objective**: Validate the entire pipeline on Qwen2-1.5B before committing to 72B sweep.

### Why Start Small
- 1.5B fits entirely in VRAM
- Full compression + eval in ~1 hour
- Catches pipeline bugs before 3-day experiment

### Tasks

1. **Compress Qwen2-1.5B uniform** (all layers same config)
   - Config: b_q=2, rank=64
   - Measure: memory, PPL on WikiText-2

2. **Compute layer sensitivities**
   - Run `layer_fidelity.py` on 1.5B
   - Verify sensitivity varies across layers

3. **Compress with manual "optimal" allocation**
   - High-sensitivity layers: b_q=3, rank=128
   - Low-sensitivity layers: b_q=2, rank=32
   - Same total memory as uniform

4. **Compare PPL**
   - If manual optimal beats uniform → proceed
   - If not → investigate before 72B sweep

### GO/NO-GO Gate
- [ ] Sensitivity varies >3x across layers
- [ ] Manual allocation beats uniform by >2% PPL

### Files to Create
- `configs/qwen2_1.5b_caldera_uniform.yaml`
- `configs/qwen2_1.5b_caldera_manual_optimal.yaml`
- `scripts/small_model_validation.py`

---

## Phase 1: Error Model Data Collection (Local 3090, ~3 days)

**Objective**: Collect empirical data mapping (layer, bits, rank) → reconstruction error

### Tasks

1. **Create sweep script** `scripts/error_model_sweep.py`
   - Load individual layer weights via safetensors (avoid full model load)
   - Compress each layer at all (b_q, rank) configurations
   - Measure: raw MSE, normalized MSE, calibration-weighted MSE
   - Resume support for crash recovery
   - JSONL output for easy analysis

2. **Prepare calibration data**
   - Verify `calibration/qwen2_72b_calib.npz` has activation statistics
   - If missing: run hook-based collection on 1024 samples

3. **Run sweep**
   - 80 layers × 7 projections × 15 configs = 8,400 experiments
   - Estimated: 24-72 hours on 3090
   - Output: `artifacts/error_model_sweep/results.jsonl`

### GO/NO-GO Gate
- [ ] Reconstruction error varies >10x across configs for same layer
- [ ] Different layers show different error profiles
- [ ] Data collection completes without OOM/crashes

### Files to Create/Modify
- `scripts/error_model_sweep.py` (new)
- `src/caldera_pipeline.py` (minor: expose single-layer compression)

---

## Phase 2: Parametric Model Fitting (Local, ~2 hours)

**Objective**: Fit E_i(b, r) = s_i × (α·2^(-β·b) + γ·(r+1)^(-δ))

### Tasks

1. **Create analysis script** `scripts/fit_error_model.py`
   - Load sweep results
   - Fit parametric model via scipy.optimize.curve_fit
   - Validate fit quality (R², residual plots)
   - Output: fitted parameters + validation metrics

2. **Analyze layer clustering**
   - Do attention vs MLP layers have different error profiles?
   - Do early vs late layers differ?
   - Visualize: heatmaps of error by (layer_idx, config)

### GO/NO-GO Gate
- [ ] Parametric model R² > 0.7 (or 0.6 with non-parametric fallback)
- [ ] Fitted parameters are physically reasonable (β, δ > 0)
- [ ] Model generalizes across layer types (attention vs MLP)

### Files to Create
- `scripts/fit_error_model.py` (new)
- `notebooks/error_model_analysis.ipynb` (new, optional)

---

## Phase 3: Budget Allocator Implementation (Local, ~1 day)

**Objective**: Implement DP-based optimal allocation algorithm

### Tasks

1. **Create allocator module** `src/budget_allocator.py`
   - `LayerConfig` dataclass
   - `compute_layer_configs()` - generate configs with costs
   - `solve_allocation_dp()` - knapsack DP solver
   - `generate_yaml_config()` - output allocation as YAML overrides

2. **Create allocation script** `scripts/compute_optimal_allocation.py`
   - Load fitted error model
   - Compute optimal allocation for target budgets (18GB, 20GB, 22GB)
   - Output: YAML configs for each budget

3. **Validate allocator**
   - Compare to uniform baseline
   - Verify memory estimates match actual compression

### GO/NO-GO Gate
- [ ] Optimal allocation differs from uniform (not all layers get same config)
- [ ] Predicted total memory matches target budget ±10%
- [ ] Allocator runs in <1 minute

### Files to Create
- `src/budget_allocator.py` (new)
- `scripts/compute_optimal_allocation.py` (new)
- `configs/qwen2_7b_optimal.yaml` (generated)

---

## Phase 4: Scaled Validation on Qwen2-7B (Local 3090, ~1 day)

**Objective**: Validate approach on medium model before committing to 72B

### Why 7B Before 72B
- 7B fits in 3090 with offloading
- 10x faster iteration than 72B
- If it doesn't work at 7B, won't work at 72B

### Tasks

1. **Run error sweep on Qwen2-7B**
   - Same script as Phase 1, just different model
   - ~1,000 experiments (32 layers × 7 proj × 5 configs)
   - Output: `artifacts/qwen2-7b/error_sweep/`

2. **Fit error model for 7B**
   - Verify parametric form transfers from 1.5B

3. **Compute optimal allocation**
   - Target: 8GB compressed (fits 3090 comfortably)
   - Compare uniform vs optimal

4. **Full compression + eval**
   - Compress both variants
   - Eval: PPL + downstream (MMLU subset, HumanEval)

### GO/NO-GO Gate
- [ ] Optimal allocation beats uniform by >3% PPL
- [ ] Downstream tasks preserved within 2%
- [ ] Error model R² > 0.6

### Decision Point
If 7B results are strong:
- Option A: Publish on 7B (faster, lower risk)
- Option B: Scale to 72B (stronger paper, needs RunPod)

### Files to Create
- `configs/qwen2_7b_caldera_uniform.yaml`
- `configs/qwen2_7b_caldera_optimal.yaml`

---

## Phase 5: Evaluation & Goodhart Analysis (Local 3090, ~3-5 days)

**Objective**: Compare uniform vs optimal on perplexity + downstream tasks, analyze metric divergence

### Tasks

1. **Perplexity evaluation** (Qwen2-7B compressed variants)
   - WikiText-2, C4 test set
   - Both uniform and optimal allocation
   - Script: `scripts/eval.py` (existing)

2. **Downstream benchmarks**
   - MMLU (5-shot, subset for speed)
   - HumanEval (coding)
   - GSM8K (math reasoning)
   - Script: `scripts/eval_downstream.py` (new)

3. **Goodhart analysis** - The secondary contribution
   - Sweep multiple compression levels (6GB, 8GB, 10GB targets)
   - Plot PPL vs downstream accuracy for each
   - Key question: Does optimizing for PPL hurt downstream?
   - Key question: Do optimal vs uniform allocations diverge on this?

### GO/NO-GO Gate
- [ ] Optimal allocation beats uniform on PPL
- [ ] Results are consistent across benchmarks (or divergence is interesting)

### Files to Create/Modify
- `scripts/eval_downstream.py` (new)
- `scripts/analyze_goodhart.py` (new)
- `notebooks/goodhart_analysis.ipynb` (visualization)

---

## Phase 6: Paper Writeup (~1 week)

### Structure

1. **Introduction**: Uniform allocation is suboptimal
2. **Method**:
   - Layer sensitivity measurement
   - Parametric error model
   - DP allocation algorithm
3. **Experiments**:
   - Error model validation
   - Compression comparison (uniform vs optimal)
   - Goodhart analysis
4. **Results**:
   - X% PPL improvement at same budget
   - Downstream task preservation
5. **Conclusion**: Principled allocation matters

### Deliverables
- `docs/paper/main.tex`
- Figures: error heatmaps, allocation visualization, PPL curves, Goodhart plots

---

## Timeline (Validation-First)

| Day | Phase | Gate | Output |
|-----|-------|------|--------|
| 1 | Phase 0: 1.5B validation | Sensitivity varies? | GO/NO-GO decision |
| 2-4 | Phase 1: 7B error sweep | Error varies >10x? | results.jsonl |
| 5 | Phase 2: Fit model | R² > 0.7? | fitted params |
| 5-6 | Phase 3: Allocator | Optimal ≠ uniform? | budget_allocator.py |
| 7-8 | Phase 4: 7B full test | >3% PPL gain? | compressed models |
| 9-12 | Phase 5: Eval | Downstream holds? | metrics + Goodhart analysis |
| 13+ | Phase 6: Paper | - | main.tex |

**Key insight**: We can fail fast. If Phase 0 fails on Day 1, we pivot immediately.

---

## Success Criteria

| Metric | Minimum (publishable) | Target | Stretch |
|--------|----------------------|--------|---------|
| PPL improvement vs uniform | 3% | 5% | 15% |
| Downstream preservation | <5% drop | <2% drop | <1% drop |
| Error model R² | 0.6 | 0.8 | 0.9 |
| Paper scope | 7B results | + 72B | + multi-model |

---

## Risk Mitigation

| Risk | Detection | Mitigation |
|------|-----------|------------|
| Sensitivity doesn't vary | Phase 0 gate | Pivot to Goodhart-only paper |
| Error model doesn't fit | Phase 2 R² check | Non-parametric (kNN/interpolation) |
| Optimal ≈ uniform | Phase 3 comparison | Paper: "uniform is near-optimal" (still publishable) |
| PPL improves but downstream hurts | Phase 5 eval | Leads into Goodhart analysis |

---

## Immediate Next Steps

1. **Day 1 Morning**: Create `scripts/small_model_validation.py`
2. **Day 1 Afternoon**: Run Phase 0 on Qwen2-1.5B
3. **Day 1 Evening**: Evaluate GO/NO-GO
4. **If GO**: Start 7B sweep overnight
5. **If NO-GO**: Regroup, analyze why sensitivity doesn't predict error
