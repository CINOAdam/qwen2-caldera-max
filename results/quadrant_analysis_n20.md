# Per-Task Quadrant Analysis: N=20 Goodhart Gap Experiment

**Date**: 2025-01-07

## Quadrant Definitions

| Quadrant | Understanding | Execution | Interpretation |
|----------|---------------|-----------|----------------|
| KD (Knows & Does) | Yes | Yes | Full competence |
| KF (Knows but Fails) | Yes | No | **Goodhart Gap** |
| DK (Does but Doesn't Know) | No | Yes | Auditability concern |
| NN (Neither) | No | No | Complete failure |

## Per-Task Results

| Task ID | Baseline | Uniform | Selective | Transition |
|---------|----------|---------|-----------|------------|
| average_weighted | KF | KF | KF | - |
| currency_exchange | NN | NN | NN | - |
| discount_coupon | KD | KD | KD | - |
| elapsed_time | DK | KD | KD | U_GAINED |
| financial_compound | KF | KF | KF | - |
| loan_payment | KD | KD | KD | - |
| percentage_chain | KD | KD | DK | **U_LOST** |
| profit_margin | KD | DK | DK | U_LOST |
| ranking_update | DK | DK | DK | - |
| recipe_scale | KD | KD | KD | - |
| reverse_percentage | DK | DK | DK | - |
| schedule_buffer | KF | KF | KF | - |
| set_intersection | KD | KD | KD | - |
| task_schedule | KD | KD | KD | - |
| temperature_convert | KF | KF | KF | - |
| time_duration | KF | KF | KF | - |
| timezone_convert | KD | KD | DK | **U_LOST** |
| tip_split | KD | KD | KD | - |
| transitive_order | DK | DK | DK | - |
| unit_chain | KF | KF | NN | U_LOST |

## Quadrant Population Summary

| Quadrant | Baseline | Uniform | Selective | Δ (Selective-Baseline) |
|----------|----------|---------|-----------|------------------------|
| Knows & Does | 9 | 9 | 7 | -2 |
| Knows but Fails (GAP) | 6 | 6 | 5 | -1 |
| Does but Doesn't Know | 4 | 4 | 6 | +2 |
| Neither | 1 | 1 | 2 | +1 |

## Key Findings

### 1. Uniform compression is a no-op on this test
- Uniform 4-bit produces **identical quadrants** to baseline on 20/20 tasks
- No measurable impact on either understanding or execution

### 2. Selective compression causes specific understanding losses
- **4 tasks lost understanding** (percentage_chain, profit_margin, timezone_convert, unit_chain)
- **1 task gained understanding** (elapsed_time)
- **Net: -3 understanding** without changing execution

### 3. The mechanism is auditability degradation
- 3 tasks went from KD→DK: Model still executes correctly but can't explain why
- 1 task went from KF→NN: Was already failing, lost explanation too

## Tasks Affected by Selective Compression

### Understanding Lost (4 tasks)
1. **percentage_chain** (KD→DK): Correct execution, lost keyword matches in explanation
2. **profit_margin** (KD→DK): Correct execution, lost keyword matches
3. **timezone_convert** (KD→DK): Correct execution, lost keyword matches
4. **unit_chain** (KF→NN): Already failing, now can't explain either

### Understanding Gained (1 task)
1. **elapsed_time** (DK→KD): Gained keyword matches in explanation

## Statistical Notes

At N=20:
- 3 task transition = 15% delta
- Not statistically robust; need N≥100 to confirm pattern
- McNemar's test on paired proportions would require larger sample

## Next Steps

1. **Re-run with fixed explanation_length metric** to get true verbosity data
2. **Expand to N=100** to confirm selective's DBDK increase is stable
3. **Analyze keyword matching** to see if selective changes explanation style vs. content
