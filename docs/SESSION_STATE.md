# Session State - Goodhart Gap N=100 Experiment

## Last Updated
2026-01-09 ~20:30 UTC

## Project Location
Originally: `/home/adam/Projects/qwen2-caldera-max`

## Completed Tasks
1. Fixed explanation_length bug in `scripts/eval_goodhart_gap.py` (line 533)
2. Added stochastic sampling (--num-runs, --temperature, --seed) to eval script
3. Created RunPod launch config `configs/runpod_n100_experiment.yaml`
4. Created experiment runner `scripts/run_n100_experiment.sh`
5. Ran N=100 experiment on RunPod (2x A100 80GB, ~14 hours)
6. Analyzed results and wrote `docs/N100_EXPERIMENT_ANALYSIS.md`

## Key Results

### DBDK Rates (Does-But-Doesn't-Know)
| Condition | DBDK Rate |
|-----------|-----------|
| Baseline | 35.8% |
| 4-bit Uniform | 35.3% |
| 4-bit Selective | 38.7% |

### Key Finding
Selective compression shows modest (+3%) DBDK increase vs N=20 pilot (+15%). Effect is real but smaller than expected. `timezone_convert` is the clearest single-task signal (0/5 -> 3/5 DBDK).

## Data Locations

### S3 (Primary Backup)
```
s3://caldera-artifacts-20260107/results/n100/
  - goodhart_n100_baseline_20260109_052317.json
  - goodhart_n100_4bit_uniform_20260109_052317.json
  - goodhart_n100_4bit_selective_20260109_052317.json
```

### CALDERA Artifacts (S3)
```
s3://caldera-artifacts-20260107/
  - qwen2-72b-4bit-uniform/ (~73GB)
  - qwen2-72b-4bit-selective/ (~108GB)
```

### Local Results
```
results/n100/
  - goodhart_n100_baseline.json
  - goodhart_n100_uniform.json
  - goodhart_n100_selective.json
```

## Important Files
- `scripts/eval_goodhart_gap.py` - Main evaluation script with stochastic sampling
- `scripts/run_n100_experiment.sh` - RunPod experiment orchestrator
- `configs/runpod_n100_experiment.yaml` - Pod config (2x A100, 1TB storage)
- `docs/N100_EXPERIMENT_ANALYSIS.md` - Comprehensive analysis doc
- `results/quadrant_analysis_n20.md` - N=20 per-task breakdown

## Git Status (at session end)
```
Modified:
  - configs/qwen2_72b_caldera_4bit_uniform.yaml
  - docs/PROGRESS.md
  - scripts/runpod_setup.sh
  - src/caldera_pipeline.py
  - src/caldera_runtime.py

Untracked:
  - configs/qwen2_72b_caldera_4bit_selective.yaml
  - docs/GOODHART_GAP_RESULTS.md
  - docs/LAYER_SENSITIVITY_ANALYSIS.md
  - docs/RUNPOD_SETUP.md
  - docs/STATUS.md
  - docs/N100_EXPERIMENT_ANALYSIS.md
  - docs/SESSION_STATE.md
  - results/
  - scripts/eval_goodhart_gap.py
  - scripts/eval_layer_ablation.py
  - scripts/run_n100_experiment.sh
  - scripts/runpod_health_monitor.sh
```

## Environment
- Model: Qwen/Qwen2-72B-Instruct
- Hardware used: 2x NVIDIA A100 80GB (RunPod)
- Python packages: transformers, accelerate, safetensors, torch
- AWS credentials in `.env` for S3 access

## Next Steps (if continuing)
1. Run N=500 for statistical significance
2. Test multiple seed families
3. Test on other models (Llama, Mistral)
4. Expand timezone/scheduling task coverage
5. Write up for publication (with modest claims)

## Credentials/Secrets
All in `.env` file:
- HF_TOKEN for Hugging Face model access
- RUNPOD_API_KEY for GPU cloud
- AWS credentials for S3 artifact sync
