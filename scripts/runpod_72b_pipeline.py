#!/usr/bin/env python3
"""
Full 72B compression pipeline for RunPod.
Runs fidelity measurement, compression, and evaluation.

Usage:
    python scripts/runpod_72b_pipeline.py --phase all
    python scripts/runpod_72b_pipeline.py --phase fidelity
    python scripts/runpod_72b_pipeline.py --phase compress --config 3bit-uniform
    python scripts/runpod_72b_pipeline.py --phase eval --config 3bit-uniform
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODEL_ID = "Qwen/Qwen2-72B-Instruct"
ARTIFACTS_DIR = Path("artifacts/qwen2-72b")

CONFIGS = {
    "3bit-uniform": "configs/qwen2_72b_caldera_3bit_uniform.yaml",
    "3bit-ultra": "configs/qwen2_72b_caldera_3bit_ultra.yaml",
    "4bit-uniform": "configs/qwen2_72b_caldera_4bit_uniform.yaml",
}


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if check and result.returncode != 0:
        log(f"Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def phase_fidelity():
    """Run layer fidelity measurement."""
    log("=== Phase: Fidelity Measurement ===")

    output_path = ARTIFACTS_DIR / "layer_fidelity.json"
    if output_path.exists():
        log(f"Fidelity already computed: {output_path}")
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    run_cmd([
        "python", "scripts/layer_fidelity.py",
        "--model-id", MODEL_ID,
        "--samples", "64",
        "--output", str(output_path),
    ])

    log(f"Fidelity saved to {output_path}")


def phase_compress(config_name: str):
    """Run compression with specified config."""
    log(f"=== Phase: Compression ({config_name}) ===")

    if config_name not in CONFIGS:
        log(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
        sys.exit(1)

    config_path = CONFIGS[config_name]

    # Check if already compressed
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    layers_dir = output_dir / "layers"

    if layers_dir.exists() and len(list(layers_dir.glob("*.pt"))) > 100:
        log(f"Compression already done: {layers_dir}")
        return

    run_cmd([
        "python", "scripts/compress.py",
        "--config", config_path,
    ])

    log(f"Compression complete: {output_dir}")


def phase_eval(config_name: str):
    """Evaluate compressed model."""
    log(f"=== Phase: Evaluation ({config_name}) ===")

    if config_name not in CONFIGS:
        log(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
        sys.exit(1)

    import yaml
    with open(CONFIGS[config_name]) as f:
        config = yaml.safe_load(f)

    caldera_dir = config["output_dir"]

    # PPL evaluation
    log("Running perplexity evaluation...")
    run_cmd([
        "python", "scripts/eval.py",
        "--model-id", MODEL_ID,
        "--caldera-dir", caldera_dir,
        "--samples", "256",
        "--device-map", "auto",
    ])

    # MMLU evaluation (subset)
    log("Running MMLU evaluation...")
    mmlu_output = Path(caldera_dir) / "mmlu_results.json"
    run_cmd([
        "python", "scripts/eval_downstream.py",
        "--model-id", MODEL_ID,
        "--caldera-dir", caldera_dir,
        "--subjects", "abstract_algebra", "high_school_mathematics",
        "computer_security", "college_physics",
        "--max-samples", "50",
        "--output", str(mmlu_output),
    ])

    log(f"Evaluation complete for {config_name}")


def phase_baseline():
    """Evaluate baseline (uncompressed) model."""
    log("=== Phase: Baseline Evaluation ===")

    # PPL
    log("Running baseline perplexity...")
    run_cmd([
        "python", "scripts/eval.py",
        "--model-id", MODEL_ID,
        "--samples", "256",
        "--device-map", "auto",
    ])

    # MMLU
    log("Running baseline MMLU...")
    mmlu_output = ARTIFACTS_DIR / "baseline_mmlu_results.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "python", "scripts/eval_downstream.py",
        "--model-id", MODEL_ID,
        "--subjects", "abstract_algebra", "high_school_mathematics",
        "computer_security", "college_physics",
        "--max-samples", "50",
        "--output", str(mmlu_output),
    ])


def phase_all():
    """Run complete pipeline."""
    log("=== Running Full 72B Pipeline ===")

    # 1. Baseline
    phase_baseline()

    # 2. Fidelity
    phase_fidelity()

    # 3. Compress both configs
    for config_name in ["3bit-uniform", "3bit-ultra"]:
        phase_compress(config_name)

    # 4. Evaluate both
    for config_name in ["3bit-uniform", "3bit-ultra"]:
        phase_eval(config_name)

    # 5. Summary
    log("=== Pipeline Complete ===")
    summarize_results()


def summarize_results():
    """Print summary of all results."""
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)

    # Load MMLU results
    for config_name in ["baseline", "3bit-uniform", "3bit-ultra"]:
        if config_name == "baseline":
            path = ARTIFACTS_DIR / "baseline_mmlu_results.json"
        else:
            import yaml
            with open(CONFIGS[config_name]) as f:
                config = yaml.safe_load(f)
            path = Path(config["output_dir"]) / "mmlu_results.json"

        if path.exists():
            with open(path) as f:
                results = json.load(f)
            overall = results.get("overall", {})
            acc = overall.get("accuracy", 0) * 100
            log(f"{config_name}: MMLU {acc:.1f}%")
        else:
            log(f"{config_name}: No results found at {path}")


def main():
    parser = argparse.ArgumentParser(description="72B compression pipeline")
    parser.add_argument("--phase", required=True,
                       choices=["all", "fidelity", "compress", "eval", "baseline", "summary"],
                       help="Pipeline phase to run")
    parser.add_argument("--config", default=None,
                       help="Config name for compress/eval phases")
    args = parser.parse_args()

    # Ensure we're in project root
    if not Path("scripts/compress.py").exists():
        log("ERROR: Must run from project root directory")
        sys.exit(1)

    if args.phase == "all":
        phase_all()
    elif args.phase == "fidelity":
        phase_fidelity()
    elif args.phase == "compress":
        if not args.config:
            log("ERROR: --config required for compress phase")
            sys.exit(1)
        phase_compress(args.config)
    elif args.phase == "eval":
        if not args.config:
            log("ERROR: --config required for eval phase")
            sys.exit(1)
        phase_eval(args.config)
    elif args.phase == "baseline":
        phase_baseline()
    elif args.phase == "summary":
        summarize_results()


if __name__ == "__main__":
    main()
