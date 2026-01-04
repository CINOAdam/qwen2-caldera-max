#!/usr/bin/env python3
"""
Phase 0: Small Model Validation

Validates the budget allocation hypothesis on Qwen2-1.5B before committing
to larger experiments.

GO/NO-GO Gates:
1. Sensitivity varies >3x across layers
2. Manual optimal allocation beats uniform by >2% PPL

Usage:
    python scripts/small_model_validation.py --all
    python scripts/small_model_validation.py --fidelity-only
    python scripts/small_model_validation.py --compress-only
    python scripts/small_model_validation.py --eval-only
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class ValidationResult:
    sensitivity_ratio: float
    uniform_ppl: float
    optimal_ppl: float
    ppl_improvement: float
    uniform_memory_gb: float
    optimal_memory_gb: float
    gate1_passed: bool  # Sensitivity varies >3x
    gate2_passed: bool  # PPL improvement >2%
    timestamp: str


def run_command(cmd: list[str], description: str) -> subprocess.CompletedProcess:
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[ERROR] {description} failed with code {result.returncode}")
    else:
        print(f"\n[SUCCESS] {description} completed")

    return result


def run_layer_fidelity(model_id: str, output_path: Path, device: str = "cuda") -> dict:
    """Compute layer sensitivities using SipIt-style fidelity."""
    cmd = [
        sys.executable, "scripts/layer_fidelity.py",
        "--model-id", model_id,
        "--dataset", "wikitext",
        "--dataset-config", "wikitext-2-raw-v1",
        "--split", "test",
        "--samples", "32",
        "--max-length", "512",
        "--samples-per-seq", "16",
        "--negatives", "1024",
        "--device", device,
        "--dtype", "bf16",
        "--seed", "42",
        "--output", str(output_path),
    ]

    result = run_command(cmd, "Layer fidelity analysis")
    if result.returncode != 0:
        raise RuntimeError("Layer fidelity failed")

    with open(output_path) as f:
        return json.load(f)


def analyze_sensitivity(fidelity_data: dict) -> tuple[float, float, float]:
    """Analyze sensitivity variation across layers.

    Returns:
        (min_fidelity, max_fidelity, ratio)
    """
    results = fidelity_data.get("results", [])

    # Skip embeddings (index 0), only look at transformer layers
    layer_fidelities = [r["fidelity"] for r in results if r["block_index"] is not None]

    if not layer_fidelities:
        return 0.0, 0.0, 1.0

    min_f = min(layer_fidelities)
    max_f = max(layer_fidelities)

    # Ratio of max to min (higher = more variation)
    # Add small epsilon to avoid division by zero
    ratio = max_f / (min_f + 1e-6) if min_f > 0 else float('inf')

    return min_f, max_f, ratio


def run_compression(config_path: Path) -> None:
    """Run CALDERA compression with given config."""
    cmd = [
        sys.executable, "scripts/compress.py",
        "--config", str(config_path),
    ]

    result = run_command(cmd, f"Compression with {config_path.name}")
    if result.returncode != 0:
        raise RuntimeError(f"Compression failed for {config_path}")


def run_evaluation(
    model_id: str,
    caldera_dir: Path,
    device: str = "cuda",
    samples: int = 64,
) -> tuple[float, float]:
    """Evaluate perplexity and return (ppl, memory_gb).

    Returns:
        (perplexity, peak_memory_gb)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.caldera_loader import load_model_with_caldera
    from src.env import load_env_file

    load_env_file(Path(".env"))

    device_obj = torch.device(device)
    dtype = torch.bfloat16

    # Reset memory stats
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load compressed model
    model, report = load_model_with_caldera(
        model_id,
        caldera_dir,
        device=device_obj,
        dtype=dtype,
        cache_dequant=False,
        chunk_size=1024,
    )
    model.eval()

    print(f"Loaded model: skipped {report.skipped_weights} weights, loaded {report.loaded_tensors} tensors")

    # Load evaluation dataset
    try:
        import datasets
    except ImportError:
        raise RuntimeError("datasets library required")

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if isinstance(t, str) and t.strip()][:samples]

    # Compute perplexity
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            input_ids = encoded["input_ids"].to(device_obj)
            attention_mask = encoded["attention_mask"].to(device_obj)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            tokens = int(attention_mask.sum().item())
            if tokens > 0:
                total_loss += float(outputs.loss) * tokens
                total_tokens += tokens

    if total_tokens == 0:
        raise RuntimeError("No tokens evaluated")

    ppl = math.exp(total_loss / total_tokens)

    # Get peak memory
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
        peak_memory = torch.cuda.max_memory_allocated(device_obj) / (1024**3)
    else:
        peak_memory = 0.0

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return ppl, peak_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: Small model validation")
    parser.add_argument("--all", action="store_true", help="Run full validation pipeline")
    parser.add_argument("--fidelity-only", action="store_true", help="Only run fidelity analysis")
    parser.add_argument("--compress-only", action="store_true", help="Only run compression")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--device", default="cuda", help="Device for computation")
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase0_validation.json"),
                        help="Output path for validation results")
    args = parser.parse_args()

    # Defaults
    model_id = "Qwen/Qwen2-1.5B-Instruct"
    uniform_config = Path("configs/qwen2_1.5b_caldera_uniform.yaml")
    optimal_config = Path("configs/qwen2_1.5b_caldera_manual_optimal.yaml")
    uniform_artifacts = Path("artifacts/qwen2-1.5b/caldera-uniform")
    optimal_artifacts = Path("artifacts/qwen2-1.5b/caldera-manual-optimal")
    fidelity_output = Path("artifacts/qwen2-1.5b/layer_fidelity.json")

    # Create output directories
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fidelity_output.parent.mkdir(parents=True, exist_ok=True)

    run_all = args.all or not (args.fidelity_only or args.compress_only or args.eval_only)

    # ========== STEP 1: Layer Fidelity ==========
    if run_all or args.fidelity_only:
        print("\n" + "="*70)
        print("STEP 1: Computing layer sensitivities (SipIt-style fidelity)")
        print("="*70)

        fidelity_data = run_layer_fidelity(model_id, fidelity_output, args.device)
        min_f, max_f, ratio = analyze_sensitivity(fidelity_data)

        print(f"\n[FIDELITY RESULTS]")
        print(f"  Min fidelity: {min_f:.4f}")
        print(f"  Max fidelity: {max_f:.4f}")
        print(f"  Ratio (max/min): {ratio:.2f}x")
        print(f"  Gate 1 (ratio > 3x): {'PASS' if ratio > 3.0 else 'FAIL'}")

        if args.fidelity_only:
            return

    # ========== STEP 2: Compression ==========
    if run_all or args.compress_only:
        print("\n" + "="*70)
        print("STEP 2: Compressing models (uniform and manual optimal)")
        print("="*70)

        # Uniform compression
        run_compression(uniform_config)

        # Manual optimal compression
        run_compression(optimal_config)

        if args.compress_only:
            return

    # ========== STEP 3: Evaluation ==========
    if run_all or args.eval_only:
        print("\n" + "="*70)
        print("STEP 3: Evaluating compressed models")
        print("="*70)

        print("\n[Evaluating uniform compression...]")
        uniform_ppl, uniform_mem = run_evaluation(model_id, uniform_artifacts, args.device)
        print(f"  Uniform PPL: {uniform_ppl:.4f}")
        print(f"  Uniform Memory: {uniform_mem:.2f} GB")

        print("\n[Evaluating manual optimal compression...]")
        optimal_ppl, optimal_mem = run_evaluation(model_id, optimal_artifacts, args.device)
        print(f"  Optimal PPL: {optimal_ppl:.4f}")
        print(f"  Optimal Memory: {optimal_mem:.2f} GB")

        # Compute improvement
        ppl_improvement = (uniform_ppl - optimal_ppl) / uniform_ppl * 100

        # Load fidelity data for final report
        if fidelity_output.exists():
            with open(fidelity_output) as f:
                fidelity_data = json.load(f)
            _, _, ratio = analyze_sensitivity(fidelity_data)
        else:
            ratio = 0.0

        # Build result
        result = ValidationResult(
            sensitivity_ratio=ratio,
            uniform_ppl=uniform_ppl,
            optimal_ppl=optimal_ppl,
            ppl_improvement=ppl_improvement,
            uniform_memory_gb=uniform_mem,
            optimal_memory_gb=optimal_mem,
            gate1_passed=ratio > 3.0,
            gate2_passed=ppl_improvement > 2.0,
            timestamp=datetime.now().isoformat(),
        )

        # Save result
        with open(args.output, "w") as f:
            json.dump(result.__dict__, f, indent=2)

        # Print final report
        print("\n" + "="*70)
        print("PHASE 0 VALIDATION RESULTS")
        print("="*70)
        print(f"\nSensitivity Analysis:")
        print(f"  Ratio (max/min fidelity): {ratio:.2f}x")
        print(f"  Gate 1 (ratio > 3x): {'PASS' if result.gate1_passed else 'FAIL'}")

        print(f"\nPerplexity Comparison:")
        print(f"  Uniform PPL:  {uniform_ppl:.4f} ({uniform_mem:.2f} GB)")
        print(f"  Optimal PPL:  {optimal_ppl:.4f} ({optimal_mem:.2f} GB)")
        print(f"  Improvement:  {ppl_improvement:+.2f}%")
        print(f"  Gate 2 (improvement > 2%): {'PASS' if result.gate2_passed else 'FAIL'}")

        print("\n" + "-"*70)
        if result.gate1_passed and result.gate2_passed:
            print("GO: Both gates passed. Proceed to Phase 1 (error model sweep).")
        elif result.gate1_passed:
            print("CAUTION: Gate 1 passed but Gate 2 failed.")
            print("  Consider: Is the manual allocation actually optimal?")
            print("  Next step: Run error model sweep to find true optimal.")
        elif result.gate2_passed:
            print("CAUTION: Gate 2 passed but Gate 1 failed.")
            print("  Consider: Low sensitivity variation may limit gains.")
        else:
            print("NO-GO: Both gates failed.")
            print("  Consider: Pivot to Goodhart-only paper or different approach.")
        print("-"*70)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
