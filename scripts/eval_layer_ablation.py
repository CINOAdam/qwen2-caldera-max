#!/usr/bin/env python3
"""Per-layer Goodhart Gap ablation study.

Tests which layers are critical for multi-step reasoning by selectively
applying CALDERA compression to different layer groups.

Approach:
1. Load baseline model
2. Apply CALDERA to a SUBSET of layers (excluding the test group)
3. Run Goodhart Gap tests
4. Measure which layer groups are critical for reasoning

Usage:
    python scripts/eval_layer_ablation.py --model-id Qwen/Qwen2-7B-Instruct \
        --caldera-dir artifacts/qwen2-7b/caldera-3bit-uniform \
        --protect-layers 23,24,25,26,27 --device-map auto

    # For 72B:
    python scripts/eval_layer_ablation.py --model-id Qwen/Qwen2-72B-Instruct \
        --caldera-dir artifacts/qwen2-72b/caldera-selective-v2-packed \
        --protect-layers 70,71,72,73,74,75,76,77,78,79 --device-map auto
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.caldera_runtime import apply_caldera, load_caldera_layer, _set_module, _get_device_for_module
from src.env import load_env_file


# Simplified Goodhart tests (subset for faster ablation)
ABLATION_TESTS = [
    {
        "id": "discount_coupon",
        "prompt": "If a shirt costs $25 and is on 20% sale, and you have a $5 coupon, what do you pay? Answer with just the number.",
        "correct": "15",
    },
    {
        "id": "financial_compound",
        "prompt": "You invest $1000 at 10% annual interest for 2 years (compounded yearly). Then pay 20% tax on the gains only. What's your final amount? Answer with just the number.",
        "correct": "1168",
    },
    {
        "id": "unit_chain",
        "prompt": "Convert 5 miles to km (1 mile = 1.6 km), add 2 km, then convert back to miles. Answer with just the number, rounded to 1 decimal.",
        "correct": "6.3",
    },
]


def get_model_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        return torch.device(first_device)
    return next(model.parameters()).device


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    device = get_model_device(model)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    return response


def extract_number(response: str) -> str:
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if numbers:
        return numbers[0].replace(",", "")
    return response[:20]


def check_answer(extracted: str, correct: str) -> bool:
    try:
        ext_num = float(extracted)
        cor_num = float(correct)
        return abs(ext_num - cor_num) < 0.5
    except:
        return correct in extracted.lower()


def apply_caldera_selective(
    model: nn.Module,
    artifacts_dir: Path,
    exclude_layers: set[int],
    *,
    cache_mode: str | None = "r",
    chunk_size: int | None = 1024,
) -> int:
    """Apply CALDERA compression, EXCLUDING specified layers.

    Creates a temp directory with filtered artifacts, then applies CALDERA.
    """
    layers_dir = artifacts_dir / "layers"
    if not layers_dir.exists():
        raise ValueError(f"No layers directory: {layers_dir}")

    # Create temp dir with only the layers we want to compress
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_layers = tmp_path / "layers"
        tmp_layers.mkdir()

        copied = 0
        for f in layers_dir.glob("*.pt"):
            # Extract layer index from filename like model_layers_23_mlp_down_proj.pt
            match = re.match(r"model_layers_(\d+)_", f.name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx in exclude_layers:
                    continue  # Skip protected layers

            # Copy artifact to temp dir
            shutil.copy(f, tmp_layers / f.name)
            copied += 1

        print(f"  Copied {copied} layer artifacts (protecting {len(exclude_layers)} layers)")

        # Apply CALDERA using the filtered artifacts
        replaced = apply_caldera(
            model,
            tmp_path,
            cache_mode=cache_mode,
            chunk_size=chunk_size,
        )

    return len(replaced)


def run_ablation(model, tokenizer, tests: list[dict]) -> dict:
    """Run ablation tests and return results."""
    results = {"passed": 0, "total": len(tests), "details": []}

    for test in tests:
        response = generate_response(model, tokenizer, test["prompt"])
        extracted = extract_number(response)
        correct = check_answer(extracted, test["correct"])

        results["details"].append({
            "id": test["id"],
            "expected": test["correct"],
            "got": extracted,
            "passed": correct,
        })
        if correct:
            results["passed"] += 1

    results["accuracy"] = results["passed"] / results["total"]
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer ablation for Goodhart Gap")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--caldera-dir", required=True)
    parser.add_argument("--protect-layers", default="", help="Comma-separated layers to protect (not compress)")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    load_env_file(Path(".env"))

    # Parse protected layers
    if args.protect_layers:
        protect_set = set(int(x) for x in args.protect_layers.split(","))
    else:
        protect_set = set()

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )

    caldera_path = Path(args.caldera_dir)
    print(f"Applying CALDERA from: {caldera_path}")
    print(f"Protecting layers: {sorted(protect_set) if protect_set else 'NONE (full compression)'}")

    replaced = apply_caldera_selective(model, caldera_path, protect_set)
    print(f"Replaced {replaced} modules")

    model.eval()

    print("\nRunning ablation tests...")
    results = run_ablation(model, tokenizer, ABLATION_TESTS)

    print(f"\n{'='*50}")
    print(f"ABLATION RESULTS")
    print(f"Protected layers: {sorted(protect_set) if protect_set else 'NONE'}")
    print(f"Accuracy: {results['passed']}/{results['total']} ({100*results['accuracy']:.0f}%)")
    print(f"{'='*50}")

    for d in results["details"]:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"  {d['id']}: expected {d['expected']}, got {d['got']} [{status}]")

    if args.output:
        output_data = {
            "model_id": args.model_id,
            "caldera_dir": args.caldera_dir,
            "protected_layers": sorted(protect_set),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
