#!/usr/bin/env python3
"""Quick downstream evaluation for CALDERA compressed models."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.caldera_loader import load_model_with_caldera
from src.caldera_runtime import apply_caldera


def load_compressed_model(model_id: str, caldera_dir: str, dtype: torch.dtype, device_map: str = "auto"):
    """Load model with CALDERA compression applied."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    caldera_path = Path(caldera_dir)
    if caldera_path.exists():
        layers_applied = apply_caldera(model, caldera_path)
        print(f"Applied CALDERA layers: {layers_applied}")

    model.eval()
    return model


def evaluate_mmlu(model, tokenizer, subjects: list[str], num_shots: int = 5, max_samples: int = 100):
    """Evaluate on MMLU subjects."""
    results = {}

    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

        # Limit samples
        if len(dataset) > max_samples:
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)

        correct = 0
        total = 0

        # Get few-shot examples from validation set
        try:
            val_dataset = load_dataset("cais/mmlu", subject, split="validation", trust_remote_code=True)
            few_shot_examples = list(val_dataset)[:num_shots]
        except:
            few_shot_examples = []

        for item in tqdm(dataset, desc=subject, leave=False):
            # Build prompt
            prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"

            # Add few-shot examples
            for ex in few_shot_examples:
                prompt += format_question(ex) + f"\nAnswer: {['A', 'B', 'C', 'D'][ex['answer']]}\n\n"

            # Add test question
            prompt += format_question(item) + "\nAnswer:"

            # Get model prediction
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Check if correct
            correct_answer = ['A', 'B', 'C', 'D'][item['answer']]
            if response.upper().startswith(correct_answer):
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[subject] = {"correct": correct, "total": total, "accuracy": accuracy}
        print(f"{subject}: {accuracy:.1%} ({correct}/{total})")

    # Overall
    total_correct = sum(r["correct"] for r in results.values())
    total_samples = sum(r["total"] for r in results.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    results["overall"] = {"correct": total_correct, "total": total_samples, "accuracy": overall_acc}

    return results


def format_question(item):
    """Format MMLU question."""
    q = item["question"]
    choices = item["choices"]
    formatted = f"Question: {q}\n"
    for i, choice in enumerate(choices):
        formatted += f"{['A', 'B', 'C', 'D'][i]}. {choice}\n"
    return formatted.strip()


def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation for CALDERA models")
    parser.add_argument("--model-id", required=True, help="Base model ID")
    parser.add_argument("--caldera-dir", default=None, help="Path to CALDERA artifacts")
    parser.add_argument("--subjects", nargs="+", default=["abstract_algebra", "college_physics", "computer_security", "high_school_mathematics"],
                        help="MMLU subjects to evaluate")
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per subject")
    parser.add_argument("--device-map", default="auto", help="Device map (auto for multi-GPU)")
    parser.add_argument("--dtype", default="bf16", help="Model dtype")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if args.caldera_dir:
        print(f"Loading compressed model from {args.caldera_dir}")
        model = load_compressed_model(args.model_id, args.caldera_dir, dtype, args.device_map)
    else:
        print(f"Loading base model {args.model_id}")
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map=args.device_map)
        model.eval()

    # Run evaluation
    print(f"\nEvaluating on {len(args.subjects)} MMLU subjects ({args.shots}-shot)")
    results = evaluate_mmlu(model, tokenizer, args.subjects, args.shots, args.max_samples)

    print(f"\n{'='*50}")
    print(f"Overall: {results['overall']['accuracy']:.1%} ({results['overall']['correct']}/{results['overall']['total']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
