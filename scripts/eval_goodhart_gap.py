#!/usr/bin/env python3
"""Goodhart Gap evaluation for CALDERA compressed models.

Tests whether compression induces or amplifies the Goodhart Gap -
the phenomenon where models can explain procedures correctly but
fail to execute them.

Usage:
    # Baseline (no compression)
    python scripts/eval_goodhart_gap.py --model-id Qwen/Qwen2-72B-Instruct --device-map auto

    # With CALDERA compression
    python scripts/eval_goodhart_gap.py --model-id Qwen/Qwen2-72B-Instruct \
        --caldera-dir artifacts/qwen2-72b/caldera-3bit-uniform --device-map auto
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.caldera_runtime import apply_caldera
from src.env import load_env_file


# Core Goodhart Gap tests - multi-step reasoning that exposes the gap
GOODHART_TESTS = [
    # The "killer test" from cross-scale validation
    {
        "id": "discount_coupon",
        "category": "multi_step_math",
        "prompt": "If a shirt costs $25 and is on 20% sale, and you have a $5 coupon, what do you pay? Answer with just the number.",
        "correct_answer": "15",
        "understanding_prompt": "To solve this problem: first apply the 20% discount to $25, then subtract the $5 coupon. What are the two steps?",
        "understanding_keywords": ["discount", "20%", "coupon", "subtract"],
        "explanation": "25 × 0.8 = 20, then 20 - 5 = 15",
    },
    # Time arithmetic
    {
        "id": "time_duration",
        "category": "time",
        "prompt": "A meeting starts at 2:45 PM and lasts 1 hour 30 minutes. What time does it end? Answer in H:MM PM format.",
        "correct_answer": "4:15 PM",
        "understanding_prompt": "How do you add 1 hour 30 minutes to 2:45 PM?",
        "understanding_keywords": ["add", "hour", "minute", "3:45", "4:15"],
        "explanation": "2:45 + 1:30 = 4:15 PM",
    },
    # Compound interest + tax (hardest from multi-domain)
    {
        "id": "financial_compound",
        "category": "financial",
        "prompt": "You invest $1000 at 10% annual interest for 2 years (compounded yearly). Then pay 20% tax on the gains only. What's your final amount? Answer with just the number.",
        "correct_answer": "1168",
        "understanding_prompt": "For compound interest followed by tax on gains: what are the steps?",
        "understanding_keywords": ["compound", "interest", "gains", "tax", "principal"],
        "explanation": "1000 × 1.1² = 1210, gains = 210, tax = 42, final = 1210 - 42 = 1168",
    },
    # Recipe scaling
    {
        "id": "recipe_scale",
        "category": "recipe",
        "prompt": "A recipe for 4 servings needs 2 cups flour. You want 6 servings, then double it for a party. How many cups of flour? Answer with just the number.",
        "correct_answer": "6",
        "understanding_prompt": "To scale a recipe from 4 to 6 servings then double: what are the steps?",
        "understanding_keywords": ["scale", "multiply", "double", "ratio"],
        "explanation": "2 cups × (6/4) = 3 cups, then × 2 = 6 cups",
    },
    # Unit conversion chain
    {
        "id": "unit_chain",
        "category": "units",
        "prompt": "Convert 5 miles to km (1 mile = 1.6 km), add 2 km, then convert back to miles. Answer with just the number, rounded to 1 decimal.",
        "correct_answer": "6.3",
        "understanding_prompt": "To convert miles to km, add km, then convert back: what are the steps?",
        "understanding_keywords": ["convert", "multiply", "1.6", "divide", "add"],
        "explanation": "5 × 1.6 = 8 km, + 2 = 10 km, ÷ 1.6 = 6.25 ≈ 6.3 miles",
    },
    # Logic/ordering
    {
        "id": "transitive_order",
        "category": "logic",
        "prompt": "Alice is taller than Bob. Bob is taller than Carol. David is shorter than Carol. List them from tallest to shortest, separated by commas.",
        "correct_answer": "Alice, Bob, Carol, David",
        "understanding_prompt": "Given A > B > C and D < C, what's the order from tallest to shortest?",
        "understanding_keywords": ["Alice", "Bob", "Carol", "David", "transitive"],
        "explanation": "A > B > C > D",
    },
    # Scheduling with dependencies
    {
        "id": "task_schedule",
        "category": "scheduling",
        "prompt": "Task A takes 2 hours. Task B takes 1 hour and must start after A finishes. Task C takes 3 hours and can run parallel to A and B. If all start at 9 AM when possible, when does everything finish? Answer in H AM/PM format.",
        "correct_answer": "12 PM",
        "understanding_prompt": "With parallel and sequential task dependencies, how do you find the finish time?",
        "understanding_keywords": ["parallel", "sequential", "critical path", "dependency"],
        "explanation": "A(9-11) then B(11-12) = 12 PM. C(9-12) = 12 PM. Both paths finish at 12 PM.",
    },
    # === NEW TESTS (Phase 1 expansion) ===
    # Multi-step math (3 new)
    {
        "id": "percentage_chain",
        "category": "multi_step_math",
        "prompt": "A product costs $80. First add 25% markup, then apply a 10% discount. What's the final price? Answer with just the number.",
        "correct_answer": "90",
        "understanding_prompt": "To add markup then discount: what are the calculation steps?",
        "understanding_keywords": ["markup", "25%", "discount", "10%", "multiply"],
        "explanation": "80 × 1.25 = 100, then 100 × 0.9 = 90",
    },
    {
        "id": "average_weighted",
        "category": "multi_step_math",
        "prompt": "A student scores 80 on test 1 (weight 30%), 90 on test 2 (weight 30%), and 70 on test 3 (weight 40%). What's the weighted average? Answer with just the number.",
        "correct_answer": "79",
        "understanding_prompt": "How do you calculate a weighted average from three scores?",
        "understanding_keywords": ["weight", "multiply", "sum", "percentage"],
        "explanation": "80×0.3 + 90×0.3 + 70×0.4 = 24 + 27 + 28 = 79",
    },
    {
        "id": "reverse_percentage",
        "category": "multi_step_math",
        "prompt": "After a 20% discount, a jacket costs $64. What was the original price? Answer with just the number.",
        "correct_answer": "80",
        "understanding_prompt": "If a discounted price is known, how do you find the original price?",
        "understanding_keywords": ["original", "divide", "0.8", "80%", "reverse"],
        "explanation": "64 ÷ 0.8 = 80",
    },
    # Time/scheduling (3 new)
    {
        "id": "timezone_convert",
        "category": "time",
        "prompt": "A call is at 3:30 PM EST. What time is it in PST (3 hours behind EST)? Answer in H:MM PM format.",
        "correct_answer": "12:30 PM",
        "understanding_prompt": "How do you convert from EST to PST?",
        "understanding_keywords": ["subtract", "3 hours", "behind", "timezone"],
        "explanation": "3:30 PM - 3 hours = 12:30 PM",
    },
    {
        "id": "elapsed_time",
        "category": "time",
        "prompt": "A movie starts at 7:45 PM and ends at 10:20 PM. How long is the movie in minutes? Answer with just the number.",
        "correct_answer": "155",
        "understanding_prompt": "How do you calculate elapsed time between two clock times?",
        "understanding_keywords": ["subtract", "hours", "minutes", "convert"],
        "explanation": "10:20 - 7:45 = 2 hours 35 minutes = 155 minutes",
    },
    {
        "id": "schedule_buffer",
        "category": "scheduling",
        "prompt": "You need to arrive at 2 PM. The trip takes 45 minutes. You want 15 minutes buffer. What time should you leave? Answer in H:MM PM format.",
        "correct_answer": "1:00 PM",
        "understanding_prompt": "How do you calculate departure time with travel time and buffer?",
        "understanding_keywords": ["subtract", "buffer", "travel", "45", "15"],
        "explanation": "2:00 PM - 45 min - 15 min = 1:00 PM",
    },
    # Unit conversion (2 new)
    {
        "id": "temperature_convert",
        "category": "units",
        "prompt": "Convert 68°F to Celsius using C = (F-32) × 5/9, then add 10°C. What's the result? Answer with just the number.",
        "correct_answer": "30",
        "understanding_prompt": "How do you convert Fahrenheit to Celsius and then add?",
        "understanding_keywords": ["subtract", "32", "multiply", "5/9", "add"],
        "explanation": "(68-32) × 5/9 = 36 × 5/9 = 20, then 20 + 10 = 30",
    },
    {
        "id": "currency_exchange",
        "category": "units",
        "prompt": "You have $100 USD. Convert to EUR at 0.85 EUR/USD, spend 30 EUR, convert remaining back to USD. How much USD do you have? Answer with just the number, rounded to nearest whole.",
        "correct_answer": "65",
        "understanding_prompt": "How do you convert currency, spend, and convert back?",
        "understanding_keywords": ["multiply", "0.85", "subtract", "divide", "convert"],
        "explanation": "100 × 0.85 = 85 EUR, 85 - 30 = 55 EUR, 55 ÷ 0.85 = 64.7 ≈ 65 USD",
    },
    # Logic/ordering (2 new)
    {
        "id": "ranking_update",
        "category": "logic",
        "prompt": "In a race: Amy finishes before Ben, Ben before Cara, Cara before Dan. Then Dan beats Amy in a rematch. Who is now objectively fastest based on head-to-head? Answer with one name.",
        "correct_answer": "Dan",
        "understanding_prompt": "How do you determine ranking when new results contradict old ones?",
        "understanding_keywords": ["transitive", "update", "head-to-head", "beat"],
        "explanation": "Dan beat Amy, Amy beat Ben, Ben beat Cara. So Dan > Amy > Ben > Cara. Dan is fastest.",
    },
    {
        "id": "set_intersection",
        "category": "logic",
        "prompt": "Set A = {1,2,3,4,5}. Set B = {3,4,5,6,7}. Set C = {5,6,7,8,9}. What numbers are in all three sets? Answer with just the number(s).",
        "correct_answer": "5",
        "understanding_prompt": "How do you find the intersection of three sets?",
        "understanding_keywords": ["intersection", "common", "all three", "overlap"],
        "explanation": "A∩B = {3,4,5}, (A∩B)∩C = {5}",
    },
    # Financial (3 new)
    {
        "id": "loan_payment",
        "category": "financial",
        "prompt": "You borrow $1000 at 5% simple annual interest for 2 years. What's the total amount you repay? Answer with just the number.",
        "correct_answer": "1100",
        "understanding_prompt": "How do you calculate simple interest repayment?",
        "understanding_keywords": ["principal", "interest", "rate", "time", "simple"],
        "explanation": "Interest = 1000 × 0.05 × 2 = 100. Total = 1000 + 100 = 1100",
    },
    {
        "id": "tip_split",
        "category": "financial",
        "prompt": "A dinner bill is $120. Add 20% tip, then split equally among 4 people. How much does each person pay? Answer with just the number.",
        "correct_answer": "36",
        "understanding_prompt": "How do you calculate tip and split a bill?",
        "understanding_keywords": ["tip", "20%", "total", "divide", "split"],
        "explanation": "120 × 1.2 = 144, then 144 ÷ 4 = 36",
    },
    {
        "id": "profit_margin",
        "category": "financial",
        "prompt": "A product costs $40 to make and sells for $60. You sell 50 units. What's the total profit? Answer with just the number.",
        "correct_answer": "1000",
        "understanding_prompt": "How do you calculate total profit from cost, price, and volume?",
        "understanding_keywords": ["profit", "revenue", "cost", "margin", "multiply"],
        "explanation": "Profit per unit = 60 - 40 = 20. Total = 20 × 50 = 1000",
    },
]


@dataclass
class TestResult:
    id: str
    category: str
    understands: bool
    understanding_response: str
    understanding_length: int  # Raw length before truncation
    executes_correctly: bool
    execution_response: str
    execution_length: int  # Raw length before truncation
    extracted_answer: str
    correct_answer: str
    gap_type: str | None  # "KNOWS_BUT_FAILS", "DOES_BUT_DOESNT_KNOW", or None


def get_model_device(model) -> torch.device:
    """Get the input device for a model (handles sharded models)."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        return torch.device(first_device)
    return next(model.parameters()).device


def generate_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 128,
    temperature: float = 0.0, seed: int | None = None
) -> str:
    """Generate a response from the model."""
    device = get_model_device(model)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Set seed for reproducibility in stochastic runs
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return response


def check_understanding(response: str, keywords: list[str]) -> bool:
    """Check if response demonstrates understanding via keyword matching."""
    response_lower = response.lower()
    matches = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matches >= len(keywords) * 0.5  # At least 50% of keywords


def extract_answer(response: str) -> str:
    """Extract numerical or structured answer from response."""
    response = response.strip()

    # Try to find numbers
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)

    # For time answers
    time_match = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?", response)
    if time_match:
        hour, minute, period = time_match.groups()
        period = period.upper() if period else ""
        return f"{hour}:{minute} {period}".strip()

    # For ordering (comma-separated)
    if "," in response:
        parts = [p.strip() for p in response.split(",")]
        if len(parts) >= 3 and all(any(c.isalpha() for c in p) for p in parts):
            return ", ".join(parts[:4])

    # Return first number
    if numbers:
        return numbers[0].replace(",", "")

    return response[:50]


def validate_answer(extracted: str, correct: str, category: str) -> bool:
    """Validate if extracted answer matches correct answer."""
    extracted = extracted.lower().strip()
    correct = correct.lower().strip()

    # Direct match
    if correct in extracted or extracted in correct:
        return True

    # Numeric comparison with tolerance
    try:
        ext_num = float(re.sub(r"[^\d.-]", "", extracted))
        cor_num = float(re.sub(r"[^\d.-]", "", correct))
        if abs(ext_num - cor_num) < 0.5:  # Allow rounding
            return True
    except (ValueError, AttributeError):
        pass

    # Time comparison
    if category == "time":
        ext_time = re.sub(r"\s+", "", extracted)
        cor_time = re.sub(r"\s+", "", correct)
        if ext_time == cor_time:
            return True

    # Ordering comparison
    if category == "logic" and "," in correct:
        ext_items = [x.strip().lower() for x in extracted.split(",")]
        cor_items = [x.strip().lower() for x in correct.split(",")]
        if ext_items == cor_items:
            return True

    return False


def run_test(model, tokenizer, test: dict, temperature: float = 0.0, seed: int | None = None) -> TestResult:
    """Run a single Goodhart Gap test."""
    # Test understanding
    understanding_resp = generate_response(
        model, tokenizer, test["understanding_prompt"],
        temperature=temperature, seed=seed
    )
    understands = check_understanding(understanding_resp, test["understanding_keywords"])

    # Test execution (use seed+1 to get different randomness for execution vs understanding)
    exec_seed = seed + 1 if seed is not None else None
    execution_resp = generate_response(
        model, tokenizer, test["prompt"],
        temperature=temperature, seed=exec_seed
    )
    extracted = extract_answer(execution_resp)
    executes = validate_answer(extracted, test["correct_answer"], test["category"])

    # Determine gap type
    gap_type = None
    if understands and not executes:
        gap_type = "KNOWS_BUT_FAILS"  # The Goodhart Gap!
    elif not understands and executes:
        gap_type = "DOES_BUT_DOESNT_KNOW"

    return TestResult(
        id=test["id"],
        category=test["category"],
        understands=understands,
        understanding_response=understanding_resp[:200],
        understanding_length=len(understanding_resp),  # Raw length before truncation
        executes_correctly=executes,
        execution_response=execution_resp[:200],
        execution_length=len(execution_resp),  # Raw length before truncation
        extracted_answer=extracted,
        correct_answer=test["correct_answer"],
        gap_type=gap_type,
    )


def run_evaluation(
    model, tokenizer, tests: list[dict],
    num_runs: int = 1, temperature: float = 0.0, base_seed: int = 42
) -> list[TestResult]:
    """Run all Goodhart Gap tests with optional stochastic sampling."""
    results = []
    total_tests = len(tests) * num_runs

    for run_idx in range(num_runs):
        run_label = f"[Run {run_idx+1}/{num_runs}] " if num_runs > 1 else ""
        for test_idx, test in enumerate(tests):
            test_num = run_idx * len(tests) + test_idx + 1
            print(f"  {run_label}({test_num}/{total_tests}) {test['id']}...", end=" ", flush=True)

            # Generate unique seed for this test+run combination
            seed = base_seed + run_idx * 1000 + test_idx if temperature > 0 else None
            result = run_test(model, tokenizer, test, temperature=temperature, seed=seed)

            # Add run metadata to result ID if doing multiple runs
            if num_runs > 1:
                result = TestResult(
                    id=f"{result.id}_run{run_idx}",
                    category=result.category,
                    understands=result.understands,
                    understanding_response=result.understanding_response,
                    understanding_length=result.understanding_length,
                    executes_correctly=result.executes_correctly,
                    execution_response=result.execution_response,
                    execution_length=result.execution_length,
                    extracted_answer=result.extracted_answer,
                    correct_answer=result.correct_answer,
                    gap_type=result.gap_type,
                )

            u_icon = "U" if result.understands else "."
            e_icon = "E" if result.executes_correctly else "."
            gap_icon = f" GAP:{result.gap_type}" if result.gap_type else ""
            print(f"[{u_icon}{e_icon}]{gap_icon}")

            results.append(result)

    return results


def print_summary(results: list[TestResult], model_name: str):
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"GOODHART GAP RESULTS: {model_name}")
    print(f"{'='*60}")

    total = len(results)
    understands = sum(1 for r in results if r.understands)
    executes = sum(1 for r in results if r.executes_correctly)
    gaps = [r for r in results if r.gap_type == "KNOWS_BUT_FAILS"]
    does_but_doesnt_know = [r for r in results if r.gap_type == "DOES_BUT_DOESNT_KNOW"]

    print(f"\nUnderstanding: {understands}/{total} ({100*understands/total:.0f}%)")
    print(f"Execution:     {executes}/{total} ({100*executes/total:.0f}%)")
    print(f"Goodhart Gaps: {len(gaps)}/{total}")

    # Conditional gap rate (gaps / understanding_correct)
    # This prevents "0 gaps" looking good when model can't understand
    gap_rate_cond = len(gaps) / understands if understands > 0 else 0
    print(f"Gap Rate|Understand: {len(gaps)}/{understands} ({100*gap_rate_cond:.0f}%)")

    # Does-but-doesn't-know rate = #(execution correct AND understanding wrong) / #(execution correct)
    # This is the mirror metric: execution preserved but meta-cognitive explanation degraded
    dbdk_rate = len(does_but_doesnt_know) / executes if executes > 0 else 0
    print(f"Does-But-Doesn't-Know: {len(does_but_doesnt_know)}/{executes} ({100*dbdk_rate:.0f}%)")

    # Mean explanation length (chars) - measures verbosity/compression of explanations
    # Use raw length field, not truncated response
    explanation_lengths = [r.understanding_length for r in results]
    mean_explanation_len = sum(explanation_lengths) / len(explanation_lengths) if explanation_lengths else 0
    print(f"Mean Explanation Length: {mean_explanation_len:.0f} chars")

    if gaps:
        print(f"\nGAPS DETECTED:")
        for r in gaps:
            print(f"  - {r.id}: expected '{r.correct_answer}', got '{r.extracted_answer}'")

    # By category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"pass": 0, "fail": 0, "gap": 0}
        if r.executes_correctly:
            categories[r.category]["pass"] += 1
        else:
            categories[r.category]["fail"] += 1
        if r.gap_type == "KNOWS_BUT_FAILS":
            categories[r.category]["gap"] += 1

    print(f"\nBy Category:")
    for cat, stats in sorted(categories.items()):
        total_cat = stats["pass"] + stats["fail"]
        pct = 100 * stats["pass"] / total_cat if total_cat > 0 else 0
        gap_note = f" (gap: {stats['gap']})" if stats["gap"] > 0 else ""
        print(f"  {cat:<20} {stats['pass']}/{total_cat} ({pct:.0f}%){gap_note}")

    return {
        "total": total,
        "understands": understands,
        "executes": executes,
        "gaps": len(gaps),
        "gap_rate_conditional": round(gap_rate_cond, 3),  # gaps / understanding_correct
        "does_but_doesnt_know": len(does_but_doesnt_know),
        "dbdk_rate": round(dbdk_rate, 3),  # #(exec correct AND understand wrong) / #exec correct
        "mean_explanation_length": round(mean_explanation_len, 1),
        "gap_ids": [r.id for r in gaps],
        "dbdk_ids": [r.id for r in does_but_doesnt_know],
        "by_category": categories,
    }


def main():
    parser = argparse.ArgumentParser(description="Goodhart Gap evaluation for compressed models")
    parser.add_argument("--model-id", required=True, help="Base model ID")
    parser.add_argument("--caldera-dir", default=None, help="Path to CALDERA artifacts")
    parser.add_argument("--device-map", default="auto", help="Device map for model loading")
    parser.add_argument("--dtype", default="bf16", help="Model dtype (bf16, fp16)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per test (for stochastic sampling)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    model_name = args.model_id.split("/")[-1]

    # Apply CALDERA if specified
    if args.caldera_dir:
        caldera_path = Path(args.caldera_dir)
        if caldera_path.exists():
            print(f"Applying CALDERA from: {args.caldera_dir}")
            replaced = apply_caldera(model, caldera_path)
            print(f"Replaced {len(replaced)} layers")
            model_name = f"{model_name} + {caldera_path.name}"
        else:
            print(f"WARNING: CALDERA dir not found: {args.caldera_dir}")

    model.eval()

    # Run evaluation
    total_evals = len(GOODHART_TESTS) * args.num_runs
    print(f"\nRunning Goodhart Gap tests (N={total_evals}, temp={args.temperature})...")
    results = run_evaluation(
        model, tokenizer, GOODHART_TESTS,
        num_runs=args.num_runs, temperature=args.temperature, base_seed=args.seed
    )

    # Print summary
    summary = print_summary(results, model_name)

    # Save results
    if args.output:
        output_data = {
            "model_id": args.model_id,
            "caldera_dir": args.caldera_dir,
            "config": {
                "num_runs": args.num_runs,
                "temperature": args.temperature,
                "seed": args.seed,
                "total_evaluations": len(results),
            },
            "summary": summary,
            "results": [
                {
                    "id": r.id,
                    "category": r.category,
                    "understands": r.understands,
                    "executes_correctly": r.executes_correctly,
                    "extracted_answer": r.extracted_answer,
                    "correct_answer": r.correct_answer,
                    "gap_type": r.gap_type,
                    "explanation_length": r.understanding_length,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
