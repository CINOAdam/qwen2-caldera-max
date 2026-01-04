#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.caldera_loader import load_model_with_caldera
from src.caldera_runtime import apply_caldera
from src.env import load_env_file


def _resolve_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def load_dataset_texts(name: str, config: str, split: str, max_samples: int) -> list[str]:
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`.") from exc

    dataset = datasets.load_dataset(name, config, split=split)
    texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip()]
    return texts[:max_samples]


def eval_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    model.eval()

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            tokens = int(attention_mask.sum().item())
            if tokens == 0:
                continue
            total_loss += float(outputs.loss) * tokens
            total_tokens += tokens

    if total_tokens == 0:
        raise RuntimeError("No tokens were evaluated; check dataset inputs.")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def report_memory(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    to_gib = 1024 ** 3
    print(
        f"[mem] {label}: alloc={allocated / to_gib:.2f}GiB "
        f"reserved={reserved / to_gib:.2f}GiB "
        f"max_alloc={max_allocated / to_gib:.2f}GiB "
        f"max_reserved={max_reserved / to_gib:.2f}GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick perplexity check for CALDERA artifacts.")
    parser.add_argument("--model-id", required=True, help="Base model ID or local path.")
    parser.add_argument("--caldera-dir", type=Path, default=None, help="Path to CALDERA artifacts dir.")
    parser.add_argument("--dataset", default="wikitext", help="HF dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="HF dataset config.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--samples", type=int, default=32, help="Number of samples to eval.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--device", default="cuda", help="Device for evaluation.")
    parser.add_argument("--dtype", default="bf16", help="Model dtype.")
    parser.add_argument("--device-map", default=None, help="Transformers device_map.")
    parser.add_argument("--cache-dequant", action="store_true", help="Cache dequantized weights.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Output chunk size.")
    parser.add_argument(
        "--skip-linear-weights",
        action="store_true",
        help="Skip loading linear layer weights (requires CALDERA artifacts).",
    )
    parser.add_argument(
        "--offload-embeddings",
        action="store_true",
        help="Keep input/output embeddings on CPU and transfer outputs to GPU.",
    )
    parser.add_argument(
        "--offload-lm-head",
        action="store_true",
        help="Keep lm_head on CPU and transfer outputs to GPU.",
    )
    parser.add_argument("--report-memory", action="store_true", help="Print GPU memory stats.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    device_map = args.device_map if args.device_map not in {None, "none"} else None
    if args.report_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        report_memory(device, "start")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.skip_linear_weights:
        if args.caldera_dir is None:
            raise SystemExit("--skip-linear-weights requires --caldera-dir.")
        model, report = load_model_with_caldera(
            args.model_id,
            args.caldera_dir,
            device=device,
            dtype=dtype,
            cache_dequant=args.cache_dequant,
            chunk_size=args.chunk_size,
            offload_input_embeddings=args.offload_embeddings,
            offload_output_embeddings=args.offload_lm_head,
        )
        print(
            "CALDERA loader:",
            f"skipped {report.skipped_weights} weights, loaded {report.loaded_tensors} tensors,",
            f"init_empty={report.used_init_empty}",
        )
        if args.report_memory:
            report_memory(device, "after_load")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            model.to(device)
            if args.report_memory:
                report_memory(device, "after_load")
        elif args.caldera_dir is not None:
            raise SystemExit("CALDERA apply is only supported for single-device runs.")

        if args.caldera_dir is not None:
            replaced = apply_caldera(
                model,
                args.caldera_dir,
                cache_dequant=args.cache_dequant,
                chunk_size=args.chunk_size,
            )
            if not replaced:
                raise SystemExit(f"No CALDERA layers found in {args.caldera_dir}/layers.")
            print(f"Applied CALDERA layers: {len(replaced)}")
            if args.report_memory:
                report_memory(device, "after_apply")

    texts = load_dataset_texts(args.dataset, args.dataset_config, args.split, args.samples)
    ppl = eval_perplexity(model, tokenizer, texts, device, args.max_length)
    print(f"Perplexity ({args.dataset}/{args.dataset_config} {args.split}, {len(texts)} samples): {ppl:.4f}")
    if args.report_memory:
        report_memory(device, "after_eval")


if __name__ == "__main__":
    main()
