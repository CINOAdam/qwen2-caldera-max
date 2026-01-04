#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.env import load_env_file
from src.mixkvq import MixKVQCache, MixKVQConfig, patch_qwen2_attention_for_mixkvq


def _resolve_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


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


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / max(temperature, 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        sorted_logits[cutoff] = torch.finfo(sorted_logits.dtype).min
        scaled = torch.zeros_like(scaled).scatter(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    cache,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return input_ids
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        next_token = _sample_next_token(outputs.logits[:, -1, :], temperature, top_p)
        generated = torch.cat([input_ids, next_token], dim=-1)
        for _ in range(max_new_tokens - 1):
            outputs = model(input_ids=next_token, use_cache=True, past_key_values=cache)
            next_token = _sample_next_token(outputs.logits[:, -1, :], temperature, top_p)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="MixKVQ KV-cache smoke test on a small model.")
    parser.add_argument("--model-id", default="Qwen/Qwen2-0.5B-Instruct", help="Model ID or local path.")
    parser.add_argument("--prompt", default="Write a short poem about compression.", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate.")
    parser.add_argument("--device", default="cuda", help="Device for generation.")
    parser.add_argument("--dtype", default="fp16", help="Model dtype.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling cutoff.")
    parser.add_argument("--no-mixkvq", action="store_true", help="Disable MixKVQ and use DynamicCache.")
    parser.add_argument("--ratio-bf16", type=float, default=0.1, help="Fraction of key dims in BF16.")
    parser.add_argument("--ratio-int4", type=float, default=0.2, help="Fraction of key dims in INT4.")
    parser.add_argument("--key-bits-low", type=int, default=2, help="Low-precision bits for keys.")
    parser.add_argument("--key-bits-mid", type=int, default=4, help="Mid-precision bits for keys.")
    parser.add_argument("--value-bits", type=int, default=2, help="Value cache bit-width.")
    parser.add_argument("--update-interval", type=int, default=32, help="Steps between salience updates.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Full-precision buffer length before quantizing (defaults to update interval).",
    )
    parser.add_argument(
        "--cache-dequant",
        action="store_true",
        help="Cache dequantized KV chunks to speed long-context runs (uses more memory).",
    )
    parser.add_argument("--no-pack", action="store_true", help="Disable bit packing for cache storage.")
    parser.add_argument("--no-query", action="store_true", help="Disable query-based salience.")
    parser.add_argument("--use-key-scale", action="store_true", help="Include key scale in salience.")
    parser.add_argument("--report-memory", action="store_true", help="Print GPU memory stats.")
    parser.add_argument("--use-mojo-fused", action="store_true", help="Use Mojo fused attention path.")
    parser.add_argument("--use-max-op", action="store_true", help="Use MAX custom op attention path.")
    parser.add_argument(
        "--log-timing",
        action="store_true",
        help="Print per-stage timing and token throughput.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    if args.report_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        report_memory(device, "start")

    timings = {}
    start_total = time.perf_counter() if args.log_timing else None

    start = time.perf_counter() if args.log_timing else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.log_timing:
        timings["tokenizer_s"] = time.perf_counter() - start

    start = time.perf_counter() if args.log_timing else None
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(device)
    if args.log_timing:
        timings["model_load_s"] = time.perf_counter() - start

    if args.report_memory:
        report_memory(device, "after_load")

    start = time.perf_counter() if args.log_timing else None
    if args.no_mixkvq:
        cache = DynamicCache(config=model.config)
        print("Using DynamicCache (no MixKVQ).")
    else:
        patched = patch_qwen2_attention_for_mixkvq(model)
        if patched == 0:
            print("Warning: Qwen2 attention not patched; falling back to key-only salience.")
        mix_config = MixKVQConfig(
            ratio_bf16=args.ratio_bf16,
            ratio_int4=args.ratio_int4,
            key_bits_low=args.key_bits_low,
            key_bits_mid=args.key_bits_mid,
            value_bits=args.value_bits,
            update_interval=args.update_interval,
            buffer_size=args.buffer_size,
            cache_dequant=args.cache_dequant,
            pack_bits=not args.no_pack,
            use_query=not args.no_query,
            use_key_scale=args.use_key_scale,
            use_mojo_fused=args.use_mojo_fused,
            use_max_op=args.use_max_op,
        )
        cache = MixKVQCache(model.config, mix_config)
        print(f"MixKVQ enabled: bf16={args.ratio_bf16:.2f}, int4={args.ratio_int4:.2f}")
    if args.log_timing:
        timings["cache_setup_s"] = time.perf_counter() - start

    start = time.perf_counter() if args.log_timing else None
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    if args.log_timing:
        timings["encode_s"] = time.perf_counter() - start
        gen_start = time.perf_counter()
    output_ids = generate(
        model,
        input_ids,
        cache=cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.log_timing:
        gen_s = time.perf_counter() - gen_start
        timings["generate_s"] = gen_s
        timings["tokens_per_s"] = float(args.max_new_tokens) / gen_s if gen_s > 0 else 0.0

    if args.report_memory:
        report_memory(device, "after_generate")

    start = time.perf_counter() if args.log_timing else None
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)
    if args.log_timing:
        timings["decode_s"] = time.perf_counter() - start
        timings["total_s"] = time.perf_counter() - start_total if start_total is not None else 0.0
        print("[timing]", " ".join(f"{k}={v:.4f}" for k, v in timings.items()))


if __name__ == "__main__":
    main()
