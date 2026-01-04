#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _compare_logits(name: str, base: torch.Tensor, mix: torch.Tensor) -> None:
    if base.shape != mix.shape:
        raise RuntimeError(f"{name}: shape mismatch {tuple(base.shape)} vs {tuple(mix.shape)}")
    diff = (mix - base).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = base.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item())
    print(f"{name}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e}")


def run_prefill(model: torch.nn.Module, input_ids: torch.Tensor, cache) -> torch.Tensor:
    outputs = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
    return outputs.logits[:, -1, :]


def run_incremental(model: torch.nn.Module, input_ids: torch.Tensor, cache) -> torch.Tensor:
    logits = None
    for idx in range(input_ids.shape[1]):
        token = input_ids[:, idx : idx + 1]
        outputs = model(input_ids=token, use_cache=True, past_key_values=cache)
        logits = outputs.logits[:, -1, :]
    if logits is None:
        raise RuntimeError("No logits produced; empty input?")
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MixKVQ streaming attention logits vs baseline.")
    parser.add_argument("--model-id", default="Qwen/Qwen2-0.5B-Instruct", help="Model ID or local path.")
    parser.add_argument("--prompt", default="Explain low-rank compression in one paragraph.", help="Prompt text.")
    parser.add_argument("--device", default="cuda", help="Device for comparison.")
    parser.add_argument("--dtype", default="fp16", help="Model dtype.")
    parser.add_argument(
        "--mode",
        choices=("prefill", "incremental", "both"),
        default="both",
        help="Compare full prefill, incremental cache updates, or both.",
    )
    parser.add_argument("--buffer-size", type=int, default=4096, help="MixKVQ full-precision buffer length.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    mix_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    patched = patch_qwen2_attention_for_mixkvq(mix_model)
    if patched == 0:
        raise SystemExit("MixKVQ patch failed; Qwen2 attention not found.")

    mix_config = MixKVQConfig(
        ratio_bf16=1.0,
        ratio_int4=0.0,
        key_bits_low=4,
        key_bits_mid=4,
        value_bits=8,
        update_interval=16,
        buffer_size=args.buffer_size,
        pack_bits=False,
        use_query=True,
        use_key_scale=False,
    )

    base_model.eval()
    mix_model.eval()
    with torch.no_grad():
        if args.mode in {"prefill", "both"}:
            base_cache = DynamicCache(config=base_model.config)
            mix_cache = MixKVQCache(mix_model.config, mix_config)
            base_logits = run_prefill(base_model, input_ids, base_cache)
            mix_logits = run_prefill(mix_model, input_ids, mix_cache)
            _compare_logits("prefill", base_logits, mix_logits)

        if args.mode in {"incremental", "both"}:
            base_cache = DynamicCache(config=base_model.config)
            mix_cache = MixKVQCache(mix_model.config, mix_config)
            base_logits = run_incremental(base_model, input_ids, base_cache)
            mix_logits = run_incremental(mix_model, input_ids, mix_cache)
            _compare_logits("incremental", base_logits, mix_logits)


if __name__ == "__main__":
    main()
