#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.caldera_loader import load_model_with_caldera
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


def load_dataset_texts(
    name: str,
    config: str,
    split: str,
    max_samples: int,
    column: str,
    columns: list[str] | None,
) -> list[str]:
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`.") from exc

    dataset = datasets.load_dataset(name, config, split=split)
    texts = []
    for row in dataset:
        if columns:
            parts = [str(row.get(col, "")).strip() for col in columns]
            text = "\n".join([part for part in parts if part])
        else:
            text = row.get(column, "")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def _sample_pairs(
    attention_mask: torch.Tensor,
    samples_per_seq: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    pairs: list[tuple[int, int]] = []
    batch, seq_len = attention_mask.shape
    for b in range(batch):
        valid = torch.nonzero(attention_mask[b], as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        count = min(samples_per_seq, valid.numel())
        perm = torch.randperm(valid.numel(), generator=generator)[:count]
        sampled = valid[perm].tolist()
        for pos in sampled:
            pairs.append((b, int(pos)))
    if not pairs:
        return None
    pair_batch = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    pair_pos = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return pair_batch, pair_pos


def _sampled_fidelity(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    embeddings: torch.Tensor,
    negatives: int,
    generator: torch.Generator,
) -> tuple[int, float, int]:
    if hidden.numel() == 0:
        return 0, 0.0, 0
    hidden = F.normalize(hidden.float(), dim=-1)
    target_ids = target_ids.to(torch.long)
    vocab_size = embeddings.shape[0]

    if negatives <= 0:
        cand_ids = target_ids.unsqueeze(1)
    else:
        neg = torch.randint(0, vocab_size, (hidden.shape[0], negatives), generator=generator)
        mask = neg.eq(target_ids.unsqueeze(1))
        while mask.any():
            neg[mask] = torch.randint(0, vocab_size, (mask.sum().item(),), generator=generator)
            mask = neg.eq(target_ids.unsqueeze(1))
        cand_ids = torch.cat([target_ids.unsqueeze(1), neg], dim=1)

    cand_embeds = embeddings[cand_ids].float()
    cand_embeds = F.normalize(cand_embeds, dim=-1)
    sims = torch.einsum("nd,nkd->nk", hidden, cand_embeds)

    preds = sims.argmax(dim=1)
    correct = int((preds == 0).sum().item())
    if cand_ids.shape[1] > 1:
        best_neg = sims[:, 1:].max(dim=1).values
        margin = float((sims[:, 0] - best_neg).sum().item())
    else:
        margin = float(sims[:, 0].sum().item())
    total = int(hidden.shape[0])
    return correct, margin, total


def _block_modules(block_idx: int, target_suffixes: list[str]) -> list[str]:
    base = f"model.layers.{block_idx}"
    modules = []
    for suffix in target_suffixes:
        if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            modules.append(f"{base}.self_attn.{suffix}")
        else:
            modules.append(f"{base}.mlp.{suffix}")
    return modules


class _LayerCollector:
    def __init__(
        self,
        embeddings: torch.Tensor,
        negatives: int,
        samples_per_seq: int,
        generator: torch.Generator,
        device: torch.device,
        num_layers: int,
    ) -> None:
        self.embeddings = embeddings
        self.negatives = negatives
        self.samples_per_seq = samples_per_seq
        self.generator = generator
        self.device = device
        self.layer_correct = [0 for _ in range(num_layers)]
        self.layer_total = [0 for _ in range(num_layers)]
        self.layer_margin = [0.0 for _ in range(num_layers)]
        self.pair_batch: torch.Tensor | None = None
        self.pair_pos: torch.Tensor | None = None
        self.target_ids: torch.Tensor | None = None

    def set_pairs(self, input_ids_cpu: torch.Tensor, attention_mask_cpu: torch.Tensor) -> bool:
        pairs = _sample_pairs(attention_mask_cpu, self.samples_per_seq, self.generator)
        if pairs is None:
            self.pair_batch = None
            self.pair_pos = None
            self.target_ids = None
            return False
        pair_batch_cpu, pair_pos_cpu = pairs
        self.pair_batch = pair_batch_cpu.to(self.device)
        self.pair_pos = pair_pos_cpu.to(self.device)
        self.target_ids = input_ids_cpu[pair_batch_cpu, pair_pos_cpu]
        return True

    def process_layer(self, layer_idx: int, hidden: torch.Tensor | tuple | list) -> None:
        if self.pair_batch is None or self.pair_pos is None or self.target_ids is None:
            return
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        selected = hidden[self.pair_batch, self.pair_pos].detach().cpu()
        correct, margin, total = _sampled_fidelity(
            selected,
            self.target_ids,
            self.embeddings,
            self.negatives,
            self.generator,
        )
        self.layer_correct[layer_idx] += correct
        self.layer_total[layer_idx] += total
        self.layer_margin[layer_idx] += margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate layer sensitivity via SipIt-style fidelity.")
    parser.add_argument("--model-id", required=True, help="Base model ID or local path.")
    parser.add_argument("--caldera-dir", type=Path, default=None, help="Path to CALDERA artifacts dir.")
    parser.add_argument("--dataset", default="wikitext", help="HF dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="HF dataset config.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--dataset-column", default="text", help="Dataset text column.")
    parser.add_argument(
        "--dataset-columns",
        nargs="*",
        default=None,
        help="Optional columns to join instead of --dataset-column.",
    )
    parser.add_argument("--samples", type=int, default=16, help="Number of samples to score.")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--samples-per-seq", type=int, default=8, help="Token positions per sample.")
    parser.add_argument("--negatives", type=int, default=2048, help="Negative tokens per position.")
    parser.add_argument("--device", default="cuda", help="Device for evaluation.")
    parser.add_argument("--dtype", default="bf16", help="Model dtype.")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results here.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.skip_linear_weights:
        if args.caldera_dir is None:
            raise SystemExit("--skip-linear-weights requires --caldera-dir.")
        model, _ = load_model_with_caldera(
            args.model_id,
            args.caldera_dir,
            device=device,
            dtype=dtype,
            cache_dequant=args.cache_dequant,
            chunk_size=args.chunk_size,
            offload_input_embeddings=args.offload_embeddings,
            offload_output_embeddings=args.offload_lm_head,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map=None,
        ).to(device)

    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    embed_module = model.get_input_embeddings()
    if hasattr(embed_module, "weight"):
        embed_weight = embed_module.weight
    elif hasattr(embed_module, "embedding") and hasattr(embed_module.embedding, "weight"):
        embed_weight = embed_module.embedding.weight
    else:
        raise SystemExit("Unable to locate embedding weights (unexpected embedding wrapper).")
    embeddings = embed_weight.detach().cpu().to(torch.float16)
    generator = torch.Generator().manual_seed(args.seed)

    texts = load_dataset_texts(
        args.dataset,
        args.dataset_config,
        args.split,
        args.samples,
        args.dataset_column,
        args.dataset_columns,
    )

    if not texts:
        raise SystemExit("No dataset samples loaded; check dataset settings.")

    num_layers = model.config.num_hidden_layers + 1
    collector = _LayerCollector(
        embeddings=embeddings,
        negatives=args.negatives,
        samples_per_seq=args.samples_per_seq,
        generator=generator,
        device=device,
        num_layers=num_layers,
    )

    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    with torch.no_grad():
        handles = []
        try:
            embed_module = model.get_input_embeddings()

            def _embed_hook(_module, _inp, output):
                collector.process_layer(0, output)

            handles.append(embed_module.register_forward_hook(_embed_hook))

            layers = getattr(model.model, "layers", None)
            if layers is None:
                raise SystemExit("Unexpected model structure: missing model.layers for hook setup.")

            for idx, layer in enumerate(layers, start=1):
                def _make_hook(layer_idx: int):
                    def _hook(_module, _inp, output):
                        collector.process_layer(layer_idx, output)
                    return _hook

                handles.append(layer.register_forward_hook(_make_hook(idx)))

            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                input_ids_cpu = encoded["input_ids"].cpu()
                attention_mask_cpu = encoded["attention_mask"].cpu()

                if not collector.set_pairs(input_ids_cpu, attention_mask_cpu):
                    continue

                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            for handle in handles:
                handle.remove()

    results = []
    for idx in range(num_layers):
        total = collector.layer_total[idx]
        fidelity = collector.layer_correct[idx] / total if total else 0.0
        margin = collector.layer_margin[idx] / total if total else 0.0
        if idx == 0:
            label = "embeddings"
            modules = []
            block_idx = None
        else:
            block_idx = idx - 1
            label = f"layer_{block_idx}"
            modules = _block_modules(block_idx, target_suffixes)
        results.append(
            {
                "index": idx,
                "label": label,
                "block_index": block_idx,
                "fidelity": fidelity,
                "margin": margin,
                "modules": modules,
                "samples": total,
            }
        )

    payload = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "samples": args.samples,
        "samples_per_seq": args.samples_per_seq,
        "negatives": args.negatives,
        "max_length": args.max_length,
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    else:
        for row in sorted(results, key=lambda item: item["fidelity"]):
            print(f"{row['label']}: fidelity={row['fidelity']:.4f} margin={row['margin']:.4f}")


if __name__ == "__main__":
    main()
