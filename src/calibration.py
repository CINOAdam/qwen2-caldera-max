from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


def _require_dependency(name: str, install_hint: str) -> Any:
    try:
        return __import__(name)
    except Exception as exc:  # pragma: no cover - env dependent
        raise SystemExit(f"Missing dependency: {name}. Install with `{install_hint}`.") from exc


datasets = _require_dependency("datasets", "pip install datasets")
transformers = _require_dependency("transformers", "pip install transformers")
np = _require_dependency("numpy", "pip install numpy")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    weight: float
    hf_id: str
    split: str
    text_field: str
    config: str | None = None
    format: str = "plain"


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "c4": {
        "hf_id": "allenai/c4",
        "config": "en",
        "split": "train",
        "text_field": "text",
        "format": "plain",
    },
    "sharegpt": {
        "hf_id": "Aeala/ShareGPT_Vicuna_unfiltered",
        "config": None,
        "split": "train",
        "text_field": "conversations",
        "format": "sharegpt",
    },
}


def _resolve_dataset_spec(item: dict[str, Any]) -> DatasetSpec:
    name = item["name"]
    if name not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise SystemExit(f"Unknown dataset '{name}'. Known: {known}")
    base = DATASET_REGISTRY[name]
    return DatasetSpec(
        name=name,
        weight=float(item.get("weight", 1.0)),
        hf_id=base["hf_id"],
        split=base["split"],
        text_field=base["text_field"],
        config=base.get("config"),
        format=base.get("format", "plain"),
    )


def _format_sharegpt(conversations: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in conversations:
        role = (turn.get("from") or turn.get("role") or "").lower()
        if role in {"human", "user"}:
            label = "User"
        elif role in {"assistant", "gpt", "bot"}:
            label = "Assistant"
        else:
            label = role.title() if role else "User"
        content = turn.get("value") or turn.get("content") or ""
        content = str(content).strip()
        if content:
            lines.append(f"{label}: {content}")
    return "\n".join(lines).strip()


def _extract_text(spec: DatasetSpec, row: dict[str, Any]) -> str:
    if spec.format == "sharegpt":
        conversations = row.get(spec.text_field) or []
        if isinstance(conversations, str):
            return conversations.strip()
        if isinstance(conversations, list):
            return _format_sharegpt(conversations)
        return ""
    value = row.get(spec.text_field, "")
    return str(value).strip()


def _load_streaming_dataset(spec: DatasetSpec, seed: int, buffer_size: int, token: str | None):
    kwargs = {
        "path": spec.hf_id,
        "split": spec.split,
        "streaming": True,
    }
    if spec.config:
        kwargs["name"] = spec.config
    if token:
        kwargs["token"] = token
    dataset = datasets.load_dataset(**kwargs)
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)


def _sample_counts(specs: list[DatasetSpec], total: int) -> dict[str, int]:
    weight_sum = sum(spec.weight for spec in specs)
    if weight_sum <= 0:
        raise SystemExit("Calibration dataset weights must sum to > 0.")
    counts: dict[str, int] = {}
    remaining = total
    for spec in specs[:-1]:
        share = spec.weight / weight_sum
        count = int(math.floor(total * share))
        counts[spec.name] = count
        remaining -= count
    counts[specs[-1].name] = remaining
    return counts


def build_calibration(
    model_id: str,
    dataset_items: list[dict[str, Any]],
    output_dir: Path,
    samples: int,
    sequence_length: int,
    seed: int = 42,
    buffer_size: int = 10_000,
    progress_every: int = 200,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [_resolve_dataset_spec(item) for item in dataset_items]
    counts = _sample_counts(specs, samples)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    dataset_stats: dict[str, int] = {spec.name: 0 for spec in specs}
    start_time = time.time()

    for spec in specs:
        dataset_start = time.time()
        dataset = _load_streaming_dataset(spec, seed=seed, buffer_size=buffer_size, token=hf_token)
        target = counts[spec.name]
        for row in dataset:
            if dataset_stats[spec.name] >= target:
                break
            text = _extract_text(spec, row)
            if not text:
                continue
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=sequence_length,
                padding="max_length",
            )
            input_ids_list.append(encoded["input_ids"])
            attention_mask_list.append(encoded["attention_mask"])
            dataset_stats[spec.name] += 1
            if progress_every and (
                dataset_stats[spec.name] % progress_every == 0
                or dataset_stats[spec.name] == target
            ):
                elapsed = time.time() - dataset_start
                total_elapsed = time.time() - start_time
                print(
                    f"[calibration] {spec.name}: {dataset_stats[spec.name]}/{target} "
                    f"samples ({elapsed:.1f}s, total {total_elapsed:.1f}s)",
                    flush=True,
                )
        if dataset_stats[spec.name] < target:
            raise SystemExit(
                f"Dataset '{spec.name}' yielded {dataset_stats[spec.name]} samples "
                f"but {target} were requested."
            )

    input_ids = np.array(input_ids_list, dtype=np.int32)
    attention_mask = np.array(attention_mask_list, dtype=np.uint8)

    np.savez_compressed(
        output_dir / "calibration.npz",
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    stats = {
        "samples": int(input_ids.shape[0]),
        "sequence_length": int(sequence_length),
        "dataset_counts": dataset_stats,
        "model_id": model_id,
    }
    return stats
