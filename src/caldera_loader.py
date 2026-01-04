from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import hub

from src.caldera_runtime import apply_caldera, offload_embeddings

try:  # optional dependency
    from safetensors.torch import safe_open  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    safe_open = None

try:  # optional dependency
    from accelerate import init_empty_weights  # type: ignore
    from accelerate.utils import modeling as accel_modeling  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    init_empty_weights = None
    accel_modeling = None


@dataclass(frozen=True)
class LoadReport:
    skipped_weights: int
    loaded_tensors: int
    used_init_empty: bool


def _load_index_file(model_id: str) -> Path:
    candidates = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    for filename in candidates:
        try:
            return Path(hub.cached_file(model_id, filename))
        except Exception:
            continue
    raise SystemExit(
        "Unable to locate a model index file. Try `pip install safetensors` and ensure the "
        "model has sharded weights."
    )


def _collect_caldera_modules(artifacts_dir: Path) -> list[str]:
    layers_dir = artifacts_dir / "layers"
    if not layers_dir.exists():
        raise SystemExit(f"Missing CALDERA layers in {layers_dir}")
    module_names = []
    for path in layers_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})
        name = meta.get("module_name")
        if name:
            module_names.append(name)
    if not module_names:
        raise SystemExit(
            "CALDERA artifacts do not include module names. Re-run compression with the "
            "updated pipeline to populate meta.module_name."
        )
    return module_names


def _skip_weight_keys(module_names: Iterable[str]) -> set[str]:
    return {f"{name}.weight" for name in module_names}


def _load_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        model.load_state_dict(state_dict, strict=False)


def load_model_with_caldera(
    model_id: str,
    artifacts_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cache_dequant: bool = False,
    cache_mode: str | None = None,
    chunk_size: int | None = 1024,
    offload_input_embeddings: bool = False,
    offload_output_embeddings: bool = False,
) -> tuple[torch.nn.Module, LoadReport]:
    module_names = _collect_caldera_modules(artifacts_dir)
    skip_keys = _skip_weight_keys(module_names)

    index_path = _load_index_file(model_id)
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    shard_map: dict[str, list[str]] = {}
    skipped = 0
    for name, shard in weight_map.items():
        if name in skip_keys:
            skipped += 1
            continue
        shard_map.setdefault(shard, []).append(name)

    if index_path.name.endswith(".bin.index.json"):
        raise SystemExit(
            "Bin index file detected; selective loading requires safetensors shards. "
            "Try using a safetensors model or convert the checkpoint."
        )

    if safe_open is None:
        raise SystemExit("Missing dependency: safetensors. Install with `pip install safetensors`.")

    config = AutoConfig.from_pretrained(model_id)
    used_init = init_empty_weights is not None and accel_modeling is not None
    if init_empty_weights is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=None)
    else:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        model.tie_weights()
        if accel_modeling is None:
            raise SystemExit("Missing accelerate.utils.modeling; cannot load selectively.")
        tied_params = accel_modeling.find_tied_parameters(model)

        loaded_tensors = 0
        for shard, keys in shard_map.items():
            shard_path = Path(hub.cached_file(model_id, shard))
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in keys:
                    tensor = handle.get_tensor(key)
                    if tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    param_device = device
                    if offload_input_embeddings and key.startswith("model.embed_tokens."):
                        param_device = torch.device("cpu")
                    if offload_output_embeddings and key.startswith("lm_head."):
                        param_device = torch.device("cpu")
                    accel_modeling.set_module_tensor_to_device(
                        model,
                        key,
                        param_device,
                        value=tensor,
                        dtype=dtype,
                    )
                    loaded_tensors += 1
        replaced = apply_caldera(
            model,
            artifacts_dir,
            cache_dequant=cache_dequant,
            cache_mode=cache_mode,
            chunk_size=chunk_size,
            device=device,
        )
        if not replaced:
            raise SystemExit(f"No CALDERA layers found in {artifacts_dir}/layers.")

        accel_modeling.retie_parameters(model, tied_params)

        if offload_input_embeddings or offload_output_embeddings:
            offload_embeddings(
                model,
                device,
                offload_input=offload_input_embeddings,
                offload_output=offload_output_embeddings,
            )

        return model, LoadReport(skipped_weights=skipped, loaded_tensors=loaded_tensors, used_init_empty=True)

    replaced = apply_caldera(
        model,
        artifacts_dir,
        cache_dequant=cache_dequant,
        cache_mode=cache_mode,
        chunk_size=chunk_size,
    )
    if not replaced:
        raise SystemExit(f"No CALDERA layers found in {artifacts_dir}/layers.")
    if offload_input_embeddings or offload_output_embeddings:
        offload_embeddings(
            model,
            device,
            offload_input=offload_input_embeddings,
            offload_output=offload_output_embeddings,
        )
    model.to(device)
    return model, LoadReport(skipped_weights=skipped, loaded_tensors=0, used_init_empty=used_init)
