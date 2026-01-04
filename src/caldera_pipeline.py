from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scales: torch.Tensor
    group_size: int
    num_bits: int


def _pad_to_group(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    last_dim = x.shape[-1]
    remainder = last_dim % group_size
    if remainder == 0:
        return x, 0
    pad = group_size - remainder
    pad_shape = list(x.shape)
    pad_shape[-1] = pad
    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=-1), pad


def quantize_groupwise(x: torch.Tensor, group_size: int, num_bits: int) -> QuantizedTensor:
    if num_bits < 2 or num_bits > 8:
        raise ValueError("num_bits must be in [2, 8]")
    x = x.float()
    padded, pad = _pad_to_group(x, group_size)
    last_dim = padded.shape[-1]
    groups = last_dim // group_size

    reshaped = padded.reshape(-1, groups, group_size)
    max_abs = torch.max(torch.abs(reshaped), dim=-1, keepdim=True).values
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
    quant = torch.round(reshaped / scale).clamp(qmin, qmax).to(torch.int8)

    if pad:
        quant = quant[:, :, : group_size - pad]

    quant = quant.reshape(x.shape)
    scales = scale.reshape(*x.shape[:-1], groups)

    return QuantizedTensor(values=quant, scales=scales, group_size=group_size, num_bits=num_bits)


def dequantize_groupwise(q: QuantizedTensor) -> torch.Tensor:
    values = q.values.float()
    group_size = q.group_size
    last_dim = values.shape[-1]
    groups = (last_dim + group_size - 1) // group_size

    padded, pad = _pad_to_group(values, group_size)
    reshaped = padded.reshape(-1, groups, group_size)

    scales = q.scales.float().reshape(-1, groups, 1)
    dequant = reshaped * scales
    if pad:
        dequant = dequant[:, :, : group_size - pad]

    return dequant.reshape(values.shape)


def _resolve_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_calibration_npz(path: Path) -> TensorDataset:
    data = np.load(path)
    input_ids = torch.from_numpy(data["input_ids"])
    attention_mask = torch.from_numpy(data["attention_mask"])
    return TensorDataset(input_ids, attention_mask)


def _iter_target_modules(model: nn.Module, target_suffixes: Iterable[str]) -> list[tuple[str, nn.Module]]:
    suffixes = tuple(target_suffixes)
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if suffixes and not any(name.endswith(suffix) for suffix in suffixes):
            continue
        modules.append((name, module))
    return modules


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def _layer_overrides(
    name: str,
    base_cfg: dict,
    overrides: dict,
    pattern_overrides: dict,
) -> dict:
    layer_cfg = dict(base_cfg)
    if name in overrides:
        layer_cfg.update(overrides[name])
    for pattern, values in pattern_overrides.items():
        if fnmatch.fnmatch(name, pattern):
            layer_cfg.update(values)
    return layer_cfg


def _collect_inputs(
    model: nn.Module,
    module: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> torch.Tensor:
    inputs: list[torch.Tensor] = []
    total = 0

    def hook(_module: nn.Module, args: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        nonlocal total
        if total >= max_samples:
            return
        x = args[0].detach().float().cpu()
        x = x.reshape(-1, x.shape[-1])
        if total + x.shape[0] > max_samples:
            x = x[: max_samples - total]
        inputs.append(x)
        total += x.shape[0]

    handle = module.register_forward_hook(hook)
    model.eval()

    # Detect input device for sharded models
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Model is sharded - find the input device (usually the embed_tokens device)
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            input_device = torch.device(f"cuda:{first_device}")
        else:
            input_device = torch.device(first_device)
    else:
        input_device = device

    with torch.no_grad():
        for batch in dataloader:
            if total >= max_samples:
                break
            input_ids, attention_mask = (item.to(input_device) for item in batch)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()

    if not inputs:
        raise RuntimeError("No calibration inputs collected for module.")
    return torch.cat(inputs, dim=0)


def _low_rank_from_svd(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    root = torch.sqrt(s)
    l = u * root
    r = root[:, None] * vh
    return l, r


def _solve_r_from_calibration(z: torch.Tensor, x: torch.Tensor, ridge: float) -> torch.Tensor:
    xtx = x.t() @ x
    ridge = max(ridge, 1e-6)
    xtx = xtx + ridge * torch.eye(xtx.shape[0], device=xtx.device, dtype=xtx.dtype)
    rhs = (z @ x).t()
    try:
        chol = torch.linalg.cholesky(xtx)
        r_t = torch.cholesky_solve(rhs, chol)
    except RuntimeError:
        r_t = torch.linalg.solve(xtx, rhs)
    return r_t.t()


def decompose_weight(
    weight: torch.Tensor,
    rank: int,
    bits_q: int,
    bits_lr: int,
    group_size: int,
    calibration_x: torch.Tensor | None = None,
    ridge: float = 0.0,
    device: torch.device | None = None,
) -> dict[str, QuantizedTensor]:
    device = device or weight.device
    weight = weight.to(device=device, dtype=torch.float32)

    q_q = quantize_groupwise(weight, group_size, bits_q)
    q_deq = dequantize_groupwise(q_q)
    residual = weight - q_deq

    if calibration_x is not None:
        x = calibration_x.to(device=device, dtype=torch.float32)
        residual_xt = residual @ x.t()
        l, z = _low_rank_from_svd(residual_xt, rank)
        r = _solve_r_from_calibration(z, x, ridge)
    else:
        l, r = _low_rank_from_svd(residual, rank)

    l_q = quantize_groupwise(l, group_size, bits_lr)
    r_q = quantize_groupwise(r, group_size, bits_lr)

    return {
        "q": q_q,
        "l": l_q,
        "r": r_q,
    }


def compress_model(config: dict) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = config.get("runtime", {})
    dtype = _resolve_dtype(runtime_cfg.get("dtype", "bf16"))
    device_map = runtime_cfg.get("device_map", None)
    device = torch.device(runtime_cfg.get("device", "cuda"))

    model_id = config["model_id"]
    calibration_path = output_dir / "calibration.npz"
    dataset = _load_calibration_npz(calibration_path)
    batch_size = int(config.get("caldera", {}).get("calibration_batch_size", 1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if device_map:
        # Multi-GPU loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
    else:
        # Single-GPU loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=None,
        )
        model.to(device)

    caldera_cfg = config["caldera"]
    compute_device = torch.device(caldera_cfg.get("compute_device", "cpu"))
    target_suffixes = caldera_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    skip_layers = caldera_cfg.get("skip_layers", [])
    layer_overrides = caldera_cfg.get("layer_overrides", {})
    pattern_overrides = caldera_cfg.get("pattern_overrides", {})

    modules = _iter_target_modules(model, target_suffixes)
    max_modules = caldera_cfg.get("max_modules")
    if max_modules:
        modules = modules[: int(max_modules)]

    stats = {
        "compressed_layers": [],
        "total_layers": len(modules),
    }

    for name, module in modules:
        if _matches_any(name, skip_layers):
            print(f"[caldera] skipping {name}")
            continue
        layer_cfg = _layer_overrides(name, caldera_cfg, layer_overrides, pattern_overrides)
        if layer_cfg.get("skip", False):
            print(f"[caldera] skipping {name} (override)")
            continue
        print(f"[caldera] compressing {name}")
        calibration_samples = int(layer_cfg.get("calibration_samples", 1024))
        calibration_x = _collect_inputs(model, module, dataloader, device, calibration_samples)

        result = decompose_weight(
            module.weight.data,
            rank=int(layer_cfg.get("rank", 128)),
            bits_q=int(layer_cfg.get("bq", 2)),
            bits_lr=int(layer_cfg.get("bl", 4)),
            group_size=int(layer_cfg.get("group_size", 128)),
            calibration_x=calibration_x if layer_cfg.get("use_calibration", True) else None,
            ridge=float(layer_cfg.get("ridge_lambda", 0.0)),
            device=compute_device,
        )

        layer_path = output_dir / "layers" / f"{name.replace('.', '_')}.pt"
        layer_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_values": result["q"].values.cpu(),
                "q_scales": result["q"].scales.cpu(),
                "l_values": result["l"].values.cpu(),
                "l_scales": result["l"].scales.cpu(),
                "r_values": result["r"].values.cpu(),
                "r_scales": result["r"].scales.cpu(),
                "meta": {
                    "module_name": name,
                    "group_size": result["q"].group_size,
                    "bits_q": result["q"].num_bits,
                    "bits_lr": result["l"].num_bits,
                    "rank": int(layer_cfg.get("rank", 128)),
                    "weight_shape": tuple(module.weight.shape),
                },
            },
            layer_path,
        )

        stats["compressed_layers"].append(name)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "compression_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
