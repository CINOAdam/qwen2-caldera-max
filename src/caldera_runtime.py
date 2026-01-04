from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from src.packing import unpack_packed


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scales: torch.Tensor
    group_size: int
    num_bits: int
    packed_bits: int | None = None
    packed_cols: int | None = None


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


def dequantize_groupwise(q: QuantizedTensor) -> torch.Tensor:
    values = q.values.float()
    group_size = q.group_size
    last_dim = values.shape[-1]
    groups = math.ceil(last_dim / group_size)

    padded, pad = _pad_to_group(values, group_size)
    reshaped = padded.reshape(-1, groups, group_size)

    scales = q.scales.float().reshape(-1, groups, 1)
    dequant = reshaped * scales
    if pad:
        dequant = dequant[:, :, : group_size - pad]

    return dequant.reshape(values.shape)


class CalderaLinear(nn.Module):
    def __init__(
        self,
        q_weight: QuantizedTensor,
        l_weight: QuantizedTensor,
        r_weight: QuantizedTensor,
        bias: torch.Tensor | None = None,
        cache_dequant: bool = False,
        cache_mode: str | None = None,
        chunk_size: int | None = 1024,
    ) -> None:
        super().__init__()
        self.in_features = q_weight.values.shape[1]
        self.out_features = q_weight.values.shape[0]
        self.group_size = q_weight.group_size
        if cache_mode is None:
            cache_mode = "qlr" if cache_dequant else "none"
        elif cache_dequant:
            raise ValueError("Specify either cache_dequant or cache_mode, not both.")
        if cache_mode not in {"none", "r", "lr", "qlr"}:
            raise ValueError("cache_mode must be one of: none, r, lr, qlr.")
        self.cache_mode = cache_mode
        self.cache_dequant = cache_mode == "qlr"
        self.chunk_size = chunk_size

        self.register_buffer("q_values", q_weight.values)
        self.register_buffer("q_scales", q_weight.scales)
        self.register_buffer("l_values", l_weight.values)
        self.register_buffer("l_scales", l_weight.scales)
        self.register_buffer("r_values", r_weight.values)
        self.register_buffer("r_scales", r_weight.scales)
        self.q_pack_bits = q_weight.packed_bits
        self.q_packed_cols = q_weight.packed_cols
        self.l_pack_bits = l_weight.packed_bits
        self.l_packed_cols = l_weight.packed_cols
        self.r_pack_bits = r_weight.packed_bits
        self.r_packed_cols = r_weight.packed_cols

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)

        self._q_deq: torch.Tensor | None = None
        self._l_deq: torch.Tensor | None = None
        self._r_deq: torch.Tensor | None = None

    def _unpack_values(
        self,
        values: torch.Tensor,
        pack_bits: int | None,
        packed_cols: int | None,
    ) -> torch.Tensor:
        if pack_bits is None:
            return values.to(torch.int8)
        if packed_cols is None:
            raise ValueError("Packed columns metadata is missing.")
        return unpack_packed(values, pack_bits, packed_cols)

    def _cache_q(self) -> bool:
        return self.cache_mode == "qlr"

    def _cache_l(self) -> bool:
        return self.cache_mode in {"qlr", "lr"}

    def _cache_r(self) -> bool:
        return self.cache_mode in {"qlr", "lr", "r"}

    def _maybe_dequant_q(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cache_q()
            and self._q_deq is not None
            and self._q_deq.dtype == dtype
            and self._q_deq.device == device
        ):
            return self._q_deq

        q_vals = self._unpack_values(self.q_values, self.q_pack_bits, self.q_packed_cols)
        q = dequantize_groupwise(
            QuantizedTensor(q_vals, self.q_scales, self.group_size, 0)
        )
        if q.dtype != dtype or q.device != device:
            q = q.to(device=device, dtype=dtype)

        if self._cache_q():
            self._q_deq = q

        return q

    def _maybe_dequant_l(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cache_l()
            and self._l_deq is not None
            and self._l_deq.dtype == dtype
            and self._l_deq.device == device
        ):
            return self._l_deq

        l_vals = self._unpack_values(self.l_values, self.l_pack_bits, self.l_packed_cols)
        l = dequantize_groupwise(
            QuantizedTensor(l_vals, self.l_scales, self.group_size, 0)
        )
        if l.dtype != dtype or l.device != device:
            l = l.to(device=device, dtype=dtype)

        if self._cache_l():
            self._l_deq = l

        return l

    def _maybe_dequant_r(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cache_r()
            and self._r_deq is not None
            and self._r_deq.dtype == dtype
            and self._r_deq.device == device
        ):
            return self._r_deq

        r_vals = self._unpack_values(self.r_values, self.r_pack_bits, self.r_packed_cols)
        r = dequantize_groupwise(
            QuantizedTensor(r_vals, self.r_scales, self.group_size, 0)
        )
        if r.dtype != dtype or r.device != device:
            r = r.to(device=device, dtype=dtype)

        if self._cache_r():
            self._r_deq = r

        return r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        device = x.device
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])

        if self.chunk_size is None or self.chunk_size >= self.out_features:
            q = self._maybe_dequant_q(dtype, device)
            l = self._maybe_dequant_l(dtype, device)
            r = self._maybe_dequant_r(dtype, device)
            xr = x @ r.t()
            base = x @ q.t()
            low_rank = xr @ l.t()
            out = base + low_rank
        else:
            r = self._maybe_dequant_r(dtype, device)
            xr = x @ r.t()
            q_full = None
            l_full = None
            if self._cache_q():
                q_full = self._maybe_dequant_q(dtype, device)
            if self._cache_l():
                l_full = self._maybe_dequant_l(dtype, device)
            outputs = []
            for start in range(0, self.out_features, self.chunk_size):
                end = min(start + self.chunk_size, self.out_features)
                if q_full is None:
                    q_vals = self._unpack_values(
                        self.q_values[start:end],
                        self.q_pack_bits,
                        self.q_packed_cols,
                    )
                    q_chunk = dequantize_groupwise(
                        QuantizedTensor(
                            q_vals,
                            self.q_scales[start:end],
                            self.group_size,
                            0,
                        )
                    ).to(device=device, dtype=dtype)
                else:
                    q_chunk = q_full[start:end]

                if l_full is None:
                    l_vals = self._unpack_values(
                        self.l_values[start:end],
                        self.l_pack_bits,
                        self.l_packed_cols,
                    )
                    l_chunk = dequantize_groupwise(
                        QuantizedTensor(
                            l_vals,
                            self.l_scales[start:end],
                            self.group_size,
                            0,
                        )
                    ).to(device=device, dtype=dtype)
                else:
                    l_chunk = l_full[start:end]
                base = x @ q_chunk.t()
                low_rank = xr @ l_chunk.t()
                outputs.append(base + low_rank)
            out = torch.cat(outputs, dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*original_shape[:-1], self.out_features)


class OffloadedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, output_device: torch.device) -> None:
        super().__init__()
        self.embedding = embedding
        self.output_device = output_device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.device != self.embedding.weight.device:
            input_ids = input_ids.to(self.embedding.weight.device)
        out = self.embedding(input_ids)
        if out.device != self.output_device:
            out = out.to(self.output_device)
        return out


class OffloadedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, output_device: torch.device | None) -> None:
        super().__init__()
        self.linear = linear
        self.output_device = output_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.linear.weight.device:
            x = x.to(self.linear.weight.device)
        out = self.linear(x)
        if self.output_device is not None and out.device != self.output_device:
            out = out.to(self.output_device)
        return out


def _load_quantized(name: str, data: dict, group_size: int, num_bits: int) -> QuantizedTensor:
    if f"{name}_packed" in data:
        values = data[f"{name}_packed"]
        packed_bits = int(data[f"{name}_packed_bits"])
        packed_cols = int(data[f"{name}_packed_cols"])
    else:
        values = data[f"{name}_values"]
        packed_bits = None
        packed_cols = None
    scales = data[f"{name}_scales"]
    return QuantizedTensor(
        values=values,
        scales=scales,
        group_size=group_size,
        num_bits=num_bits,
        packed_bits=packed_bits,
        packed_cols=packed_cols,
    )


def load_caldera_layer(
    path: Path,
    bias: torch.Tensor | None = None,
    cache_dequant: bool = False,
    cache_mode: str | None = None,
    chunk_size: int | None = 1024,
) -> CalderaLinear:
    payload = torch.load(path, map_location="cpu")
    meta = payload["meta"]
    group_size = int(meta["group_size"])
    q = _load_quantized("q", payload, group_size, int(meta.get("bits_q", 0)))
    l = _load_quantized("l", payload, group_size, int(meta.get("bits_lr", 0)))
    r = _load_quantized("r", payload, group_size, int(meta.get("bits_lr", 0)))
    return CalderaLinear(
        q,
        l,
        r,
        bias=bias,
        cache_dequant=cache_dequant,
        cache_mode=cache_mode,
        chunk_size=chunk_size,
    )


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _get_device_for_module(model: nn.Module, module_name: str) -> torch.device | None:
    """Get the target device for a module from model's hf_device_map.

    For sharded models with device_map='auto', the hf_device_map stores device
    assignments at the block/layer level, not individual submodules. We need to
    find the parent layer that contains this module.
    """
    if not hasattr(model, "hf_device_map") or not model.hf_device_map:
        return None

    device_map = model.hf_device_map

    # Try exact match first
    if module_name in device_map:
        dev = device_map[module_name]
        if isinstance(dev, int):
            return torch.device(f"cuda:{dev}")
        return torch.device(dev)

    # Try parent prefixes (e.g., "model.layers.0.self_attn.q_proj" -> "model.layers.0")
    parts = module_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in device_map:
            dev = device_map[prefix]
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            return torch.device(dev)

    return None


def apply_caldera(
    model: nn.Module,
    artifacts_dir: Path,
    *,
    cache_dequant: bool = False,
    cache_mode: str | None = None,
    chunk_size: int | None = 1024,
    device: torch.device | None = None,
    strict: bool = False,
) -> list[str]:
    layers_dir = artifacts_dir / "layers"
    replaced: list[str] = []

    # Collect modules to replace first (can't modify during iteration)
    modules_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        artifact = layers_dir / f"{name.replace('.', '_')}.pt"
        if not artifact.exists():
            continue
        modules_to_replace.append((name, module, artifact))

    for name, module, artifact in modules_to_replace:
        # Determine target device BEFORE freeing the original module
        if device is not None:
            target_device = device
        else:
            mapped_device = _get_device_for_module(model, name)
            if mapped_device is not None:
                target_device = mapped_device
            else:
                weight_device = module.weight.device
                if weight_device.type == "meta":
                    try:
                        target_device = next(
                            param.device for param in model.parameters() if not param.is_meta
                        )
                    except StopIteration:
                        target_device = torch.device("cpu")
                else:
                    target_device = weight_device

        # Save bias before freeing the module
        bias = module.bias.detach().clone().cpu() if module.bias is not None else None

        # FREE THE ORIGINAL MODULE'S MEMORY before loading replacement
        # Move weight to CPU to free GPU memory
        if module.weight.device.type == "cuda":
            module.weight.data = module.weight.data.to("cpu")
            if module.bias is not None:
                module.bias.data = module.bias.data.to("cpu")
            torch.cuda.empty_cache()

        # Load CALDERA layer (starts on CPU)
        caldera_layer = load_caldera_layer(
            artifact,
            bias=bias,
            cache_dequant=cache_dequant,
            cache_mode=cache_mode,
            chunk_size=chunk_size,
        )

        # Move to target device
        caldera_layer = caldera_layer.to(target_device)
        _set_module(model, name, caldera_layer)
        replaced.append(name)

        # Clear cache after each replacement to free memory
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    if strict and not replaced:
        raise RuntimeError(f"No CALDERA layers found in {layers_dir}")

    return replaced


def offload_embeddings(
    model: nn.Module,
    output_device: torch.device,
    *,
    offload_input: bool = True,
    offload_output: bool = True,
) -> list[str]:
    replaced: list[str] = []

    if offload_input and hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        if embedding is not None:
            embedding = embedding.to(torch.device("cpu"))
            wrapped = OffloadedEmbedding(embedding, output_device)
            if hasattr(model, "set_input_embeddings"):
                model.set_input_embeddings(wrapped)
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                model.model.embed_tokens = wrapped
            replaced.append("input_embeddings")

    if offload_output and hasattr(model, "get_output_embeddings"):
        output = model.get_output_embeddings()
        if output is not None:
            output = output.to(torch.device("cpu"))
            wrapped = OffloadedLinear(output, output_device)
            if hasattr(model, "set_output_embeddings"):
                model.set_output_embeddings(wrapped)
            elif hasattr(model, "lm_head"):
                model.lm_head = wrapped
            replaced.append("output_embeddings")

    return replaced
