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
        chunk_size: int | None = 1024,
    ) -> None:
        super().__init__()
        self.in_features = q_weight.values.shape[1]
        self.out_features = q_weight.values.shape[0]
        self.group_size = q_weight.group_size
        self.cache_dequant = cache_dequant
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

    def _maybe_dequant(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.cache_dequant
            and self._q_deq is not None
            and self._q_deq.dtype == dtype
            and self._q_deq.device == device
        ):
            return self._q_deq, self._l_deq, self._r_deq  # type: ignore[return-value]

        q_vals = self._unpack_values(self.q_values, self.q_pack_bits, self.q_packed_cols)
        l_vals = self._unpack_values(self.l_values, self.l_pack_bits, self.l_packed_cols)
        r_vals = self._unpack_values(self.r_values, self.r_pack_bits, self.r_packed_cols)
        q = dequantize_groupwise(
            QuantizedTensor(q_vals, self.q_scales, self.group_size, 0)
        )
        l = dequantize_groupwise(
            QuantizedTensor(l_vals, self.l_scales, self.group_size, 0)
        )
        r = dequantize_groupwise(
            QuantizedTensor(r_vals, self.r_scales, self.group_size, 0)
        )

        if q.dtype != dtype or q.device != device:
            q = q.to(device=device, dtype=dtype)
            l = l.to(device=device, dtype=dtype)
            r = r.to(device=device, dtype=dtype)

        if self.cache_dequant:
            self._q_deq = q
            self._l_deq = l
            self._r_deq = r

        return q, l, r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        device = x.device
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])

        r_vals = self._unpack_values(self.r_values, self.r_pack_bits, self.r_packed_cols)
        r = dequantize_groupwise(
            QuantizedTensor(r_vals, self.r_scales, self.group_size, 0)
        ).to(device=device, dtype=dtype)
        xr = x @ r.t()

        if self.chunk_size is None or self.chunk_size >= self.out_features:
            q, l, _ = self._maybe_dequant(dtype, device)
            base = x @ q.t()
            low_rank = xr @ l.t()
            out = base + low_rank
        else:
            outputs = []
            for start in range(0, self.out_features, self.chunk_size):
                end = min(start + self.chunk_size, self.out_features)
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
        chunk_size=chunk_size,
    )


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def apply_caldera(
    model: nn.Module,
    artifacts_dir: Path,
    *,
    cache_dequant: bool = False,
    chunk_size: int | None = 1024,
    device: torch.device | None = None,
    strict: bool = False,
) -> list[str]:
    layers_dir = artifacts_dir / "layers"
    replaced: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        artifact = layers_dir / f"{name.replace('.', '_')}.pt"
        if not artifact.exists():
            continue
        bias = module.bias.detach().clone() if module.bias is not None else None
        caldera_layer = load_caldera_layer(
            artifact,
            bias=bias,
            cache_dequant=cache_dequant,
            chunk_size=chunk_size,
        )
        target_device = device
        if target_device is None:
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
        caldera_layer = caldera_layer.to(target_device)
        _set_module(model, name, caldera_layer)
        replaced.append(name)

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
