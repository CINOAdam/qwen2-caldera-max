from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.mixkvq_mojo import _unpack_bits

if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e5m2"):
    torch.float8_e8m0fnu = torch.float8_e5m2  # type: ignore[attr-defined]

try:
    from max.torch import CustomOpLibrary
except Exception:  # pragma: no cover - optional dependency
    CustomOpLibrary = None


_LIB_PATH = Path(__file__).resolve().parents[1] / "kernels" / "mixkvq_max_op"


@dataclass
class _MaxOpCache:
    epoch: int
    key_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    head_dim: int
    device: torch.device
    k_q: torch.Tensor
    v_q: torch.Tensor
    k_scales: torch.Tensor
    v_scales: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor


_OP_CACHE: dict[Path, Any] = {}


def _load_op() -> Any:
    if CustomOpLibrary is None:
        raise RuntimeError("max.torch is not available; install MAX to use the custom op.")
    op = _OP_CACHE.get(_LIB_PATH)
    if op is None:
        lib = CustomOpLibrary(_LIB_PATH)
        op = lib.mixkvq_streaming_fused
        _OP_CACHE[_LIB_PATH] = op
    return op


def _chunk_to_uint8(values: torch.Tensor, *, bits: int, packed: bool, orig_dim: int) -> torch.Tensor:
    if packed:
        values = _unpack_bits(values, bits, orig_dim)
    return values.to(torch.uint8)


def _prepare_segments_int4_torch(
    key_segments: Sequence,
    value_segments: Sequence,
    *,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not key_segments or not value_segments:
        raise ValueError("No segments to prepare.")
    if len(key_segments) != len(value_segments):
        raise ValueError("Key/value segment length mismatch.")

    offsets = []
    lengths = []
    total = 0
    k_chunks = []
    k_scales = []
    v_chunks = []
    v_scales = []

    for key_seg, val_seg in zip(key_segments, value_segments):
        if key_seg.int4 is None or key_seg.int2 is not None or key_seg.bf16 is not None:
            raise ValueError("MAX op int4 path requires all key dims in int4.")
        if key_seg.int4.bits != 4:
            raise ValueError("MAX op int4 path requires 4-bit keys.")
        if val_seg.chunk.bits != 4:
            raise ValueError("MAX op int4 path requires 4-bit values.")
        if key_seg.int4_idx.numel() != head_dim:
            raise ValueError("MAX op int4 path requires full-dim int4 keys.")

        offsets.append(total)
        lengths.append(key_seg.seq_len)
        total += key_seg.seq_len

        k_vals = _chunk_to_uint8(
            key_seg.int4.values,
            bits=key_seg.int4.bits,
            packed=key_seg.int4.packed,
            orig_dim=key_seg.int4.orig_dim,
        )
        if not torch.equal(key_seg.int4_idx, torch.arange(head_dim, device=key_seg.int4_idx.device)):
            k_full = torch.empty((*k_vals.shape[:-1], head_dim), dtype=k_vals.dtype, device=k_vals.device)
            k_full.index_copy_(-1, key_seg.int4_idx, k_vals)
            k_vals = k_full
            scale_full = torch.empty_like(key_seg.int4.scales)
            scale_full.index_copy_(-1, key_seg.int4_idx, key_seg.int4.scales)
            k_scales.append(scale_full)
        else:
            k_scales.append(key_seg.int4.scales)
        v_vals = _chunk_to_uint8(
            val_seg.chunk.values,
            bits=val_seg.chunk.bits,
            packed=val_seg.chunk.packed,
            orig_dim=val_seg.chunk.orig_dim,
        )
        k_chunks.append(k_vals)
        v_chunks.append(v_vals)
        v_scales.append(val_seg.chunk.scales)

    k_cat = torch.cat(k_chunks, dim=-2).to(device=device, dtype=torch.int32)
    v_cat = torch.cat(v_chunks, dim=-2).to(device=device, dtype=torch.int32)
    v_scales_cat = torch.cat(v_scales, dim=-2).squeeze(-1).to(device=device, dtype=torch.float32)
    k_scales_cat = torch.stack(k_scales, dim=1).to(device=device, dtype=torch.float32)

    offsets_t = torch.tensor(offsets, device=device, dtype=torch.int32)
    lengths_t = torch.tensor(lengths, device=device, dtype=torch.int32)

    return k_cat, v_cat, k_scales_cat, v_scales_cat, offsets_t, lengths_t


def _quantize_k_per_chunk_torch(
    k: torch.Tensor,
    *,
    qmax: int,
    zero: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = k.abs().amax(dim=-2)
    scale = torch.clamp(max_abs / qmax, min=eps)
    q = torch.round(k / scale.unsqueeze(-2)).clamp(-qmax - 1, qmax)
    return (q + zero).to(torch.int32), scale.to(torch.float32)


def _quantize_v_per_token_torch(
    v: torch.Tensor,
    *,
    qmax: int,
    zero: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = v.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=eps)
    q = torch.round(v / scale).clamp(-qmax - 1, qmax)
    v_q = (q + zero).to(torch.int32)
    return v_q, scale.squeeze(-1).to(torch.float32)


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    q_len: int,
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.zeros((batch, q_len, total_len), device=device, dtype=torch.float32)
    mask = attention_mask.to(device=device, dtype=torch.float32)
    if mask.dim() == 4:
        if mask.shape[1] != 1:
            raise ValueError("MAX op expects attention_mask with a singleton head dimension.")
        mask = mask[:, 0]
    elif mask.dim() != 3:
        raise ValueError("MAX op expects attention_mask with shape [batch, 1, q_len, kv_len].")
    if mask.shape[0] != batch or mask.shape[-2] != q_len or mask.shape[-1] != total_len:
        raise ValueError("MAX op attention_mask shape mismatch.")
    return mask.contiguous()


def _get_cached_segments(layer: Any, *, head_dim: int, device: torch.device):
    key_segments = layer.key_segments
    value_segments = layer.value_segments
    key_ids = tuple(id(seg) for seg in key_segments)
    val_ids = tuple(id(seg) for seg in value_segments)
    cache = layer._max_op_cache
    if (
        isinstance(cache, _MaxOpCache)
        and cache.epoch == layer._mojo_cache_epoch
        and cache.head_dim == head_dim
        and cache.device == device
        and cache.key_ids == key_ids
        and cache.val_ids == val_ids
    ):
        return cache

    k_q, v_q, k_scales, v_scales, offsets, lengths = _prepare_segments_int4_torch(
        key_segments,
        value_segments,
        head_dim=head_dim,
        device=device,
    )
    cache = _MaxOpCache(
        epoch=layer._mojo_cache_epoch,
        key_ids=key_ids,
        val_ids=val_ids,
        head_dim=head_dim,
        device=device,
        k_q=k_q,
        v_q=v_q,
        k_scales=k_scales,
        v_scales=v_scales,
        offsets=offsets,
        lengths=lengths,
    )
    layer._max_op_cache = cache
    return cache


def streaming_attention_max_from_layer(
    query_states: torch.Tensor,
    *,
    layer: Any,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    device = query_states.device
    q = query_states.detach().to(torch.float32)
    batch, heads, q_len, head_dim = q.shape
    expected_scaling = 1.0 / (head_dim ** 0.5)
    if abs(scaling - expected_scaling) > 1.0e-6:
        q = q * (scaling / expected_scaling)

    if layer.key_segments:
        cache = _get_cached_segments(layer, head_dim=head_dim, device=device)
        k_q = cache.k_q
        v_q = cache.v_q
        k_scales = cache.k_scales
        v_scales = cache.v_scales
        offsets = cache.offsets
        lengths = cache.lengths
    else:
        k_q = v_q = k_scales = v_scales = offsets = lengths = None

    if layer.fp_buffer_k is not None and layer.fp_buffer_v is not None:
        buf_k = layer.fp_buffer_k.detach().to(torch.float32)
        buf_v = layer.fp_buffer_v.detach().to(torch.float32)
        if buf_k.shape[-2] != buf_v.shape[-2]:
            raise ValueError("MixKVQ buffer key/value length mismatch.")
        qmax = 7
        zero = 8
        buf_kq, buf_k_scales = _quantize_k_per_chunk_torch(buf_k, qmax=qmax, zero=zero)
        buf_vq, buf_v_scales = _quantize_v_per_token_torch(buf_v, qmax=qmax, zero=zero)
        buf_offsets = torch.tensor([0], device=device, dtype=torch.int32)
        buf_lengths = torch.tensor([buf_k.shape[-2]], device=device, dtype=torch.int32)

        if k_q is None:
            k_q = buf_kq
            v_q = buf_vq
            k_scales = buf_k_scales.unsqueeze(1)
            v_scales = buf_v_scales
            offsets = buf_offsets
            lengths = buf_lengths
        else:
            total_len = k_q.shape[-2]
            k_q = torch.cat([k_q, buf_kq], dim=-2)
            v_q = torch.cat([v_q, buf_vq], dim=-2)
            k_scales = torch.cat([k_scales, buf_k_scales.unsqueeze(1)], dim=1)
            v_scales = torch.cat([v_scales, buf_v_scales], dim=-1)
            offsets = torch.cat([offsets, torch.tensor([total_len], device=device, dtype=torch.int32)])
            lengths = torch.cat([lengths, buf_lengths])

    if k_q is None or v_q is None:
        raise ValueError("MAX op requires quantized segments or a full-precision buffer.")

    kv_heads = k_q.shape[1]
    if heads % num_key_value_groups != 0 or kv_heads != heads // num_key_value_groups:
        raise ValueError("Head mismatch for GQA fused path.")

    total_len = k_q.shape[-2]
    mask = _prepare_attention_mask(
        attention_mask,
        batch=batch,
        q_len=q_len,
        total_len=total_len,
        device=device,
    )

    out = torch.empty((batch, heads, q_len, head_dim), device=device, dtype=torch.float32)
    op = _load_op()
    op(out, q, k_q, v_q, k_scales, v_scales, offsets, lengths, mask)

    return out.to(dtype=query_states.dtype)
