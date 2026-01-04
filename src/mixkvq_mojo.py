from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


_LIB_NAME = "libmixkvq_streaming_gpu_fused.so"


@dataclass
class _MojoInt4Cache:
    epoch: int
    key_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    head_dim: int
    k_q: np.ndarray
    v_q: np.ndarray
    k_scales: np.ndarray
    v_scales: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray


def _lib_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "artifacts" / "mojo" / _LIB_NAME


def _build_lib() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_mojo_kernels.sh"
    subprocess.run([str(script)], check=True)


def _load_lib() -> ctypes.CDLL:
    lib_path = _lib_path()
    if not lib_path.exists():
        _build_lib()
    lib = ctypes.CDLL(str(lib_path))
    func = lib.mixkvq_streaming_fused_host
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    func.restype = ctypes.c_int64
    func_fp16 = lib.mixkvq_streaming_fused_host_fp16
    func_fp16.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    func_fp16.restype = ctypes.c_int64
    return lib


def _as_int32_array(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


def _unpack_2bit(values: torch.Tensor, orig_dim: int) -> torch.Tensor:
    total_dim = values.shape[-1] * 4
    out = torch.empty((*values.shape[:-1], total_dim), dtype=torch.uint8, device=values.device)
    out[..., 0::4] = values & 0x3
    out[..., 1::4] = (values >> 2) & 0x3
    out[..., 2::4] = (values >> 4) & 0x3
    out[..., 3::4] = (values >> 6) & 0x3
    return out[..., :orig_dim]


def _unpack_4bit(values: torch.Tensor, orig_dim: int) -> torch.Tensor:
    total_dim = values.shape[-1] * 2
    out = torch.empty((*values.shape[:-1], total_dim), dtype=torch.uint8, device=values.device)
    out[..., 0::2] = values & 0xF
    out[..., 1::2] = (values >> 4) & 0xF
    return out[..., :orig_dim]


def _unpack_bits(values: torch.Tensor, bits: int, orig_dim: int) -> torch.Tensor:
    if bits == 2:
        return _unpack_2bit(values, orig_dim)
    if bits == 4:
        return _unpack_4bit(values, orig_dim)
    raise ValueError(f"Unsupported pack bits: {bits}")


def _chunk_to_uint8(values: torch.Tensor, *, bits: int, packed: bool, orig_dim: int) -> torch.Tensor:
    if packed:
        values = _unpack_bits(values, bits, orig_dim)
    return values.to(torch.uint8)


def _quantize_k_per_chunk(
    k: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    zero: int,
    qmax: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    batch, kv_heads, total_len, head_dim = k.shape
    k_q = np.empty_like(k, dtype=np.int32)
    k_scales = np.empty((batch, len(offsets), kv_heads, head_dim), dtype=np.float32)
    for c, (offset, length) in enumerate(zip(offsets, lengths)):
        sl = slice(int(offset), int(offset + length))
        chunk = k[:, :, sl, :]
        max_abs = np.max(np.abs(chunk), axis=2)
        scale = np.maximum(max_abs / qmax, eps)
        k_scales[:, c] = scale
        q = np.round(chunk / scale[:, :, None, :]).clip(-qmax - 1, qmax)
        k_q[:, :, sl, :] = q.astype(np.int32) + zero
    return k_q, k_scales


def _quantize_v_per_token(
    v: np.ndarray,
    *,
    zero: int,
    qmax: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(v), axis=-1, keepdims=True)
    scale = np.maximum(max_abs / qmax, eps)
    q = np.round(v / scale).clip(-qmax - 1, qmax)
    v_q = q.astype(np.int32) + zero
    v_scales = scale[..., 0].astype(np.float32)
    return v_q, v_scales


def _concat_chunks(chunks: list[tuple[torch.Tensor, torch.Tensor]]):
    offsets = []
    lengths = []
    total = 0
    ks = []
    vs = []
    for k, v in chunks:
        seq = k.shape[-2]
        offsets.append(total)
        lengths.append(seq)
        total += seq
        ks.append(k)
        vs.append(v)
    k_cat = torch.cat(ks, dim=-2)
    v_cat = torch.cat(vs, dim=-2)
    return k_cat, v_cat, offsets, lengths


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    q_len: int,
    total_len: int,
) -> np.ndarray | None:
    if attention_mask is None:
        return None
    mask = attention_mask.detach().to(torch.float32).cpu()
    if mask.dim() == 4:
        if mask.shape[1] != 1:
            raise ValueError("Mojo fused path expects attention_mask with a singleton head dimension.")
        mask = mask[:, 0]
    elif mask.dim() != 3:
        raise ValueError("Mojo fused path expects attention_mask with shape [batch, 1, q_len, kv_len].")
    if mask.shape[0] != batch or mask.shape[-2] != q_len or mask.shape[-1] != total_len:
        raise ValueError("Mojo fused path attention_mask shape mismatch.")
    return mask.contiguous().view(batch, q_len, total_len).numpy()


def _prepare_segments_int4(
    key_segments: Sequence,
    value_segments: Sequence,
    *,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            raise ValueError("Mojo fused int4 path requires all key dims in int4.")
        if key_seg.int4.bits != 4:
            raise ValueError("Mojo fused int4 path requires 4-bit keys.")
        if val_seg.chunk.bits != 4:
            raise ValueError("Mojo fused int4 path requires 4-bit values.")
        if key_seg.int4_idx.numel() != head_dim:
            raise ValueError("Mojo fused int4 path requires full-dim int4 keys.")

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

    k_cat = torch.cat(k_chunks, dim=-2).to(torch.int32)
    v_cat = torch.cat(v_chunks, dim=-2).to(torch.int32)
    v_scales_cat = torch.cat(v_scales, dim=-2)

    k_scales_cat = torch.stack(k_scales, dim=1)
    v_scales_cat = v_scales_cat.squeeze(-1)

    return (
        k_cat.cpu().numpy(),
        v_cat.cpu().numpy(),
        k_scales_cat.cpu().numpy(),
        v_scales_cat.cpu().numpy(),
        _as_int32_array(offsets),
        _as_int32_array(lengths),
    )


def _prepare_segments_int4_cached(layer: Any, *, head_dim: int):
    key_segments = layer.key_segments
    value_segments = layer.value_segments
    key_ids = tuple(id(seg) for seg in key_segments)
    val_ids = tuple(id(seg) for seg in value_segments)
    cache = layer._mojo_int4_cache
    if (
        isinstance(cache, _MojoInt4Cache)
        and cache.epoch == layer._mojo_cache_epoch
        and cache.head_dim == head_dim
        and cache.key_ids == key_ids
        and cache.val_ids == val_ids
    ):
        return cache.k_q, cache.v_q, cache.k_scales, cache.v_scales, cache.offsets, cache.lengths

    k_q, v_q, k_scales, v_scales, offsets, lengths = _prepare_segments_int4(
        key_segments,
        value_segments,
        head_dim=head_dim,
    )
    layer._mojo_int4_cache = _MojoInt4Cache(
        epoch=layer._mojo_cache_epoch,
        key_ids=key_ids,
        val_ids=val_ids,
        head_dim=head_dim,
        k_q=k_q,
        v_q=v_q,
        k_scales=k_scales,
        v_scales=v_scales,
        offsets=offsets,
        lengths=lengths,
    )
    return k_q, v_q, k_scales, v_scales, offsets, lengths


def streaming_attention_fused(
    query_states: torch.Tensor,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_key_value_groups: int,
    scaling: float,
    zero: int = 8,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused Mojo kernel, using GPU quantization when available (prototype)."""
    if not chunks:
        raise ValueError("No KV chunks provided.")

    device = query_states.device
    q = query_states.detach().to(torch.float32).cpu()
    k_cat, v_cat, offsets, lengths = _concat_chunks(
        [(k.detach().to(torch.float32).cpu(), v.detach().to(torch.float32).cpu()) for k, v in chunks]
    )

    batch, heads, q_len, head_dim = q.shape
    kv_heads = k_cat.shape[1]
    if heads % num_key_value_groups != 0 or kv_heads != heads // num_key_value_groups:
        raise ValueError("Head mismatch for GQA fused path.")

    offsets_np = _as_int32_array(offsets)
    lengths_np = _as_int32_array(lengths)

    q_np = q.numpy()
    k_np = k_cat.numpy()
    v_np = v_cat.numpy()

    out = np.zeros_like(q_np, dtype=np.float32)

    mask_np = _prepare_attention_mask(attention_mask, batch=batch, q_len=q_len, total_len=k_cat.shape[-2])
    use_mask = 0
    if mask_np is None:
        mask_np = np.zeros(1, dtype=np.float32)
    else:
        use_mask = 1

    lib = _load_lib()
    rc = lib.mixkvq_streaming_fused_host_fp16(
        q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        offsets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lengths_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(len(offsets_np)),
        ctypes.c_int64(k_cat.shape[-2]),
        ctypes.c_int64(batch),
        ctypes.c_int64(heads),
        ctypes.c_int64(q_len),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(num_key_value_groups),
        ctypes.c_float(scaling),
        ctypes.c_int64(zero),
        mask_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(use_mask),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        qmax = 7
        k_q, k_scales = _quantize_k_per_chunk(k_np, offsets_np, lengths_np, zero=zero, qmax=qmax)
        v_q, v_scales = _quantize_v_per_token(v_np, zero=zero, qmax=qmax)
        rc = lib.mixkvq_streaming_fused_host(
            q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            v_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            k_scales.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_scales.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            offsets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lengths_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(len(offsets_np)),
            ctypes.c_int64(k_cat.shape[-2]),
            ctypes.c_int64(batch),
            ctypes.c_int64(heads),
            ctypes.c_int64(q_len),
            ctypes.c_int64(head_dim),
            ctypes.c_int64(num_key_value_groups),
            ctypes.c_float(scaling),
            ctypes.c_int64(zero),
            mask_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(use_mask),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"Mojo fused kernel failed with code {rc}.")

    return torch.from_numpy(out).to(device=device, dtype=query_states.dtype)


def streaming_attention_fused_from_layer(
    query_states: torch.Tensor,
    *,
    layer: Any,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not layer.key_segments:
        chunks = layer.iter_kv_chunks(include_buffer=True)
        return streaming_attention_fused(
            query_states,
            chunks,
            num_key_value_groups=num_key_value_groups,
            scaling=scaling,
            attention_mask=attention_mask,
        )

    device = query_states.device
    q = query_states.detach().to(torch.float32).cpu()
    batch, heads, q_len, head_dim = q.shape
    k_q, v_q, k_scales, v_scales, offsets_np, lengths_np = _prepare_segments_int4_cached(
        layer,
        head_dim=head_dim,
    )

    kv_heads = k_q.shape[1]
    if heads % num_key_value_groups != 0 or kv_heads != heads // num_key_value_groups:
        raise ValueError("Head mismatch for GQA fused path.")

    total_len = k_q.shape[-2]
    if layer.fp_buffer_k is not None and layer.fp_buffer_v is not None:
        buf_k = layer.fp_buffer_k.detach().to(torch.float32).cpu().numpy()
        buf_v = layer.fp_buffer_v.detach().to(torch.float32).cpu().numpy()
        if buf_k.shape[-2] != buf_v.shape[-2]:
            raise ValueError("MixKVQ buffer key/value length mismatch.")
        buf_offsets = _as_int32_array([0])
        buf_lengths = _as_int32_array([buf_k.shape[-2]])
        qmax = 7
        zero = 8
        buf_kq, buf_k_scales = _quantize_k_per_chunk(buf_k, buf_offsets, buf_lengths, zero=zero, qmax=qmax)
        buf_vq, buf_v_scales = _quantize_v_per_token(buf_v, zero=zero, qmax=qmax)

        k_q = np.concatenate([k_q, buf_kq], axis=-2)
        v_q = np.concatenate([v_q, buf_vq], axis=-2)
        k_scales = np.concatenate([k_scales, buf_k_scales], axis=1)
        v_scales = np.concatenate([v_scales, buf_v_scales], axis=-1)
        offsets_np = np.concatenate([offsets_np, _as_int32_array([total_len])])
        lengths_np = np.concatenate([lengths_np, _as_int32_array([buf_k.shape[-2]])])
        total_len = k_q.shape[-2]
    else:
        zero = 8

    out = np.zeros_like(q.numpy(), dtype=np.float32)

    mask_np = _prepare_attention_mask(attention_mask, batch=batch, q_len=q_len, total_len=total_len)
    use_mask = 0
    if mask_np is None:
        mask_np = np.zeros(1, dtype=np.float32)
    else:
        use_mask = 1

    lib = _load_lib()
    rc = lib.mixkvq_streaming_fused_host(
        q.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        v_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        k_scales.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_scales.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        offsets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lengths_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(len(offsets_np)),
        ctypes.c_int64(total_len),
        ctypes.c_int64(batch),
        ctypes.c_int64(heads),
        ctypes.c_int64(q_len),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(num_key_value_groups),
        ctypes.c_float(scaling),
        ctypes.c_int64(zero),
        mask_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(use_mask),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"Mojo fused kernel failed with code {rc}.")

    return torch.from_numpy(out).to(device=device, dtype=query_states.dtype)


def streaming_attention_fused_from_segments(
    query_states: torch.Tensor,
    *,
    layer: Any | None = None,
    key_segments: Sequence,
    value_segments: Sequence,
    head_dim: int,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not key_segments:
        raise ValueError("No key segments.")

    device = query_states.device
    q = query_states.detach().to(torch.float32).cpu()
    batch, heads, q_len, _ = q.shape

    if layer is not None:
        k_q, v_q, k_scales, v_scales, offsets_np, lengths_np = _prepare_segments_int4_cached(
            layer,
            head_dim=head_dim,
        )
    else:
        k_q, v_q, k_scales, v_scales, offsets_np, lengths_np = _prepare_segments_int4(
            key_segments,
            value_segments,
            head_dim=head_dim,
        )

    kv_heads = k_q.shape[1]
    if heads % num_key_value_groups != 0 or kv_heads != heads // num_key_value_groups:
        raise ValueError("Head mismatch for GQA fused path.")

    out = np.zeros_like(q.numpy(), dtype=np.float32)
    zero = 8

    mask_np = _prepare_attention_mask(attention_mask, batch=batch, q_len=q_len, total_len=k_q.shape[-2])
    use_mask = 0
    if mask_np is None:
        mask_np = np.zeros(1, dtype=np.float32)
    else:
        use_mask = 1

    lib = _load_lib()
    func = lib.mixkvq_streaming_fused_host
    rc = func(
        q.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        v_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        k_scales.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_scales.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        offsets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lengths_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(len(offsets_np)),
        ctypes.c_int64(k_q.shape[-2]),
        ctypes.c_int64(batch),
        ctypes.c_int64(heads),
        ctypes.c_int64(q_len),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(num_key_value_groups),
        ctypes.c_float(scaling),
        ctypes.c_int64(zero),
        mask_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(use_mask),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"Mojo fused kernel failed with code {rc}.")

    return torch.from_numpy(out).to(device=device, dtype=query_states.dtype)
