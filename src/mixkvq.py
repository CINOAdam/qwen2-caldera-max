from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


@dataclass(frozen=True)
class MixKVQConfig:
    ratio_bf16: float = 0.1
    ratio_int4: float = 0.2
    key_bits_low: int = 2
    key_bits_mid: int = 4
    value_bits: int = 2
    update_interval: int = 32
    buffer_size: int | None = None
    cache_dequant: bool = False
    pack_bits: bool = True
    use_query: bool = True
    use_key_scale: bool = False
    use_mojo_fused: bool = False
    use_max_op: bool = False


@dataclass
class QuantizedChunk:
    values: torch.Tensor
    scales: torch.Tensor
    bits: int
    orig_dim: int
    packed: bool


@dataclass
class MixKVQKeySegment:
    seq_len: int
    bf16_idx: torch.Tensor
    int4_idx: torch.Tensor
    int2_idx: torch.Tensor
    bf16: Optional[torch.Tensor] = None
    int4: Optional[QuantizedChunk] = None
    int2: Optional[QuantizedChunk] = None
    dequant: Optional[torch.Tensor] = None
    dequant_dtype: Optional[torch.dtype] = None
    dequant_device: Optional[torch.device] = None


@dataclass
class MixKVQValueSegment:
    seq_len: int
    chunk: QuantizedChunk
    dequant: Optional[torch.Tensor] = None
    dequant_dtype: Optional[torch.dtype] = None
    dequant_device: Optional[torch.device] = None


def _quant_params(bits: int) -> tuple[int, int, int]:
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    zero = -qmin
    return qmin, qmax, zero


def _pad_for_pack(values: torch.Tensor, pack_factor: int) -> tuple[torch.Tensor, int]:
    last_dim = values.shape[-1]
    remainder = last_dim % pack_factor
    if remainder == 0:
        return values, 0
    pad = pack_factor - remainder
    pad_shape = list(values.shape)
    pad_shape[-1] = pad
    pad_tensor = torch.zeros(pad_shape, dtype=values.dtype, device=values.device)
    return torch.cat([values, pad_tensor], dim=-1), pad


def _pack_2bit(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    orig_dim = values.shape[-1]
    values, _ = _pad_for_pack(values, 4)
    reshaped = values.reshape(*values.shape[:-1], -1, 4)
    packed = (
        reshaped[..., 0]
        | (reshaped[..., 1] << 2)
        | (reshaped[..., 2] << 4)
        | (reshaped[..., 3] << 6)
    )
    return packed, orig_dim


def _unpack_2bit(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    total_dim = packed.shape[-1] * 4
    out = torch.empty((*packed.shape[:-1], total_dim), dtype=torch.uint8, device=packed.device)
    out[..., 0::4] = packed & 0x3
    out[..., 1::4] = (packed >> 2) & 0x3
    out[..., 2::4] = (packed >> 4) & 0x3
    out[..., 3::4] = (packed >> 6) & 0x3
    return out[..., :orig_dim]


def _pack_4bit(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    orig_dim = values.shape[-1]
    values, _ = _pad_for_pack(values, 2)
    reshaped = values.reshape(*values.shape[:-1], -1, 2)
    packed = reshaped[..., 0] | (reshaped[..., 1] << 4)
    return packed, orig_dim


def _unpack_4bit(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    total_dim = packed.shape[-1] * 2
    out = torch.empty((*packed.shape[:-1], total_dim), dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = packed & 0xF
    out[..., 1::2] = (packed >> 4) & 0xF
    return out[..., :orig_dim]


def _pack_bits(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    if bits == 2:
        return _pack_2bit(values)
    if bits == 4:
        return _pack_4bit(values)
    raise ValueError(f"Unsupported pack bits: {bits}")


def _unpack_bits(values: torch.Tensor, bits: int, orig_dim: int) -> torch.Tensor:
    if bits == 2:
        return _unpack_2bit(values, orig_dim)
    if bits == 4:
        return _unpack_4bit(values, orig_dim)
    raise ValueError(f"Unsupported pack bits: {bits}")


def _quantize_per_dim(
    x: torch.Tensor,
    *,
    bits: int,
    pack_bits: bool,
    eps: float = 1e-6,
) -> QuantizedChunk:
    qmin, qmax, zero = _quant_params(bits)
    x_float = x.float()
    max_val = x_float.abs().amax(dim=-2)
    scale = torch.clamp(max_val / qmax, min=eps)
    scale_expanded = scale.unsqueeze(-2)
    q = torch.round(x_float / scale_expanded).clamp(qmin, qmax)
    q_u = (q + zero).to(torch.uint8)
    if pack_bits and bits in (2, 4):
        packed, orig_dim = _pack_bits(q_u, bits)
        return QuantizedChunk(packed, scale, bits, orig_dim, packed=True)
    return QuantizedChunk(q_u, scale, bits, q_u.shape[-1], packed=False)


def _quantize_per_token(
    x: torch.Tensor,
    *,
    bits: int,
    pack_bits: bool,
    eps: float = 1e-6,
) -> QuantizedChunk:
    qmin, qmax, zero = _quant_params(bits)
    x_float = x.float()
    max_val = x_float.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_val / qmax, min=eps)
    q = torch.round(x_float / scale).clamp(qmin, qmax)
    q_u = (q + zero).to(torch.uint8)
    if pack_bits and bits in (2, 4):
        packed, orig_dim = _pack_bits(q_u, bits)
        return QuantizedChunk(packed, scale, bits, orig_dim, packed=True)
    return QuantizedChunk(q_u, scale, bits, q_u.shape[-1], packed=False)


def _dequantize(chunk: QuantizedChunk, *, dtype: torch.dtype) -> torch.Tensor:
    qmin, _, zero = _quant_params(chunk.bits)
    if chunk.packed:
        q_u = _unpack_bits(chunk.values, chunk.bits, chunk.orig_dim)
    else:
        q_u = chunk.values
    q = q_u.to(torch.int16) - zero
    if chunk.scales.dim() == q.dim():
        scales = chunk.scales
    else:
        scales = chunk.scales.unsqueeze(-2)
    out = q.to(scales.dtype) * scales
    return out.to(dtype=dtype)


def _split_indices(
    head_dim: int, ratio_bf16: float, ratio_int4: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bf16_count = max(0, int(round(head_dim * ratio_bf16)))
    int4_count = max(0, int(round(head_dim * ratio_int4)))
    if bf16_count + int4_count > head_dim:
        int4_count = max(0, head_dim - bf16_count)
    remaining = max(0, head_dim - bf16_count - int4_count)
    idx = torch.arange(head_dim, device=device, dtype=torch.long)
    bf16_idx = idx[:bf16_count]
    int4_idx = idx[bf16_count : bf16_count + int4_count]
    int2_idx = idx[bf16_count + int4_count : bf16_count + int4_count + remaining]
    return bf16_idx, int4_idx, int2_idx


class MixKVQDynamicLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, config: MixKVQConfig):
        super().__init__()
        self.config = config
        self.key_segments: list[MixKVQKeySegment] = []
        self.value_segments: list[MixKVQValueSegment] = []
        self.seq_len = 0
        self.cumulative_length = 0
        self._salience_accum: Optional[torch.Tensor] = None
        self._salience_steps = 0
        self._bf16_idx: Optional[torch.Tensor] = None
        self._int4_idx: Optional[torch.Tensor] = None
        self._int2_idx: Optional[torch.Tensor] = None
        self.fp_buffer_k: Optional[torch.Tensor] = None
        self.fp_buffer_v: Optional[torch.Tensor] = None
        self.head_dim: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None
        self._mojo_cache_epoch = 0
        self._mojo_int4_cache: Optional[dict[str, Any]] = None
        self._max_op_cache: Optional[dict[str, Any]] = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.head_dim = key_states.shape[-1]
        self._bf16_idx, self._int4_idx, self._int2_idx = _split_indices(
            self.head_dim, self.config.ratio_bf16, self.config.ratio_int4, key_states.device
        )
        self.is_initialized = True

    def _invalidate_mojo_cache(self) -> None:
        self._mojo_cache_epoch += 1
        self._mojo_int4_cache = None
        self._max_op_cache = None

    def _current_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._bf16_idx is None or self._int4_idx is None or self._int2_idx is None:
            raise RuntimeError("MixKVQ indices are not initialized.")
        return self._bf16_idx, self._int4_idx, self._int2_idx

    def _buffer_target(self) -> int:
        if self.config.buffer_size is not None:
            return max(1, int(self.config.buffer_size))
        return max(1, int(self.config.update_interval))

    def _maybe_update_salience(self, cache_kwargs: Optional[dict[str, Any]], key_states: torch.Tensor) -> None:
        if not self.config.use_query and not self.config.use_key_scale:
            return
        if cache_kwargs is None:
            cache_kwargs = {}
        query_states = cache_kwargs.get("query_states")
        if query_states is None:
            if not self.config.use_query and not self.config.use_key_scale:
                return
            query_states = key_states
        elif not self.config.use_query:
            query_states = key_states
        query_abs = query_states.abs().mean(dim=(0, 1, 2))
        if self.config.use_key_scale:
            key_scale = key_states.abs().mean(dim=(0, 1, 2))
            query_abs = query_abs * key_scale

        if self._salience_accum is None:
            self._salience_accum = query_abs
        else:
            self._salience_accum = self._salience_accum + query_abs
        self._salience_steps += 1

        if self._salience_steps < self.config.update_interval:
            return
        salience = self._salience_accum / float(self._salience_steps)
        self._salience_accum = None
        self._salience_steps = 0

        head_dim = salience.shape[0]
        bf16_count = max(0, int(round(head_dim * self.config.ratio_bf16)))
        int4_count = max(0, int(round(head_dim * self.config.ratio_int4)))
        if bf16_count + int4_count > head_dim:
            int4_count = max(0, head_dim - bf16_count)
        sorted_idx = torch.argsort(salience, descending=True)
        bf16_idx = sorted_idx[:bf16_count]
        int4_idx = sorted_idx[bf16_count : bf16_count + int4_count]
        int2_idx = sorted_idx[bf16_count + int4_count :]
        self._bf16_idx = bf16_idx
        self._int4_idx = int4_idx
        self._int2_idx = int2_idx

    def _quantize_keys(self, key_states: torch.Tensor) -> MixKVQKeySegment:
        bf16_idx, int4_idx, int2_idx = self._current_indices()
        bf16 = None
        int4 = None
        int2 = None
        if bf16_idx.numel():
            bf16 = key_states.index_select(-1, bf16_idx)
        if int4_idx.numel():
            int4 = _quantize_per_dim(
                key_states.index_select(-1, int4_idx),
                bits=self.config.key_bits_mid,
                pack_bits=self.config.pack_bits,
            )
        if int2_idx.numel():
            int2 = _quantize_per_dim(
                key_states.index_select(-1, int2_idx),
                bits=self.config.key_bits_low,
                pack_bits=self.config.pack_bits,
            )
        return MixKVQKeySegment(
            seq_len=key_states.shape[-2],
            bf16_idx=bf16_idx,
            int4_idx=int4_idx,
            int2_idx=int2_idx,
            bf16=bf16,
            int4=int4,
            int2=int2,
        )

    def _quantize_values(self, value_states: torch.Tensor) -> MixKVQValueSegment:
        chunk = _quantize_per_token(
            value_states,
            bits=self.config.value_bits,
            pack_bits=self.config.pack_bits,
        )
        return MixKVQValueSegment(seq_len=value_states.shape[-2], chunk=chunk)

    def _dequantize_key_segment(self, segment: MixKVQKeySegment) -> torch.Tensor:
        if self.head_dim is None or self.dtype is None:
            raise RuntimeError("MixKVQ layer not initialized.")
        if (
            self.config.cache_dequant
            and segment.dequant is not None
            and segment.dequant_dtype == self.dtype
            and segment.dequant_device == segment.dequant.device
        ):
            return segment.dequant
        if segment.bf16 is not None:
            base_shape = segment.bf16.shape[:-1]
            device = segment.bf16.device
        elif segment.int4 is not None:
            base_shape = segment.int4.values.shape[:-1]
            device = segment.int4.values.device
        elif segment.int2 is not None:
            base_shape = segment.int2.values.shape[:-1]
            device = segment.int2.values.device
        else:
            raise RuntimeError("MixKVQ segment has no stored key data.")

        out = torch.empty((*base_shape, self.head_dim), dtype=self.dtype, device=device)
        if segment.bf16 is not None and segment.bf16_idx.numel():
            out.index_copy_(-1, segment.bf16_idx, segment.bf16)
        if segment.int4 is not None and segment.int4_idx.numel():
            deq = _dequantize(segment.int4, dtype=self.dtype)
            out.index_copy_(-1, segment.int4_idx, deq)
        if segment.int2 is not None and segment.int2_idx.numel():
            deq = _dequantize(segment.int2, dtype=self.dtype)
            out.index_copy_(-1, segment.int2_idx, deq)
        if self.config.cache_dequant:
            segment.dequant = out
            segment.dequant_dtype = out.dtype
            segment.dequant_device = out.device
        return out

    def _dequantize_value_segment(self, segment: MixKVQValueSegment) -> torch.Tensor:
        if self.dtype is None:
            raise RuntimeError("MixKVQ layer not initialized.")
        if (
            self.config.cache_dequant
            and segment.dequant is not None
            and segment.dequant_dtype == self.dtype
            and segment.dequant_device == segment.dequant.device
        ):
            return segment.dequant
        out = _dequantize(segment.chunk, dtype=self.dtype)
        if self.config.cache_dequant:
            segment.dequant = out
            segment.dequant_dtype = out.dtype
            segment.dequant_device = out.device
        return out

    def iter_kv_chunks(self, *, include_buffer: bool = True) -> list[tuple[torch.Tensor, torch.Tensor]]:
        chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key_seg, value_seg in zip(self.key_segments, self.value_segments):
            k = self._dequantize_key_segment(key_seg)
            v = self._dequantize_value_segment(value_seg)
            chunks.append((k, v))
        if include_buffer and self.fp_buffer_k is not None and self.fp_buffer_v is not None:
            chunks.append((self.fp_buffer_k, self.fp_buffer_v))
        return chunks

    def _materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = self.iter_kv_chunks()
        if not chunks:
            raise RuntimeError("MixKVQ cache is empty.")
        keys = [k for k, _ in chunks]
        values = [v for _, v in chunks]
        return torch.cat(keys, dim=-2), torch.cat(values, dim=-2)

    def _slice_front(self, segment: MixKVQKeySegment, count: int) -> MixKVQKeySegment:
        if segment.bf16 is not None:
            segment.bf16 = segment.bf16[:, :, count:, :]
        if segment.int4 is not None:
            segment.int4.values = segment.int4.values[:, :, count:, ...]
        if segment.int2 is not None:
            segment.int2.values = segment.int2.values[:, :, count:, ...]
        segment.dequant = None
        segment.dequant_dtype = None
        segment.dequant_device = None
        segment.seq_len -= count
        return segment

    def _slice_front_values(self, segment: MixKVQValueSegment, count: int) -> MixKVQValueSegment:
        segment.chunk.values = segment.chunk.values[:, :, count:, ...]
        segment.chunk.scales = segment.chunk.scales[:, :, count:, ...]
        segment.dequant = None
        segment.dequant_dtype = None
        segment.dequant_device = None
        segment.seq_len -= count
        return segment

    def _slice_buffer_front(self, count: int) -> None:
        if self.fp_buffer_k is None or self.fp_buffer_v is None:
            return
        if self.fp_buffer_k.shape[-2] <= count:
            self.fp_buffer_k = None
            self.fp_buffer_v = None
        else:
            self.fp_buffer_k = self.fp_buffer_k[:, :, count:, :]
            self.fp_buffer_v = self.fp_buffer_v[:, :, count:, :]

    def _drop_front(self, count: int) -> None:
        to_drop = count
        while to_drop > 0 and self.key_segments:
            segment = self.key_segments[0]
            if segment.seq_len <= to_drop:
                to_drop -= segment.seq_len
                self.seq_len -= segment.seq_len
                self.key_segments.pop(0)
                self.value_segments.pop(0)
            else:
                self.key_segments[0] = self._slice_front(segment, to_drop)
                self.value_segments[0] = self._slice_front_values(self.value_segments[0], to_drop)
                self.seq_len -= to_drop
                to_drop = 0
        if to_drop > 0:
            if self.fp_buffer_k is not None:
                drop_len = min(to_drop, self.fp_buffer_k.shape[-2])
                self._slice_buffer_front(drop_len)
                self.seq_len -= drop_len
        self._invalidate_mojo_cache()

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]]) -> None:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        self._maybe_update_salience(cache_kwargs, key_states)

        if self.fp_buffer_k is None:
            self.fp_buffer_k = key_states
            self.fp_buffer_v = value_states
        else:
            self.fp_buffer_k = torch.cat([self.fp_buffer_k, key_states], dim=-2)
            self.fp_buffer_v = torch.cat([self.fp_buffer_v, value_states], dim=-2)

        added = key_states.shape[-2]
        self.seq_len += added
        self.cumulative_length += added

        target = self._buffer_target()
        if self.fp_buffer_k is not None and self.fp_buffer_k.shape[-2] >= target:
            key_segment = self._quantize_keys(self.fp_buffer_k)
            value_segment = self._quantize_values(self.fp_buffer_v)
            self.key_segments.append(key_segment)
            self.value_segments.append(value_segment)
            self.fp_buffer_k = None
            self.fp_buffer_v = None
            self._invalidate_mojo_cache()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.append(key_states, value_states, cache_kwargs)
        return self._materialize()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0

    def get_seq_length(self) -> int:
        return self.seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self.key_segments = []
        self.value_segments = []
        self.seq_len = 0
        self.cumulative_length = 0
        self._salience_accum = None
        self._salience_steps = 0
        self.fp_buffer_k = None
        self.fp_buffer_v = None
        self.is_initialized = False
        self._invalidate_mojo_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.key_segments:
            return
        for segment in self.key_segments:
            if segment.bf16 is not None:
                segment.bf16 = segment.bf16.index_select(0, beam_idx.to(segment.bf16.device))
            if segment.int4 is not None:
                segment.int4.values = segment.int4.values.index_select(0, beam_idx.to(segment.int4.values.device))
                segment.int4.scales = segment.int4.scales.index_select(0, beam_idx.to(segment.int4.scales.device))
            if segment.int2 is not None:
                segment.int2.values = segment.int2.values.index_select(0, beam_idx.to(segment.int2.values.device))
                segment.int2.scales = segment.int2.scales.index_select(0, beam_idx.to(segment.int2.scales.device))
            segment.dequant = None
            segment.dequant_dtype = None
            segment.dequant_device = None
        for segment in self.value_segments:
            segment.chunk.values = segment.chunk.values.index_select(0, beam_idx.to(segment.chunk.values.device))
            segment.chunk.scales = segment.chunk.scales.index_select(0, beam_idx.to(segment.chunk.scales.device))
            segment.dequant = None
            segment.dequant_dtype = None
            segment.dequant_device = None
        if self.fp_buffer_k is not None and self.fp_buffer_v is not None:
            self.fp_buffer_k = self.fp_buffer_k.index_select(0, beam_idx.to(self.fp_buffer_k.device))
            self.fp_buffer_v = self.fp_buffer_v.index_select(0, beam_idx.to(self.fp_buffer_v.device))
        self._invalidate_mojo_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if not self.key_segments:
            return
        for segment in self.key_segments:
            if segment.bf16 is not None:
                segment.bf16 = segment.bf16.repeat_interleave(repeats, dim=0)
            if segment.int4 is not None:
                segment.int4.values = segment.int4.values.repeat_interleave(repeats, dim=0)
                segment.int4.scales = segment.int4.scales.repeat_interleave(repeats, dim=0)
            if segment.int2 is not None:
                segment.int2.values = segment.int2.values.repeat_interleave(repeats, dim=0)
                segment.int2.scales = segment.int2.scales.repeat_interleave(repeats, dim=0)
            segment.dequant = None
            segment.dequant_dtype = None
            segment.dequant_device = None
        for segment in self.value_segments:
            segment.chunk.values = segment.chunk.values.repeat_interleave(repeats, dim=0)
            segment.chunk.scales = segment.chunk.scales.repeat_interleave(repeats, dim=0)
            segment.dequant = None
            segment.dequant_dtype = None
            segment.dequant_device = None
        if self.fp_buffer_k is not None and self.fp_buffer_v is not None:
            self.fp_buffer_k = self.fp_buffer_k.repeat_interleave(repeats, dim=0)
            self.fp_buffer_v = self.fp_buffer_v.repeat_interleave(repeats, dim=0)
        self._invalidate_mojo_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.key_segments:
            return
        for segment in self.key_segments:
            if segment.bf16 is not None:
                segment.bf16 = segment.bf16.index_select(0, indices.to(segment.bf16.device))
            if segment.int4 is not None:
                segment.int4.values = segment.int4.values.index_select(0, indices.to(segment.int4.values.device))
                segment.int4.scales = segment.int4.scales.index_select(0, indices.to(segment.int4.scales.device))
            if segment.int2 is not None:
                segment.int2.values = segment.int2.values.index_select(0, indices.to(segment.int2.values.device))
                segment.int2.scales = segment.int2.scales.index_select(0, indices.to(segment.int2.scales.device))
            segment.dequant = None
            segment.dequant_dtype = None
            segment.dequant_device = None
        for segment in self.value_segments:
            segment.chunk.values = segment.chunk.values.index_select(0, indices.to(segment.chunk.values.device))
            segment.chunk.scales = segment.chunk.scales.index_select(0, indices.to(segment.chunk.scales.device))
            segment.dequant = None
            segment.dequant_dtype = None
            segment.dequant_device = None
        if self.fp_buffer_k is not None and self.fp_buffer_v is not None:
            self.fp_buffer_k = self.fp_buffer_k.index_select(0, indices.to(self.fp_buffer_k.device))
            self.fp_buffer_v = self.fp_buffer_v.index_select(0, indices.to(self.fp_buffer_v.device))


class MixKVQSlidingLayer(MixKVQDynamicLayer):
    is_sliding = True

    def __init__(self, config: MixKVQConfig, sliding_window: int):
        super().__init__(config)
        self.sliding_window = sliding_window

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]]) -> None:
        super().append(key_states, value_states, cache_kwargs)
        if self.seq_len > self.sliding_window - 1:
            drop_count = self.seq_len - (self.sliding_window - 1)
            self._drop_front(drop_count)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.append(key_states, value_states, cache_kwargs)
        return self._materialize()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.sliding_window


class MixKVQCache(Cache):
    def __init__(self, config, mixkvq_config: MixKVQConfig):
        decoder_config = config.get_text_config(decoder=True)
        sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
            decoder_config, "attention_chunk_size", None
        )
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if sliding_window is not None else "full_attention"
                for _ in range(decoder_config.num_hidden_layers)
            ]
        if hasattr(decoder_config, "num_kv_shared_layers"):
            layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            if layer_type in ("sliding_attention", "chunked_attention"):
                layers.append(MixKVQSlidingLayer(mixkvq_config, sliding_window=sliding_window))
            else:
                layers.append(MixKVQDynamicLayer(mixkvq_config))
        super().__init__(layers=layers, offloading=False)

    def __len__(self):
        return len(self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    @property
    def is_initialized(self) -> bool:
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        raise RuntimeError("MixKVQ does not expose legacy cache tensors.")


def _streaming_attention(
    query_states: torch.Tensor,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    scaling: float,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    q = query_states.float()
    total_kv_len = sum(k.shape[-2] for k, _ in chunks)
    q_len = q.shape[-2]
    past_len = max(0, total_kv_len - q_len)
    causal_mask = None
    if attention_mask is None:
        q_pos = torch.arange(q_len, device=q.device) + past_len
        causal_mask = q_pos
    max_score: Optional[torch.Tensor] = None
    denom: Optional[torch.Tensor] = None
    out: Optional[torch.Tensor] = None
    offset = 0
    for k, v in chunks:
        k_len = k.shape[-2]
        scores = torch.matmul(q, k.float().transpose(-2, -1)) * scaling
        if attention_mask is not None:
            mask_slice = attention_mask[..., offset : offset + k_len]
            scores = scores + mask_slice
        elif causal_mask is not None:
            k_pos = torch.arange(k_len, device=q.device) + offset
            mask = k_pos[None, None, None, :] > causal_mask[None, None, :, None]
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        chunk_max = scores.max(dim=-1, keepdim=True).values
        if max_score is None:
            max_score = chunk_max
            exp_scores = torch.exp(scores - max_score)
            denom = exp_scores.sum(dim=-1, keepdim=True)
            out = torch.matmul(exp_scores, v.float())
        else:
            new_max = torch.maximum(max_score, chunk_max)
            exp_scores = torch.exp(scores - new_max)
            denom = denom * torch.exp(max_score - new_max) + exp_scores.sum(dim=-1, keepdim=True)
            out = out * torch.exp(max_score - new_max) + torch.matmul(exp_scores, v.float())
            max_score = new_max
        offset += k_len
    if out is None or denom is None:
        raise RuntimeError("MixKVQ attention received no cache chunks.")
    return (out / denom).to(dtype=query_states.dtype)


def _streaming_attention_gqa(
    query_states: torch.Tensor,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    scaling: float,
    attention_mask: Optional[torch.Tensor],
    num_key_value_groups: int,
) -> torch.Tensor:
    q = query_states.float()
    batch, num_heads, q_len, head_dim = q.shape
    if num_heads % num_key_value_groups != 0:
        raise RuntimeError("MixKVQ GQA expects num_heads divisible by num_key_value_groups.")
    kv_heads = num_heads // num_key_value_groups
    q_grouped = q.view(batch, kv_heads, num_key_value_groups, q_len, head_dim)

    total_kv_len = sum(k.shape[-2] for k, _ in chunks)
    past_len = max(0, total_kv_len - q_len)
    causal_mask = None
    if attention_mask is None:
        q_pos = torch.arange(q_len, device=q.device) + past_len
        causal_mask = q_pos

    max_score: Optional[torch.Tensor] = None
    denom: Optional[torch.Tensor] = None
    out: Optional[torch.Tensor] = None
    offset = 0
    for k, v in chunks:
        k_len = k.shape[-2]
        kf = k.float()
        vf = v.float()

        q2 = q_grouped.reshape(batch * kv_heads, num_key_value_groups * q_len, head_dim)
        k2 = kf.reshape(batch * kv_heads, k_len, head_dim)
        scores = torch.matmul(q2, k2.transpose(-2, -1))
        scores = scores.view(batch, kv_heads, num_key_value_groups, q_len, k_len)
        scores = scores * scaling

        if attention_mask is not None:
            mask_slice = attention_mask[..., offset : offset + k_len]
            scores = scores + mask_slice[:, None, None, :, :]
        elif causal_mask is not None:
            k_pos = torch.arange(k_len, device=q.device) + offset
            mask = k_pos[None, None, None, None, :] > causal_mask[None, None, None, :, None]
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        chunk_max = scores.max(dim=-1, keepdim=True).values
        if max_score is None:
            max_score = chunk_max
            exp_scores = torch.exp(scores - max_score)
            denom = exp_scores.sum(dim=-1, keepdim=True)
            exp2 = exp_scores.view(batch * kv_heads, num_key_value_groups * q_len, k_len)
            v2 = vf.reshape(batch * kv_heads, k_len, head_dim)
            out2 = torch.matmul(exp2, v2)
            out = out2.view(batch, kv_heads, num_key_value_groups, q_len, head_dim)
        else:
            new_max = torch.maximum(max_score, chunk_max)
            exp_scores = torch.exp(scores - new_max)
            denom = denom * torch.exp(max_score - new_max) + exp_scores.sum(dim=-1, keepdim=True)
            exp2 = exp_scores.view(batch * kv_heads, num_key_value_groups * q_len, k_len)
            v2 = vf.reshape(batch * kv_heads, k_len, head_dim)
            out2 = torch.matmul(exp2, v2)
            out = out * torch.exp(max_score - new_max) + out2.view(
                batch, kv_heads, num_key_value_groups, q_len, head_dim
            )
            max_score = new_max

        offset += k_len

    if out is None or denom is None:
        raise RuntimeError("MixKVQ attention received no cache chunks.")

    out = out / denom
    return out.reshape(batch, num_heads, q_len, head_dim).to(dtype=query_states.dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _mixkvq_attention(
    query_states: torch.Tensor,
    layer: MixKVQDynamicLayer,
    *,
    scaling: float,
    attention_mask: Optional[torch.Tensor],
    num_key_value_groups: int,
) -> torch.Tensor:
    chunks = layer.iter_kv_chunks()
    if not chunks:
        raise RuntimeError("MixKVQ cache has no chunks to attend to.")
    if layer.config.use_max_op:
        try:
            from src.mixkvq_max import streaming_attention_max_from_layer

            attn_output = streaming_attention_max_from_layer(
                query_states,
                layer=layer,
                num_key_value_groups=num_key_value_groups,
                scaling=scaling,
                attention_mask=attention_mask,
            )
            return attn_output.transpose(1, 2).contiguous()
        except Exception:
            # Prototype path; fall back silently if the MAX op isn't available.
            pass
    if layer.config.use_mojo_fused:
        try:
            from src.mixkvq_mojo import streaming_attention_fused_from_layer

            attn_output = streaming_attention_fused_from_layer(
                query_states,
                layer=layer,
                num_key_value_groups=num_key_value_groups,
                scaling=scaling,
                attention_mask=attention_mask,
            )
            return attn_output.transpose(1, 2).contiguous()
        except Exception:
            # Prototype path; fall back silently if the Mojo shim isn't available.
            pass
    if num_key_value_groups != 1:
        attn_output = _streaming_attention_gqa(
            query_states,
            chunks,
            scaling=scaling,
            attention_mask=attention_mask,
            num_key_value_groups=num_key_value_groups,
        )
    else:
        attn_output = _streaming_attention(
            query_states,
            chunks,
            scaling=scaling,
            attention_mask=attention_mask,
        )
    return attn_output.transpose(1, 2).contiguous()


def patch_qwen2_attention_for_mixkvq(model: torch.nn.Module) -> int:
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    except Exception:
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, qwen2_mod.Qwen2Attention):
            continue
        if getattr(module, "_mixkvq_patched", False):
            continue

        def _forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = qwen2_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None and isinstance(past_key_values, MixKVQCache):
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                    "query_states": query_states,
                }
                layer = past_key_values.layers[self.layer_idx]
                layer.append(key_states, value_states, cache_kwargs)
                attn_output = _mixkvq_attention(
                    query_states,
                    layer,
                    scaling=self.scaling,
                    attention_mask=attention_mask,
                    num_key_value_groups=self.num_key_value_groups,
                )
                attn_weights = None
            else:
                if past_key_values is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                        "query_states": query_states,
                    }
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        self.layer_idx,
                        cache_kwargs,
                    )

                attention_interface = qwen2_mod.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = qwen2_mod.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        module.forward = _forward.__get__(module, module.__class__)
        module._mixkvq_patched = True
        patched += 1
    return patched
