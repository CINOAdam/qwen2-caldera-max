from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class QuantizedGroupwise:
    values: np.ndarray
    scales: np.ndarray
    group_size: int
    num_bits: int


def _pad_to_group(x: np.ndarray, group_size: int) -> Tuple[np.ndarray, int]:
    last_dim = x.shape[-1]
    remainder = last_dim % group_size
    if remainder == 0:
        return x, 0
    pad = group_size - remainder
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, pad)
    return np.pad(x, pad_width, mode="constant"), pad


def quantize_groupwise(x: np.ndarray, group_size: int, num_bits: int) -> QuantizedGroupwise:
    if num_bits < 2 or num_bits > 8:
        raise ValueError("num_bits must be in [2, 8]")
    x = x.astype(np.float32)

    padded, pad = _pad_to_group(x, group_size)
    last_dim = padded.shape[-1]
    groups = last_dim // group_size

    reshaped = padded.reshape(-1, groups, group_size)
    max_abs = np.max(np.abs(reshaped), axis=-1, keepdims=True)

    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    scale = np.where(max_abs == 0, 1.0, max_abs / qmax)

    quant = np.round(reshaped / scale)
    quant = np.clip(quant, qmin, qmax).astype(np.int8)

    if pad:
        quant = quant[:, :, : group_size - pad]

    quant = quant.reshape(x.shape)
    scales = scale.reshape(*x.shape[:-1], groups)

    return QuantizedGroupwise(values=quant, scales=scales, group_size=group_size, num_bits=num_bits)


def dequantize_groupwise(q: QuantizedGroupwise) -> np.ndarray:
    values = q.values.astype(np.float32)
    group_size = q.group_size
    last_dim = values.shape[-1]
    groups = math.ceil(last_dim / group_size)

    padded, pad = _pad_to_group(values, group_size)
    reshaped = padded.reshape(-1, groups, group_size)

    scales = q.scales.astype(np.float32)
    scales = scales.reshape(-1, groups, 1)

    dequant = reshaped * scales
    if pad:
        dequant = dequant[:, :, : group_size - pad]

    return dequant.reshape(values.shape)


def q_lr_forward(
    q_weight: QuantizedGroupwise,
    l_weight: QuantizedGroupwise,
    r_weight: QuantizedGroupwise,
    x: np.ndarray,
) -> np.ndarray:
    q = dequantize_groupwise(q_weight)
    l = dequantize_groupwise(l_weight)
    r = dequantize_groupwise(r_weight)

    return q @ x + l @ (r @ x)
