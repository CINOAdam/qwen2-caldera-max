from __future__ import annotations

import torch


def _pack_factor(bits: int) -> int:
    if bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    return 8 // bits


def pack_packed(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    if values.dtype != torch.int8:
        values = values.to(torch.int8)
    factor = _pack_factor(bits)
    rows, cols = values.shape
    pad = (factor - (cols % factor)) % factor
    if pad:
        pad_tensor = torch.zeros((rows, pad), dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad_tensor], dim=1)

    qmin = -(2 ** (bits - 1))
    offset = -qmin
    unsigned = (values + offset).to(torch.uint8)

    if bits == 4:
        a = unsigned[:, 0::2] & 0xF
        b = (unsigned[:, 1::2] & 0xF) << 4
        packed = a | b
    else:
        v0 = unsigned[:, 0::4] & 0x3
        v1 = (unsigned[:, 1::4] & 0x3) << 2
        v2 = (unsigned[:, 2::4] & 0x3) << 4
        v3 = (unsigned[:, 3::4] & 0x3) << 6
        packed = v0 | v1 | v2 | v3

    return packed.contiguous(), pad


def unpack_packed(packed: torch.Tensor, bits: int, original_cols: int) -> torch.Tensor:
    factor = _pack_factor(bits)
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    rows = packed.shape[0]

    if bits == 4:
        a = packed & 0xF
        b = (packed >> 4) & 0xF
        vals = torch.stack([a, b], dim=-1).reshape(rows, -1)
    else:
        v0 = packed & 0x3
        v1 = (packed >> 2) & 0x3
        v2 = (packed >> 4) & 0x3
        v3 = (packed >> 6) & 0x3
        vals = torch.stack([v0, v1, v2, v3], dim=-1).reshape(rows, -1)

    vals = vals[:, :original_cols]
    offset = 2 ** (bits - 1)
    return (vals.to(torch.int16) - offset).to(torch.int8)
