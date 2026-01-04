#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess

import torch

from src.mixkvq import MixKVQConfig, MixKVQDynamicLayer, _mixkvq_attention
from src.mixkvq_mojo import streaming_attention_fused_from_segments


def main() -> None:
    subprocess.run(["scripts/build_mojo_kernels.sh"], check=True)

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(0)

    batch = 1
    heads = 4
    kv_heads = 2
    q_len = 2
    seq_len = 4
    head_dim = 8
    groups = heads // kv_heads

    key_states = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    value_states = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    query_states = torch.randn(batch, heads, q_len, head_dim, device=device, dtype=dtype)

    config = MixKVQConfig(
        ratio_bf16=0.0,
        ratio_int4=1.0,
        key_bits_low=4,
        key_bits_mid=4,
        value_bits=4,
        update_interval=1,
        buffer_size=seq_len,
        pack_bits=False,
    )
    layer = MixKVQDynamicLayer(config)
    layer.append(key_states, value_states, cache_kwargs=None)

    scaling = 1.0 / math.sqrt(head_dim)
    attn_py = _mixkvq_attention(
        query_states,
        layer,
        scaling=scaling,
        attention_mask=None,
        num_key_value_groups=groups,
    )
    attn_mojo = streaming_attention_fused_from_segments(
        query_states,
        key_segments=layer.key_segments,
        value_segments=layer.value_segments,
        head_dim=head_dim,
        num_key_value_groups=groups,
        scaling=scaling,
    )
    attn_mojo = attn_mojo.transpose(1, 2).contiguous()

    diff = (attn_py - attn_mojo).abs()
    print(f"max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e}")


if __name__ == "__main__":
    main()
