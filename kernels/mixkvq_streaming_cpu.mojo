"""
MixKVQ streaming attention prototype (CPU) in Mojo v0.26+.

This validates the chunked softmax accumulation for grouped-query attention (GQA)
without expanding KV heads. It compares a streaming implementation against a
naive full-softmax reference on small random inputs.

Usage:
  mojo kernels/mixkvq_streaming_cpu.mojo
"""

from math import exp
from memory import UnsafePointer, MutUnsafePointer, alloc
from random import random_float64


fn idx_q(
    b: Int,
    h: Int,
    t: Int,
    d: Int,
    heads: Int,
    q_len: Int,
    head_dim: Int,
) -> Int:
    return (((b * heads + h) * q_len + t) * head_dim + d)


fn idx_kv(
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    kv_heads: Int,
    kv_len: Int,
    head_dim: Int,
) -> Int:
    return (((b * kv_heads + h) * kv_len + s) * head_dim + d)


fn fill_random(ptr: MutUnsafePointer[Float32], size: Int, scale: Float32):
    for i in range(size):
        var v = random_float64().cast[DType.float32]()
        v = v * 2.0 - 1.0
        ptr[i] = v * scale


fn max_abs_diff(a: UnsafePointer[Float32], b: UnsafePointer[Float32], size: Int) -> Float32:
    var max_val: Float32 = 0.0
    for i in range(size):
        var diff = a[i] - b[i]
        if diff < 0.0:
            diff = -diff
        if diff > max_val:
            max_val = diff
    return max_val


fn attention_gqa_two_chunks_naive(
    q: UnsafePointer[Float32],
    k0: UnsafePointer[Float32],
    v0: UnsafePointer[Float32],
    k1: UnsafePointer[Float32],
    v1: UnsafePointer[Float32],
    out_ptr: MutUnsafePointer[Float32],
    batch: Int,
    heads: Int,
    kv_heads: Int,
    q_len: Int,
    k0_len: Int,
    k1_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
):
    for b in range(batch):
        for h in range(heads):
            var kv_h = h // num_key_value_groups
            for t in range(q_len):
                var max_score: Float32 = -1.0e30
                for s in range(k0_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k0[idx_kv(b, kv_h, s, d, kv_heads, k0_len, head_dim)]
                        score += qv * kv
                    if score > max_score:
                        max_score = score
                for s in range(k1_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k1[idx_kv(b, kv_h, s, d, kv_heads, k1_len, head_dim)]
                        score += qv * kv
                    if score > max_score:
                        max_score = score

                var denom: Float32 = 0.0
                for d in range(head_dim):
                    out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] = 0.0

                for s in range(k0_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k0[idx_kv(b, kv_h, s, d, kv_heads, k0_len, head_dim)]
                        score += qv * kv
                    var weight = exp(score - max_score).cast[DType.float32]()
                    denom += weight
                    for d in range(head_dim):
                        var vv = v0[idx_kv(b, kv_h, s, d, kv_heads, k0_len, head_dim)]
                        out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] += weight * vv

                for s in range(k1_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k1[idx_kv(b, kv_h, s, d, kv_heads, k1_len, head_dim)]
                        score += qv * kv
                    var weight = exp(score - max_score).cast[DType.float32]()
                    denom += weight
                    for d in range(head_dim):
                        var vv = v1[idx_kv(b, kv_h, s, d, kv_heads, k1_len, head_dim)]
                        out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] += weight * vv

                for d in range(head_dim):
                    out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] /= denom


fn attention_gqa_two_chunks_streaming(
    q: UnsafePointer[Float32],
    k0: UnsafePointer[Float32],
    v0: UnsafePointer[Float32],
    k1: UnsafePointer[Float32],
    v1: UnsafePointer[Float32],
    out_ptr: MutUnsafePointer[Float32],
    batch: Int,
    heads: Int,
    kv_heads: Int,
    q_len: Int,
    k0_len: Int,
    k1_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
):
    for b in range(batch):
        for h in range(heads):
            var kv_h = h // num_key_value_groups
            for t in range(q_len):
                var max_score: Float32 = -1.0e30
                var denom: Float32 = 0.0
                for d in range(head_dim):
                    out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] = 0.0

                # Chunk 0
                var chunk_max: Float32 = -1.0e30
                for s in range(k0_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k0[idx_kv(b, kv_h, s, d, kv_heads, k0_len, head_dim)]
                        score += qv * kv
                    if score > chunk_max:
                        chunk_max = score
                var new_max = chunk_max if chunk_max > max_score else max_score
                var scale_old: Float32 = 0.0
                if denom > 0.0:
                    scale_old = exp(max_score - new_max).cast[DType.float32]()
                if scale_old > 0.0:
                    for d in range(head_dim):
                        out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] *= scale_old
                    denom *= scale_old

                for s in range(k0_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k0[idx_kv(b, kv_h, s, d, kv_heads, k0_len, head_dim)]
                        score += qv * kv
                    var weight = exp(score - new_max).cast[DType.float32]()
                    denom += weight
                    for d in range(head_dim):
                        var vv = v0[idx_kv(b, kv_h, s, d, kv_heads, k0_len, head_dim)]
                        out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] += weight * vv
                max_score = new_max

                # Chunk 1
                chunk_max = -1.0e30
                for s in range(k1_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k1[idx_kv(b, kv_h, s, d, kv_heads, k1_len, head_dim)]
                        score += qv * kv
                    if score > chunk_max:
                        chunk_max = score
                new_max = chunk_max if chunk_max > max_score else max_score
                scale_old = exp(max_score - new_max).cast[DType.float32]()
                if scale_old > 0.0:
                    for d in range(head_dim):
                        out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] *= scale_old
                    denom *= scale_old

                for s in range(k1_len):
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                        var kv = k1[idx_kv(b, kv_h, s, d, kv_heads, k1_len, head_dim)]
                        score += qv * kv
                    var weight = exp(score - new_max).cast[DType.float32]()
                    denom += weight
                    for d in range(head_dim):
                        var vv = v1[idx_kv(b, kv_h, s, d, kv_heads, k1_len, head_dim)]
                        out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] += weight * vv
                for d in range(head_dim):
                    out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] /= denom


fn main():
    var batch = 1
    var heads = 4
    var num_key_value_groups = 2
    var kv_heads = heads // num_key_value_groups
    var q_len = 4
    var k0_len = 4
    var k1_len = 4
    var head_dim = 16

    var q = alloc[Float32](batch * heads * q_len * head_dim)
    var k0 = alloc[Float32](batch * kv_heads * k0_len * head_dim)
    var v0 = alloc[Float32](batch * kv_heads * k0_len * head_dim)
    var k1 = alloc[Float32](batch * kv_heads * k1_len * head_dim)
    var v1 = alloc[Float32](batch * kv_heads * k1_len * head_dim)

    fill_random(q, batch * heads * q_len * head_dim, 0.1)
    fill_random(k0, batch * kv_heads * k0_len * head_dim, 0.1)
    fill_random(v0, batch * kv_heads * k0_len * head_dim, 0.1)
    fill_random(k1, batch * kv_heads * k1_len * head_dim, 0.1)
    fill_random(v1, batch * kv_heads * k1_len * head_dim, 0.1)

    var out_naive = alloc[Float32](batch * heads * q_len * head_dim)
    var out_stream = alloc[Float32](batch * heads * q_len * head_dim)

    attention_gqa_two_chunks_naive(
        q,
        k0,
        v0,
        k1,
        v1,
        out_naive,
        batch,
        heads,
        kv_heads,
        q_len,
        k0_len,
        k1_len,
        head_dim,
        num_key_value_groups,
    )

    attention_gqa_two_chunks_streaming(
        q,
        k0,
        v0,
        k1,
        v1,
        out_stream,
        batch,
        heads,
        kv_heads,
        q_len,
        k0_len,
        k1_len,
        head_dim,
        num_key_value_groups,
    )

    var size = batch * heads * q_len * head_dim
    var diff = max_abs_diff(out_naive, out_stream, size)
    print("Max abs diff:", diff)

    q.free()
    k0.free()
    v0.free()
    k1.free()
    v1.free()
    out_naive.free()
    out_stream.free()
