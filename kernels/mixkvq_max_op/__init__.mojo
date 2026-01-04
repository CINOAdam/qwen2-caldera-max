"""
MAX custom op: MixKVQ streaming attention (int4-only, prototype).

This custom op consumes pre-quantized int4 K/V tensors and per-chunk/per-token
scales directly from device memory via MAX's InputTensor/OutputTensor APIs.
"""

import compiler
from math import exp, sqrt
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


fn dequant_val(qv: Int, zero: Int, scale: Float32) -> Float32:
    return Float32(qv - zero) * scale


fn streaming_attention_gqa_kernel_fused(
    q: InputTensor[dtype=DType.float32, rank=4],
    k_q: InputTensor[dtype=DType.int32, rank=4],
    v_q: InputTensor[dtype=DType.int32, rank=4],
    k_scales: InputTensor[dtype=DType.float32, rank=4],
    v_scales: InputTensor[dtype=DType.float32, rank=3],
    offsets: InputTensor[dtype=DType.int32, rank=1],
    lengths: InputTensor[dtype=DType.int32, rank=1],
    mask: InputTensor[dtype=DType.float32, rank=3],
    num_chunks: Int,
    output: OutputTensor[dtype=DType.float32, rank=4],
    batch: Int,
    heads: Int,
    q_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
    scaling: Float32,
    zero: Int,
):
    for b in range(batch):
        for h in range(heads):
            var kv_h = h // num_key_value_groups
            for t in range(q_len):
                var max_score: Float32 = -1.0e30
                var denom: Float32 = 0.0

                for d in range(head_dim):
                    output[b, h, t, d] = 0.0

                for c in range(num_chunks):
                    var offset = Int(offsets[c])
                    var length = Int(lengths[c])
                    var chunk_max: Float32 = -1.0e30

                    for s in range(length):
                        var score: Float32 = 0.0
                        var k_idx = offset + s
                        for d in range(head_dim):
                            var kq = Int(k_q[b, kv_h, k_idx, d])
                            var k_scale = k_scales[b, c, kv_h, d]
                            score += q[b, h, t, d] * dequant_val(kq, zero, k_scale)
                        score = score * scaling + mask[b, t, k_idx]
                        if score > chunk_max:
                            chunk_max = score

                    var new_max = chunk_max if chunk_max > max_score else max_score
                    var scale_old: Float32 = 0.0
                    if denom > 0.0:
                        scale_old = exp(max_score - new_max).cast[DType.float32]()
                    if scale_old > 0.0:
                        for d in range(head_dim):
                            output[b, h, t, d] *= scale_old
                        denom *= scale_old

                    for s in range(length):
                        var score: Float32 = 0.0
                        var k_idx = offset + s
                        for d in range(head_dim):
                            var kq = Int(k_q[b, kv_h, k_idx, d])
                            var k_scale = k_scales[b, c, kv_h, d]
                            score += q[b, h, t, d] * dequant_val(kq, zero, k_scale)
                        score = score * scaling + mask[b, t, k_idx]
                        var weight = exp(score - new_max).cast[DType.float32]()
                        denom += weight
                        var v_scale = v_scales[b, kv_h, k_idx]
                        for d in range(head_dim):
                            var vq = Int(v_q[b, kv_h, k_idx, d])
                            output[b, h, t, d] += weight * dequant_val(vq, zero, v_scale)

                    max_score = new_max

                for d in range(head_dim):
                    output[b, h, t, d] /= denom


@compiler.register("mixkvq_streaming_fused")
struct MixKVQStreamingFused:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=4],
        q: InputTensor[dtype=DType.float32, rank=4],
        k_q: InputTensor[dtype=DType.int32, rank=4],
        v_q: InputTensor[dtype=DType.int32, rank=4],
        k_scales: InputTensor[dtype=DType.float32, rank=4],
        v_scales: InputTensor[dtype=DType.float32, rank=3],
        offsets: InputTensor[dtype=DType.int32, rank=1],
        lengths: InputTensor[dtype=DType.int32, rank=1],
        mask: InputTensor[dtype=DType.float32, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        var batch = q.dim_size(0)
        var heads = q.dim_size(1)
        var q_len = q.dim_size(2)
        var head_dim = q.dim_size(3)
        var kv_heads = k_q.dim_size(1)
        var num_key_value_groups = heads // kv_heads
        var num_chunks = offsets.dim_size(0)
        var scaling = 1.0 / sqrt(Float32(head_dim))
        var zero = 8

        streaming_attention_gqa_kernel_fused(
            q,
            k_q,
            v_q,
            k_scales,
            v_scales,
            offsets,
            lengths,
            mask,
            num_chunks,
            output,
            batch,
            heads,
            q_len,
            head_dim,
            num_key_value_groups,
            scaling,
            zero,
        )
