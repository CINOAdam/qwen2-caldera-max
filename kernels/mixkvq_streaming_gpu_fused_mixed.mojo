"""
MixKVQ streaming attention prototype (GPU) with fused dequant and mixed key precision.

Supports per-dimension key precision via a mask:
  0 = bf16 (float), 1 = int4, 2 = int2.
Uses per-chunk K scales for int4/int2 and per-token V scales (int4 only).

Usage:
  mojo kernels/mixkvq_streaming_gpu_fused_mixed.mojo
"""

from math import exp, sqrt
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from memory import UnsafePointer, MutUnsafePointer
from random import random_float64, rand


comptime float_dtype = DType.float32
comptime q_dtype = DType.int32
comptime layout = Layout.row_major(1)


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
    total_len: Int,
    head_dim: Int,
) -> Int:
    return (((b * kv_heads + h) * total_len + s) * head_dim + d)


fn max_abs_diff(a: UnsafePointer[Float32], b: UnsafePointer[Float32], size: Int) -> Float32:
    var max_val: Float32 = 0.0
    for i in range(size):
        var diff = a[i] - b[i]
        if diff < 0.0:
            diff = -diff
        if diff > max_val:
            max_val = diff
    return max_val


fn lt_scalar(t: LayoutTensor[float_dtype, layout, MutAnyOrigin], idx: Int) -> Float32:
    return t[idx].reduce_add()


fn lt_scalar_i32(t: LayoutTensor[q_dtype, layout, MutAnyOrigin], idx: Int) -> Int:
    return Int(t[idx].reduce_add())


fn dequant_val(qv: Int, zero: Int, scale: Float32) -> Float32:
    return Float32(qv - zero) * scale


fn attention_gqa_naive_chunks_mixed(
    q: UnsafePointer[Float32],
    k_bf16: UnsafePointer[Float32],
    k_q4: UnsafePointer[Int32],
    k_q2: UnsafePointer[Int32],
    k_scale4: UnsafePointer[Float32],
    k_scale2: UnsafePointer[Float32],
    k_mask: UnsafePointer[Int32],
    v_q: UnsafePointer[Int32],
    v_scales: UnsafePointer[Float32],
    chunk_offsets: UnsafePointer[Int32],
    chunk_lens: UnsafePointer[Int32],
    num_chunks: Int,
    total_len: Int,
    out_ptr: MutUnsafePointer[Float32],
    batch: Int,
    heads: Int,
    kv_heads: Int,
    q_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
    scaling: Float32,
    zero4: Int,
    zero2: Int,
):
    for b in range(batch):
        for h in range(heads):
            var kv_h = h // num_key_value_groups
            for t in range(q_len):
                var max_score: Float32 = -1.0e30
                for c in range(num_chunks):
                    var offset = Int(chunk_offsets[c])
                    var length = Int(chunk_lens[c])
                    for s in range(length):
                        var score: Float32 = 0.0
                        var k_idx = offset + s
                        for d in range(head_dim):
                            var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                            var mask = Int(k_mask[d])
                            if mask == 0:
                                var kval = k_bf16[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                                score += qv * kval
                            elif mask == 1:
                                var kq = k_q4[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                                var scale4 = k_scale4[((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d]
                                score += qv * dequant_val(Int(kq), zero4, scale4)
                            else:
                                var kq = k_q2[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                                var scale2 = k_scale2[((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d]
                                score += qv * dequant_val(Int(kq), zero2, scale2)
                        score *= scaling
                        if score > max_score:
                            max_score = score

                var denom: Float32 = 0.0
                for d in range(head_dim):
                    out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] = 0.0

                for c in range(num_chunks):
                    var offset = Int(chunk_offsets[c])
                    var length = Int(chunk_lens[c])
                    for s in range(length):
                        var score: Float32 = 0.0
                        var k_idx = offset + s
                        for d in range(head_dim):
                            var qv = q[idx_q(b, h, t, d, heads, q_len, head_dim)]
                            var mask = Int(k_mask[d])
                            if mask == 0:
                                var kval = k_bf16[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                                score += qv * kval
                            elif mask == 1:
                                var kq = k_q4[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                                var scale4 = k_scale4[((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d]
                                score += qv * dequant_val(Int(kq), zero4, scale4)
                            else:
                                var kq = k_q2[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                                var scale2 = k_scale2[((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d]
                                score += qv * dequant_val(Int(kq), zero2, scale2)
                        score *= scaling
                        var weight = exp(score - max_score).cast[DType.float32]()
                        denom += weight
                        var v_scale = v_scales[(b * kv_heads + kv_h) * total_len + k_idx]
                        for d in range(head_dim):
                            var vq = v_q[idx_kv(b, kv_h, k_idx, d, kv_heads, total_len, head_dim)]
                            out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] += (
                                weight * dequant_val(Int(vq), zero4, v_scale)
                            )

                for d in range(head_dim):
                    out_ptr[idx_q(b, h, t, d, heads, q_len, head_dim)] /= denom


fn streaming_attention_gqa_kernel_fused_mixed(
    q: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    k_bf16: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    k_q4: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    k_q2: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    k_scale4: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    k_scale2: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    k_mask: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    v_q: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    v_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    chunk_offsets: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    chunk_lens: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    num_chunks: Int,
    total_len: Int,
    out_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    batch: Int,
    heads: Int,
    q_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
    scaling: Float32,
    zero4: Int,
    zero2: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = UInt(batch * heads * q_len)
    if tid >= total_threads:
        return

    var t = Int(tid % UInt(q_len))
    var tmp = Int(tid // UInt(q_len))
    var h = tmp % heads
    var b = tmp // heads
    var kv_heads = heads // num_key_value_groups
    var kv_h = h // num_key_value_groups
    var q_base = ((b * heads + h) * q_len + t) * head_dim
    var past_len = total_len - q_len
    if past_len < 0:
        past_len = 0
    var max_k_idx = past_len + t

    var max_score: Float32 = -1.0e30
    var denom: Float32 = 0.0

    for d in range(head_dim):
        out_tensor[q_base + d] = 0.0

    for c in range(num_chunks):
        var offset = lt_scalar_i32(chunk_offsets, c)
        var length = lt_scalar_i32(chunk_lens, c)
        if offset > max_k_idx:
            break
        var chunk_max: Float32 = -1.0e30

        for s in range(length):
            var score: Float32 = 0.0
            var k_idx = offset + s
            if k_idx > max_k_idx:
                continue
            var k_base = ((b * kv_heads + kv_h) * total_len + k_idx) * head_dim
            for d in range(head_dim):
                var mask = lt_scalar_i32(k_mask, d)
                if mask == 0:
                    score += lt_scalar(q, q_base + d) * lt_scalar(k_bf16, k_base + d)
                elif mask == 1:
                    var kq = lt_scalar_i32(k_q4, k_base + d)
                    var scale4 = lt_scalar(
                        k_scale4, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d
                    )
                    score += lt_scalar(q, q_base + d) * dequant_val(kq, zero4, scale4)
                else:
                    var kq = lt_scalar_i32(k_q2, k_base + d)
                    var scale2 = lt_scalar(
                        k_scale2, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d
                    )
                    score += lt_scalar(q, q_base + d) * dequant_val(kq, zero2, scale2)
            score *= scaling
            if score > chunk_max:
                chunk_max = score

        var new_max = chunk_max if chunk_max > max_score else max_score
        var scale_old: Float32 = 0.0
        if denom > 0.0:
            scale_old = exp(max_score - new_max).cast[DType.float32]()
        if scale_old > 0.0:
            for d in range(head_dim):
                out_tensor[q_base + d] *= scale_old
            denom *= scale_old

        for s in range(length):
            var score: Float32 = 0.0
            var k_idx = offset + s
            if k_idx > max_k_idx:
                continue
            var k_base = ((b * kv_heads + kv_h) * total_len + k_idx) * head_dim
            for d in range(head_dim):
                var mask = lt_scalar_i32(k_mask, d)
                if mask == 0:
                    score += lt_scalar(q, q_base + d) * lt_scalar(k_bf16, k_base + d)
                elif mask == 1:
                    var kq = lt_scalar_i32(k_q4, k_base + d)
                    var scale4 = lt_scalar(
                        k_scale4, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d
                    )
                    score += lt_scalar(q, q_base + d) * dequant_val(kq, zero4, scale4)
                else:
                    var kq = lt_scalar_i32(k_q2, k_base + d)
                    var scale2 = lt_scalar(
                        k_scale2, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d
                    )
                    score += lt_scalar(q, q_base + d) * dequant_val(kq, zero2, scale2)
            score *= scaling
            var weight = exp(score - new_max).cast[DType.float32]()
            denom += weight
            var v_scale = lt_scalar(v_scales, (b * kv_heads + kv_h) * total_len + k_idx)
            for d in range(head_dim):
                var vq = lt_scalar_i32(v_q, k_base + d)
                out_tensor[q_base + d] += weight * dequant_val(vq, zero4, v_scale)

        max_score = new_max

    for d in range(head_dim):
        out_tensor[q_base + d] /= denom


fn main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    var batch = 1
    var heads = 4
    var num_key_value_groups = 2
    var kv_heads = heads // num_key_value_groups
    var q_len = 4
    var head_dim = 16
    var scaling = 1.0 / sqrt(Float32(head_dim))

    var num_chunks = 3
    var chunk_lengths = [3, 4, 5]
    var total_len = 0
    for c in range(num_chunks):
        total_len += chunk_lengths[c]

    var q_size = batch * heads * q_len * head_dim
    var kv_size = batch * kv_heads * total_len * head_dim
    var k_scale_size = batch * num_chunks * kv_heads * head_dim
    var v_scale_size = batch * kv_heads * total_len
    var zero4 = 8
    var zero2 = 2

    var host_q = ctx.enqueue_create_host_buffer[float_dtype](q_size)
    var host_k_bf16 = ctx.enqueue_create_host_buffer[float_dtype](kv_size)
    var host_k_q4 = ctx.enqueue_create_host_buffer[q_dtype](kv_size)
    var host_k_q2 = ctx.enqueue_create_host_buffer[q_dtype](kv_size)
    var host_k_scale4 = ctx.enqueue_create_host_buffer[float_dtype](k_scale_size)
    var host_k_scale2 = ctx.enqueue_create_host_buffer[float_dtype](k_scale_size)
    var host_k_mask = ctx.enqueue_create_host_buffer[q_dtype](head_dim)
    var host_v_q = ctx.enqueue_create_host_buffer[q_dtype](kv_size)
    var host_v_scales = ctx.enqueue_create_host_buffer[float_dtype](v_scale_size)
    var host_out_gpu = ctx.enqueue_create_host_buffer[float_dtype](q_size)
    var host_out_ref = ctx.enqueue_create_host_buffer[float_dtype](q_size)

    var host_offsets = ctx.enqueue_create_host_buffer[DType.int32](num_chunks)
    var host_lengths = ctx.enqueue_create_host_buffer[DType.int32](num_chunks)

    rand(host_q.unsafe_ptr(), q_size)
    rand(host_k_bf16.unsafe_ptr(), kv_size)

    for i in range(kv_size):
        var rv = random_float64() * 16.0
        host_k_q4.unsafe_ptr().offset(i).store(Int32(rv))
        rv = random_float64() * 4.0
        host_k_q2.unsafe_ptr().offset(i).store(Int32(rv))
        rv = random_float64() * 16.0
        host_v_q.unsafe_ptr().offset(i).store(Int32(rv))

    for i in range(k_scale_size):
        var rv = 0.01 + random_float64() * 0.05
        host_k_scale4.unsafe_ptr().offset(i).store(rv.cast[DType.float32]())
        rv = 0.01 + random_float64() * 0.05
        host_k_scale2.unsafe_ptr().offset(i).store(rv.cast[DType.float32]())

    for i in range(v_scale_size):
        var rv = 0.01 + random_float64() * 0.05
        host_v_scales.unsafe_ptr().offset(i).store(rv.cast[DType.float32]())

    for d in range(head_dim):
        var mode = d % 3
        host_k_mask.unsafe_ptr().offset(d).store(Int32(mode))

    var offset = 0
    for c in range(num_chunks):
        host_offsets.unsafe_ptr().offset(c).store(Int32(offset))
        host_lengths.unsafe_ptr().offset(c).store(Int32(chunk_lengths[c]))
        offset += chunk_lengths[c]

    var dev_q = ctx.enqueue_create_buffer[float_dtype](q_size)
    var dev_k_bf16 = ctx.enqueue_create_buffer[float_dtype](kv_size)
    var dev_k_q4 = ctx.enqueue_create_buffer[q_dtype](kv_size)
    var dev_k_q2 = ctx.enqueue_create_buffer[q_dtype](kv_size)
    var dev_k_scale4 = ctx.enqueue_create_buffer[float_dtype](k_scale_size)
    var dev_k_scale2 = ctx.enqueue_create_buffer[float_dtype](k_scale_size)
    var dev_k_mask = ctx.enqueue_create_buffer[q_dtype](head_dim)
    var dev_v_q = ctx.enqueue_create_buffer[q_dtype](kv_size)
    var dev_v_scales = ctx.enqueue_create_buffer[float_dtype](v_scale_size)
    var dev_offsets = ctx.enqueue_create_buffer[DType.int32](num_chunks)
    var dev_lengths = ctx.enqueue_create_buffer[DType.int32](num_chunks)
    var dev_out = ctx.enqueue_create_buffer[float_dtype](q_size)

    ctx.enqueue_copy(dst_buf=dev_q, src_buf=host_q)
    ctx.enqueue_copy(dst_buf=dev_k_bf16, src_buf=host_k_bf16)
    ctx.enqueue_copy(dst_buf=dev_k_q4, src_buf=host_k_q4)
    ctx.enqueue_copy(dst_buf=dev_k_q2, src_buf=host_k_q2)
    ctx.enqueue_copy(dst_buf=dev_k_scale4, src_buf=host_k_scale4)
    ctx.enqueue_copy(dst_buf=dev_k_scale2, src_buf=host_k_scale2)
    ctx.enqueue_copy(dst_buf=dev_k_mask, src_buf=host_k_mask)
    ctx.enqueue_copy(dst_buf=dev_v_q, src_buf=host_v_q)
    ctx.enqueue_copy(dst_buf=dev_v_scales, src_buf=host_v_scales)
    ctx.enqueue_copy(dst_buf=dev_offsets, src_buf=host_offsets)
    ctx.enqueue_copy(dst_buf=dev_lengths, src_buf=host_lengths)
    ctx.synchronize()

    var q_tensor = LayoutTensor[float_dtype, layout](dev_q)
    var k_bf16_tensor = LayoutTensor[float_dtype, layout](dev_k_bf16)
    var k_q4_tensor = LayoutTensor[q_dtype, layout](dev_k_q4)
    var k_q2_tensor = LayoutTensor[q_dtype, layout](dev_k_q2)
    var k_scale4_tensor = LayoutTensor[float_dtype, layout](dev_k_scale4)
    var k_scale2_tensor = LayoutTensor[float_dtype, layout](dev_k_scale2)
    var k_mask_tensor = LayoutTensor[q_dtype, layout](dev_k_mask)
    var v_q_tensor = LayoutTensor[q_dtype, layout](dev_v_q)
    var v_scales_tensor = LayoutTensor[float_dtype, layout](dev_v_scales)
    var offsets_tensor = LayoutTensor[DType.int32, layout](dev_offsets)
    var lengths_tensor = LayoutTensor[DType.int32, layout](dev_lengths)
    var out_tensor = LayoutTensor[float_dtype, layout](dev_out)

    var block_size = 128
    var total_threads = batch * heads * q_len
    var num_blocks = (total_threads + block_size - 1) // block_size

    ctx.enqueue_function_checked[
        streaming_attention_gqa_kernel_fused_mixed,
        streaming_attention_gqa_kernel_fused_mixed,
    ](
        q_tensor,
        k_bf16_tensor,
        k_q4_tensor,
        k_q2_tensor,
        k_scale4_tensor,
        k_scale2_tensor,
        k_mask_tensor,
        v_q_tensor,
        v_scales_tensor,
        offsets_tensor,
        lengths_tensor,
        num_chunks,
        total_len,
        out_tensor,
        batch,
        heads,
        q_len,
        head_dim,
        num_key_value_groups,
        scaling,
        zero4,
        zero2,
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=host_out_gpu, src_buf=dev_out)
    ctx.synchronize()

    attention_gqa_naive_chunks_mixed(
        host_q.unsafe_ptr(),
        host_k_bf16.unsafe_ptr(),
        host_k_q4.unsafe_ptr(),
        host_k_q2.unsafe_ptr(),
        host_k_scale4.unsafe_ptr(),
        host_k_scale2.unsafe_ptr(),
        host_k_mask.unsafe_ptr(),
        host_v_q.unsafe_ptr(),
        host_v_scales.unsafe_ptr(),
        host_offsets.unsafe_ptr(),
        host_lengths.unsafe_ptr(),
        num_chunks,
        total_len,
        host_out_ref.unsafe_ptr(),
        batch,
        heads,
        kv_heads,
        q_len,
        head_dim,
        num_key_value_groups,
        scaling,
        zero4,
        zero2,
    )

    var diff = max_abs_diff(host_out_ref.unsafe_ptr(), host_out_gpu.unsafe_ptr(), q_size)
    print("Max abs diff:", diff)
