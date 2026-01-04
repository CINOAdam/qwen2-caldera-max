"""
MixKVQ streaming attention fused kernel (GPU) for shared-lib build.

Exports mixkvq_streaming_fused_host for Python ctypes integration.
"""

from math import exp
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from memory import UnsafePointer, MutUnsafePointer


comptime float_dtype = DType.float32
comptime q_dtype = DType.int32
comptime layout = Layout.row_major(1)


fn lt_scalar(t: LayoutTensor[float_dtype, layout, MutAnyOrigin], idx: Int) -> Float32:
    return t[idx].reduce_add()


fn lt_scalar_i32(t: LayoutTensor[q_dtype, layout, MutAnyOrigin], idx: Int) -> Int:
    return Int(t[idx].reduce_add())


fn dequant_val(qv: Int, zero: Int, scale: Float32) -> Float32:
    return Float32(qv - zero) * scale


fn round_to_int(x: Float32) -> Int:
    if x >= 0.0:
        return Int(x + 0.5)
    return Int(x - 0.5)


fn streaming_attention_gqa_kernel_fused(
    q: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    k_q: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    v_q: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    k_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    v_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    mask: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    chunk_offsets: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    chunk_lens: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    use_mask: Int,
    num_chunks: Int,
    total_len: Int,
    out_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    batch: Int,
    heads: Int,
    q_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
    scaling: Float32,
    zero: Int,
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
        if use_mask == 0 and offset > max_k_idx:
            break
        var chunk_max: Float32 = -1.0e30

        for s in range(length):
            var score: Float32 = 0.0
            var k_idx = offset + s
            if use_mask == 0 and k_idx > max_k_idx:
                continue
            var k_base = ((b * kv_heads + kv_h) * total_len + k_idx) * head_dim
            for d in range(head_dim):
                var kq = lt_scalar_i32(k_q, k_base + d)
                var k_scale = lt_scalar(
                    k_scales, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d
                )
                score += lt_scalar(q, q_base + d) * dequant_val(kq, zero, k_scale)
            score *= scaling
            if use_mask != 0:
                score += lt_scalar(mask, (b * q_len + t) * total_len + k_idx)
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
            if use_mask == 0 and k_idx > max_k_idx:
                continue
            var k_base = ((b * kv_heads + kv_h) * total_len + k_idx) * head_dim
            for d in range(head_dim):
                var kq = lt_scalar_i32(k_q, k_base + d)
                var k_scale = lt_scalar(
                    k_scales, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d
                )
                score += lt_scalar(q, q_base + d) * dequant_val(kq, zero, k_scale)
            score *= scaling
            if use_mask != 0:
                score += lt_scalar(mask, (b * q_len + t) * total_len + k_idx)
            var weight = exp(score - new_max).cast[DType.float32]()
            denom += weight
            var v_scale = lt_scalar(v_scales, (b * kv_heads + kv_h) * total_len + k_idx)
            for d in range(head_dim):
                var vq = lt_scalar_i32(v_q, k_base + d)
                out_tensor[q_base + d] += weight * dequant_val(vq, zero, v_scale)

        max_score = new_max

    for d in range(head_dim):
        out_tensor[q_base + d] /= denom


fn compute_k_scales_kernel(
    k: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    chunk_offsets: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    chunk_lens: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    num_chunks: Int,
    total_len: Int,
    k_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    batch: Int,
    kv_heads: Int,
    head_dim: Int,
    qmax: Int,
    eps: Float32,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = UInt(batch * kv_heads * num_chunks * head_dim)
    if tid >= total_threads:
        return

    var d = Int(tid % UInt(head_dim))
    var tmp = Int(tid // UInt(head_dim))
    var c = tmp % num_chunks
    tmp = tmp // num_chunks
    var kv_h = tmp % kv_heads
    var b = tmp // kv_heads

    var offset = lt_scalar_i32(chunk_offsets, c)
    var length = lt_scalar_i32(chunk_lens, c)
    var max_abs: Float32 = 0.0

    for s in range(length):
        var k_idx = offset + s
        var k_base = ((b * kv_heads + kv_h) * total_len + k_idx) * head_dim
        var kval = lt_scalar(k, k_base + d)
        if kval < 0.0:
            kval = -kval
        if kval > max_abs:
            max_abs = kval

    var qmax_f = Float32(qmax)
    var scale = max_abs / qmax_f
    if scale < eps:
        scale = eps
    k_scales[((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d] = scale


fn quantize_k_kernel(
    k: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    k_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    chunk_ids: LayoutTensor[DType.int32, layout, MutAnyOrigin],
    num_chunks: Int,
    total_len: Int,
    k_q: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    batch: Int,
    kv_heads: Int,
    head_dim: Int,
    qmax: Int,
    zero: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = UInt(batch * kv_heads * total_len * head_dim)
    if tid >= total_threads:
        return

    var d = Int(tid % UInt(head_dim))
    var tmp = Int(tid // UInt(head_dim))
    var s = tmp % total_len
    tmp = tmp // total_len
    var kv_h = tmp % kv_heads
    var b = tmp // kv_heads

    var c = lt_scalar_i32(chunk_ids, s)
    var scale = lt_scalar(k_scales, ((b * num_chunks + c) * kv_heads + kv_h) * head_dim + d)
    var k_base = ((b * kv_heads + kv_h) * total_len + s) * head_dim + d
    var kval = lt_scalar(k, k_base)
    var q = round_to_int(kval / scale)
    var qmin = -qmax - 1
    if q < qmin:
        q = qmin
    if q > qmax:
        q = qmax
    k_q[k_base] = Int32(q + zero)


fn quantize_v_kernel(
    v: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    v_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    v_q: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    total_len: Int,
    batch: Int,
    kv_heads: Int,
    head_dim: Int,
    qmax: Int,
    zero: Int,
    eps: Float32,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = UInt(batch * kv_heads * total_len)
    if tid >= total_threads:
        return

    var s = Int(tid % UInt(total_len))
    var tmp = Int(tid // UInt(total_len))
    var kv_h = tmp % kv_heads
    var b = tmp // kv_heads

    var base = ((b * kv_heads + kv_h) * total_len + s) * head_dim
    var max_abs: Float32 = 0.0
    for d in range(head_dim):
        var vval = lt_scalar(v, base + d)
        if vval < 0.0:
            vval = -vval
        if vval > max_abs:
            max_abs = vval

    var qmax_f = Float32(qmax)
    var scale = max_abs / qmax_f
    if scale < eps:
        scale = eps
    v_scales[(b * kv_heads + kv_h) * total_len + s] = scale

    var qmin = -qmax - 1
    for d in range(head_dim):
        var vval = lt_scalar(v, base + d)
        var q = round_to_int(vval / scale)
        if q < qmin:
            q = qmin
        if q > qmax:
            q = qmax
        v_q[base + d] = Int32(q + zero)


@export("mixkvq_streaming_fused_host", ABI="C")
fn mixkvq_streaming_fused_host(
    q_ptr: UnsafePointer[Float32, MutOrigin.external],
    k_q_ptr: UnsafePointer[Int32, MutOrigin.external],
    v_q_ptr: UnsafePointer[Int32, MutOrigin.external],
    k_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    v_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    offsets_ptr: UnsafePointer[Int32, MutOrigin.external],
    lengths_ptr: UnsafePointer[Int32, MutOrigin.external],
    num_chunks: Int,
    total_len: Int,
    batch: Int,
    heads: Int,
    q_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
    scaling: Float32,
    zero: Int,
    mask_ptr: UnsafePointer[Float32, MutOrigin.external],
    use_mask: Int,
    out_ptr: MutUnsafePointer[Float32, MutOrigin.external],
) -> Int:
    try:
        var kv_heads = heads // num_key_value_groups
        var q_size = batch * heads * q_len * head_dim
        var kv_size = batch * kv_heads * total_len * head_dim
        var k_scale_size = batch * num_chunks * kv_heads * head_dim
        var v_scale_size = batch * kv_heads * total_len
        var mask_size = batch * q_len * total_len if use_mask != 0 else 1

        var ctx = DeviceContext()
        var host_q = ctx.enqueue_create_host_buffer[float_dtype](q_size)
        var host_kq = ctx.enqueue_create_host_buffer[q_dtype](kv_size)
        var host_vq = ctx.enqueue_create_host_buffer[q_dtype](kv_size)
        var host_k_scales = ctx.enqueue_create_host_buffer[float_dtype](k_scale_size)
        var host_v_scales = ctx.enqueue_create_host_buffer[float_dtype](v_scale_size)
        var host_offsets = ctx.enqueue_create_host_buffer[DType.int32](num_chunks)
        var host_lengths = ctx.enqueue_create_host_buffer[DType.int32](num_chunks)
        var host_mask = ctx.enqueue_create_host_buffer[float_dtype](mask_size)
        var host_out = ctx.enqueue_create_host_buffer[float_dtype](q_size)

        for i in range(q_size):
            host_q.unsafe_ptr().offset(i).store(q_ptr.offset(i).load())
        for i in range(kv_size):
            host_kq.unsafe_ptr().offset(i).store(k_q_ptr.offset(i).load())
            host_vq.unsafe_ptr().offset(i).store(v_q_ptr.offset(i).load())
        for i in range(k_scale_size):
            host_k_scales.unsafe_ptr().offset(i).store(k_scales_ptr.offset(i).load())
        for i in range(v_scale_size):
            host_v_scales.unsafe_ptr().offset(i).store(v_scales_ptr.offset(i).load())
        for i in range(num_chunks):
            host_offsets.unsafe_ptr().offset(i).store(offsets_ptr.offset(i).load())
            host_lengths.unsafe_ptr().offset(i).store(lengths_ptr.offset(i).load())
        if use_mask != 0:
            for i in range(mask_size):
                host_mask.unsafe_ptr().offset(i).store(mask_ptr.offset(i).load())
        else:
            host_mask.unsafe_ptr().offset(0).store(0.0)

        var dev_q = ctx.enqueue_create_buffer[float_dtype](q_size)
        var dev_kq = ctx.enqueue_create_buffer[q_dtype](kv_size)
        var dev_vq = ctx.enqueue_create_buffer[q_dtype](kv_size)
        var dev_k_scales = ctx.enqueue_create_buffer[float_dtype](k_scale_size)
        var dev_v_scales = ctx.enqueue_create_buffer[float_dtype](v_scale_size)
        var dev_offsets = ctx.enqueue_create_buffer[DType.int32](num_chunks)
        var dev_lengths = ctx.enqueue_create_buffer[DType.int32](num_chunks)
        var dev_mask = ctx.enqueue_create_buffer[float_dtype](mask_size)
        var dev_out = ctx.enqueue_create_buffer[float_dtype](q_size)

        ctx.enqueue_copy(dst_buf=dev_q, src_buf=host_q)
        ctx.enqueue_copy(dst_buf=dev_kq, src_buf=host_kq)
        ctx.enqueue_copy(dst_buf=dev_vq, src_buf=host_vq)
        ctx.enqueue_copy(dst_buf=dev_k_scales, src_buf=host_k_scales)
        ctx.enqueue_copy(dst_buf=dev_v_scales, src_buf=host_v_scales)
        ctx.enqueue_copy(dst_buf=dev_offsets, src_buf=host_offsets)
        ctx.enqueue_copy(dst_buf=dev_lengths, src_buf=host_lengths)
        ctx.enqueue_copy(dst_buf=dev_mask, src_buf=host_mask)
        ctx.synchronize()

        var q_tensor = LayoutTensor[float_dtype, layout](dev_q)
        var kq_tensor = LayoutTensor[q_dtype, layout](dev_kq)
        var vq_tensor = LayoutTensor[q_dtype, layout](dev_vq)
        var k_scales_tensor = LayoutTensor[float_dtype, layout](dev_k_scales)
        var v_scales_tensor = LayoutTensor[float_dtype, layout](dev_v_scales)
        var offsets_tensor = LayoutTensor[DType.int32, layout](dev_offsets)
        var lengths_tensor = LayoutTensor[DType.int32, layout](dev_lengths)
        var mask_tensor = LayoutTensor[float_dtype, layout](dev_mask)
        var out_tensor = LayoutTensor[float_dtype, layout](dev_out)

        var block_size = 128
        var total_threads = batch * heads * q_len
        var num_blocks = (total_threads + block_size - 1) // block_size

        ctx.enqueue_function_checked[streaming_attention_gqa_kernel_fused, streaming_attention_gqa_kernel_fused](
            q_tensor,
            kq_tensor,
            vq_tensor,
            k_scales_tensor,
            v_scales_tensor,
            mask_tensor,
            offsets_tensor,
            lengths_tensor,
            use_mask,
            num_chunks,
            total_len,
            out_tensor,
            batch,
            heads,
            q_len,
            head_dim,
            num_key_value_groups,
            scaling,
            zero,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=host_out, src_buf=dev_out)
        ctx.synchronize()

        for i in range(q_size):
            out_ptr.offset(i).store(host_out.unsafe_ptr().offset(i).load())
    except e:
        return -1

    return 0


@export("mixkvq_streaming_fused_host_fp16", ABI="C")
fn mixkvq_streaming_fused_host_fp16(
    q_ptr: UnsafePointer[Float32, MutOrigin.external],
    k_ptr: UnsafePointer[Float32, MutOrigin.external],
    v_ptr: UnsafePointer[Float32, MutOrigin.external],
    offsets_ptr: UnsafePointer[Int32, MutOrigin.external],
    lengths_ptr: UnsafePointer[Int32, MutOrigin.external],
    num_chunks: Int,
    total_len: Int,
    batch: Int,
    heads: Int,
    q_len: Int,
    head_dim: Int,
    num_key_value_groups: Int,
    scaling: Float32,
    zero: Int,
    mask_ptr: UnsafePointer[Float32, MutOrigin.external],
    use_mask: Int,
    out_ptr: MutUnsafePointer[Float32, MutOrigin.external],
) -> Int:
    try:
        var kv_heads = heads // num_key_value_groups
        var q_size = batch * heads * q_len * head_dim
        var kv_size = batch * kv_heads * total_len * head_dim
        var k_scale_size = batch * num_chunks * kv_heads * head_dim
        var v_scale_size = batch * kv_heads * total_len
        var mask_size = batch * q_len * total_len if use_mask != 0 else 1
        var qmax = 7
        var eps: Float32 = 1.0e-6

        var ctx = DeviceContext()
        var host_q = ctx.enqueue_create_host_buffer[float_dtype](q_size)
        var host_k = ctx.enqueue_create_host_buffer[float_dtype](kv_size)
        var host_v = ctx.enqueue_create_host_buffer[float_dtype](kv_size)
        var host_offsets = ctx.enqueue_create_host_buffer[DType.int32](num_chunks)
        var host_lengths = ctx.enqueue_create_host_buffer[DType.int32](num_chunks)
        var host_chunk_ids = ctx.enqueue_create_host_buffer[DType.int32](total_len)
        var host_mask = ctx.enqueue_create_host_buffer[float_dtype](mask_size)
        var host_out = ctx.enqueue_create_host_buffer[float_dtype](q_size)

        for i in range(q_size):
            host_q.unsafe_ptr().offset(i).store(q_ptr.offset(i).load())
        for i in range(kv_size):
            host_k.unsafe_ptr().offset(i).store(k_ptr.offset(i).load())
            host_v.unsafe_ptr().offset(i).store(v_ptr.offset(i).load())
        for i in range(num_chunks):
            host_offsets.unsafe_ptr().offset(i).store(offsets_ptr.offset(i).load())
            host_lengths.unsafe_ptr().offset(i).store(lengths_ptr.offset(i).load())
        if use_mask != 0:
            for i in range(mask_size):
                host_mask.unsafe_ptr().offset(i).store(mask_ptr.offset(i).load())
        else:
            host_mask.unsafe_ptr().offset(0).store(0.0)

        for c in range(num_chunks):
            var offset = Int(host_offsets.unsafe_ptr().offset(c).load())
            var length = Int(host_lengths.unsafe_ptr().offset(c).load())
            for s in range(length):
                host_chunk_ids.unsafe_ptr().offset(offset + s).store(Int32(c))

        var dev_q = ctx.enqueue_create_buffer[float_dtype](q_size)
        var dev_k = ctx.enqueue_create_buffer[float_dtype](kv_size)
        var dev_v = ctx.enqueue_create_buffer[float_dtype](kv_size)
        var dev_k_scales = ctx.enqueue_create_buffer[float_dtype](k_scale_size)
        var dev_v_scales = ctx.enqueue_create_buffer[float_dtype](v_scale_size)
        var dev_kq = ctx.enqueue_create_buffer[q_dtype](kv_size)
        var dev_vq = ctx.enqueue_create_buffer[q_dtype](kv_size)
        var dev_offsets = ctx.enqueue_create_buffer[DType.int32](num_chunks)
        var dev_lengths = ctx.enqueue_create_buffer[DType.int32](num_chunks)
        var dev_chunk_ids = ctx.enqueue_create_buffer[DType.int32](total_len)
        var dev_mask = ctx.enqueue_create_buffer[float_dtype](mask_size)
        var dev_out = ctx.enqueue_create_buffer[float_dtype](q_size)

        ctx.enqueue_copy(dst_buf=dev_q, src_buf=host_q)
        ctx.enqueue_copy(dst_buf=dev_k, src_buf=host_k)
        ctx.enqueue_copy(dst_buf=dev_v, src_buf=host_v)
        ctx.enqueue_copy(dst_buf=dev_offsets, src_buf=host_offsets)
        ctx.enqueue_copy(dst_buf=dev_lengths, src_buf=host_lengths)
        ctx.enqueue_copy(dst_buf=dev_chunk_ids, src_buf=host_chunk_ids)
        ctx.enqueue_copy(dst_buf=dev_mask, src_buf=host_mask)
        ctx.synchronize()

        var q_tensor = LayoutTensor[float_dtype, layout](dev_q)
        var k_tensor = LayoutTensor[float_dtype, layout](dev_k)
        var v_tensor = LayoutTensor[float_dtype, layout](dev_v)
        var k_scales_tensor = LayoutTensor[float_dtype, layout](dev_k_scales)
        var v_scales_tensor = LayoutTensor[float_dtype, layout](dev_v_scales)
        var kq_tensor = LayoutTensor[q_dtype, layout](dev_kq)
        var vq_tensor = LayoutTensor[q_dtype, layout](dev_vq)
        var offsets_tensor = LayoutTensor[DType.int32, layout](dev_offsets)
        var lengths_tensor = LayoutTensor[DType.int32, layout](dev_lengths)
        var chunk_ids_tensor = LayoutTensor[DType.int32, layout](dev_chunk_ids)
        var mask_tensor = LayoutTensor[float_dtype, layout](dev_mask)
        var out_tensor = LayoutTensor[float_dtype, layout](dev_out)

        var block_size = 128
        var total_threads = batch * kv_heads * num_chunks * head_dim
        var num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[compute_k_scales_kernel, compute_k_scales_kernel](
            k_tensor,
            offsets_tensor,
            lengths_tensor,
            num_chunks,
            total_len,
            k_scales_tensor,
            batch,
            kv_heads,
            head_dim,
            qmax,
            eps,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        total_threads = batch * kv_heads * total_len * head_dim
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[quantize_k_kernel, quantize_k_kernel](
            k_tensor,
            k_scales_tensor,
            chunk_ids_tensor,
            num_chunks,
            total_len,
            kq_tensor,
            batch,
            kv_heads,
            head_dim,
            qmax,
            zero,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        total_threads = batch * kv_heads * total_len
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[quantize_v_kernel, quantize_v_kernel](
            v_tensor,
            v_scales_tensor,
            vq_tensor,
            total_len,
            batch,
            kv_heads,
            head_dim,
            qmax,
            zero,
            eps,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
        ctx.synchronize()

        total_threads = batch * heads * q_len
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[streaming_attention_gqa_kernel_fused, streaming_attention_gqa_kernel_fused](
            q_tensor,
            kq_tensor,
            vq_tensor,
            k_scales_tensor,
            v_scales_tensor,
            mask_tensor,
            offsets_tensor,
            lengths_tensor,
            use_mask,
            num_chunks,
            total_len,
            out_tensor,
            batch,
            heads,
            q_len,
            head_dim,
            num_key_value_groups,
            scaling,
            zero,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=host_out, src_buf=dev_out)
        ctx.synchronize()

        for i in range(q_size):
            out_ptr.offset(i).store(host_out.unsafe_ptr().offset(i).load())
    except e:
        return -1

    return 0
