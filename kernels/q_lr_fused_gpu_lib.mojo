"""
Q + LR fused kernel (GPU) for shared-lib build.

Exports q_lr_fused_host for Python ctypes integration.
"""

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from memory import UnsafePointer, MutUnsafePointer


comptime float_dtype = DType.float32
comptime q_dtype = DType.int32
comptime layout = Layout.row_major(1)
comptime tile_size = 128


fn ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


fn lt_scalar(t: LayoutTensor[float_dtype, layout, MutAnyOrigin], idx: Int) -> Float32:
    return t[idx].reduce_add()


fn lt_scalar_i32(t: LayoutTensor[q_dtype, layout, MutAnyOrigin], idx: Int) -> Int:
    return Int(t[idx].reduce_add())


fn compute_rx_kernel(
    r_vals: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    r_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    x: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rx: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rank: Int,
    cols: Int,
    r_groups: Int,
    group_size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= UInt(rank):
        return

    var k = Int(tid)
    var sum: Float32 = 0.0
    var row_base = k * cols
    for g in range(r_groups):
        var start = g * group_size
        if start >= cols:
            break
        var end = start + group_size
        if end > cols:
            end = cols
        var scale = lt_scalar(r_scales, k * r_groups + g)
        for c in range(start, end):
            var qv = lt_scalar_i32(r_vals, row_base + c)
            sum += Float32(qv) * scale * lt_scalar(x, c)
    rx[k] = sum


fn compute_output_kernel(
    q_vals: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    q_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    l_vals: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    l_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    x: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rx: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    output: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rows: Int,
    cols: Int,
    rank: Int,
    q_groups: Int,
    l_groups: Int,
    group_size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= UInt(rows):
        return

    var r = Int(tid)
    var sum_q: Float32 = 0.0
    var row_base = r * cols
    for g in range(q_groups):
        var start = g * group_size
        if start >= cols:
            break
        var end = start + group_size
        if end > cols:
            end = cols
        var scale = lt_scalar(q_scales, r * q_groups + g)
        for c in range(start, end):
            var qv = lt_scalar_i32(q_vals, row_base + c)
            sum_q += Float32(qv) * scale * lt_scalar(x, c)

    var sum_l: Float32 = 0.0
    var l_row_base = r * rank
    for g in range(l_groups):
        var start = g * group_size
        if start >= rank:
            break
        var end = start + group_size
        if end > rank:
            end = rank
        var scale = lt_scalar(l_scales, r * l_groups + g)
        for k in range(start, end):
            var qv = lt_scalar_i32(l_vals, l_row_base + k)
            sum_l += Float32(qv) * scale * lt_scalar(rx, k)

    output[r] = sum_q + sum_l


fn compute_qx_tile_kernel(
    q_vals: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    q_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    x: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    q_tiles: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rows: Int,
    cols: Int,
    q_groups: Int,
    group_size: Int,
    tile_cols: Int,
    num_tiles: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = rows * num_tiles
    if tid >= UInt(total_threads):
        return

    var tile = Int(tid % UInt(num_tiles))
    var r = Int(tid // UInt(num_tiles))
    var start = tile * tile_cols
    var end = start + tile_cols
    if end > cols:
        end = cols

    var sum_q: Float32 = 0.0
    var row_base = r * cols
    var g_start = start // group_size
    var g_end = (end - 1) // group_size
    for g in range(g_start, g_end + 1):
        var g_start_c = g * group_size
        var g_end_c = g_start_c + group_size
        if g_start_c < start:
            g_start_c = start
        if g_end_c > end:
            g_end_c = end
        var scale = lt_scalar(q_scales, r * q_groups + g)
        for c in range(g_start_c, g_end_c):
            var qv = lt_scalar_i32(q_vals, row_base + c)
            sum_q += Float32(qv) * scale * lt_scalar(x, c)

    q_tiles[r * num_tiles + tile] = sum_q


fn compute_lrx_tile_kernel(
    l_vals: LayoutTensor[q_dtype, layout, MutAnyOrigin],
    l_scales: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rx: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    l_tiles: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rows: Int,
    rank: Int,
    l_groups: Int,
    group_size: Int,
    tile_rank: Int,
    num_tiles: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = rows * num_tiles
    if tid >= UInt(total_threads):
        return

    var tile = Int(tid % UInt(num_tiles))
    var r = Int(tid // UInt(num_tiles))
    var start = tile * tile_rank
    var end = start + tile_rank
    if end > rank:
        end = rank

    var sum_l: Float32 = 0.0
    var row_base = r * rank
    var g_start = start // group_size
    var g_end = (end - 1) // group_size
    for g in range(g_start, g_end + 1):
        var g_start_k = g * group_size
        var g_end_k = g_start_k + group_size
        if g_start_k < start:
            g_start_k = start
        if g_end_k > end:
            g_end_k = end
        var scale = lt_scalar(l_scales, r * l_groups + g)
        for k in range(g_start_k, g_end_k):
            var qv = lt_scalar_i32(l_vals, row_base + k)
            sum_l += Float32(qv) * scale * lt_scalar(rx, k)

    l_tiles[r * num_tiles + tile] = sum_l


fn reduce_tiles_kernel(
    q_tiles: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    l_tiles: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    output: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rows: Int,
    q_tiles_count: Int,
    l_tiles_count: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= UInt(rows):
        return

    var r = Int(tid)
    var sum: Float32 = 0.0
    for t in range(q_tiles_count):
        sum += lt_scalar(q_tiles, r * q_tiles_count + t)
    for t in range(l_tiles_count):
        sum += lt_scalar(l_tiles, r * l_tiles_count + t)
    output[r] = sum




@export("q_lr_fused_host", ABI="C")
fn q_lr_fused_host(
    q_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    q_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    l_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    l_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    r_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    r_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    x_ptr: UnsafePointer[Float32, MutOrigin.external],
    rows: Int,
    cols: Int,
    rank: Int,
    group_size: Int,
    q_groups: Int,
    l_groups: Int,
    r_groups: Int,
    out_ptr: MutUnsafePointer[Float32, MutOrigin.external],
) -> Int:
    try:
        var q_size = rows * cols
        var l_size = rows * rank
        var r_size = rank * cols
        var x_size = cols
        var q_scales_size = rows * q_groups
        var l_scales_size = rows * l_groups
        var r_scales_size = rank * r_groups
        var out_size = rows

        var ctx = DeviceContext()
        var host_q_vals = ctx.enqueue_create_host_buffer[q_dtype](q_size)
        var host_q_scales = ctx.enqueue_create_host_buffer[float_dtype](q_scales_size)
        var host_l_vals = ctx.enqueue_create_host_buffer[q_dtype](l_size)
        var host_l_scales = ctx.enqueue_create_host_buffer[float_dtype](l_scales_size)
        var host_r_vals = ctx.enqueue_create_host_buffer[q_dtype](r_size)
        var host_r_scales = ctx.enqueue_create_host_buffer[float_dtype](r_scales_size)
        var host_x = ctx.enqueue_create_host_buffer[float_dtype](x_size)
        var host_out = ctx.enqueue_create_host_buffer[float_dtype](out_size)

        for i in range(q_size):
            host_q_vals.unsafe_ptr().offset(i).store(q_vals_ptr.offset(i).load())
        for i in range(q_scales_size):
            host_q_scales.unsafe_ptr().offset(i).store(q_scales_ptr.offset(i).load())
        for i in range(l_size):
            host_l_vals.unsafe_ptr().offset(i).store(l_vals_ptr.offset(i).load())
        for i in range(l_scales_size):
            host_l_scales.unsafe_ptr().offset(i).store(l_scales_ptr.offset(i).load())
        for i in range(r_size):
            host_r_vals.unsafe_ptr().offset(i).store(r_vals_ptr.offset(i).load())
        for i in range(r_scales_size):
            host_r_scales.unsafe_ptr().offset(i).store(r_scales_ptr.offset(i).load())
        for i in range(x_size):
            host_x.unsafe_ptr().offset(i).store(x_ptr.offset(i).load())

        var dev_q_vals = ctx.enqueue_create_buffer[q_dtype](q_size)
        var dev_q_scales = ctx.enqueue_create_buffer[float_dtype](q_scales_size)
        var dev_l_vals = ctx.enqueue_create_buffer[q_dtype](l_size)
        var dev_l_scales = ctx.enqueue_create_buffer[float_dtype](l_scales_size)
        var dev_r_vals = ctx.enqueue_create_buffer[q_dtype](r_size)
        var dev_r_scales = ctx.enqueue_create_buffer[float_dtype](r_scales_size)
        var dev_x = ctx.enqueue_create_buffer[float_dtype](x_size)
        var dev_rx = ctx.enqueue_create_buffer[float_dtype](rank)
        var dev_out = ctx.enqueue_create_buffer[float_dtype](out_size)

        ctx.enqueue_copy(dst_buf=dev_q_vals, src_buf=host_q_vals)
        ctx.enqueue_copy(dst_buf=dev_q_scales, src_buf=host_q_scales)
        ctx.enqueue_copy(dst_buf=dev_l_vals, src_buf=host_l_vals)
        ctx.enqueue_copy(dst_buf=dev_l_scales, src_buf=host_l_scales)
        ctx.enqueue_copy(dst_buf=dev_r_vals, src_buf=host_r_vals)
        ctx.enqueue_copy(dst_buf=dev_r_scales, src_buf=host_r_scales)
        ctx.enqueue_copy(dst_buf=dev_x, src_buf=host_x)
        ctx.synchronize()

        var q_vals_tensor = LayoutTensor[q_dtype, layout](dev_q_vals)
        var q_scales_tensor = LayoutTensor[float_dtype, layout](dev_q_scales)
        var l_vals_tensor = LayoutTensor[q_dtype, layout](dev_l_vals)
        var l_scales_tensor = LayoutTensor[float_dtype, layout](dev_l_scales)
        var r_vals_tensor = LayoutTensor[q_dtype, layout](dev_r_vals)
        var r_scales_tensor = LayoutTensor[float_dtype, layout](dev_r_scales)
        var x_tensor = LayoutTensor[float_dtype, layout](dev_x)
        var rx_tensor = LayoutTensor[float_dtype, layout](dev_rx)
        var out_tensor = LayoutTensor[float_dtype, layout](dev_out)

        var block_size = 128
        var total_threads = rank
        var num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[compute_rx_kernel, compute_rx_kernel](
            r_vals_tensor,
            r_scales_tensor,
            x_tensor,
            rx_tensor,
            rank,
            cols,
            r_groups,
            group_size,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        total_threads = rows
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[compute_output_kernel, compute_output_kernel](
            q_vals_tensor,
            q_scales_tensor,
            l_vals_tensor,
            l_scales_tensor,
            x_tensor,
            rx_tensor,
            out_tensor,
            rows,
            cols,
            rank,
            q_groups,
            l_groups,
            group_size,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=host_out, src_buf=dev_out)
        ctx.synchronize()

        for i in range(out_size):
            out_ptr.offset(i).store(host_out.unsafe_ptr().offset(i).load())
    except e:
        return -1

    return 0


@export("q_lr_fused_host_tiled", ABI="C")
fn q_lr_fused_host_tiled(
    q_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    q_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    l_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    l_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    r_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    r_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    x_ptr: UnsafePointer[Float32, MutOrigin.external],
    rows: Int,
    cols: Int,
    rank: Int,
    group_size: Int,
    q_groups: Int,
    l_groups: Int,
    r_groups: Int,
    out_ptr: MutUnsafePointer[Float32, MutOrigin.external],
) -> Int:
    try:
        var q_size = rows * cols
        var l_size = rows * rank
        var r_size = rank * cols
        var x_size = cols
        var q_scales_size = rows * q_groups
        var l_scales_size = rows * l_groups
        var r_scales_size = rank * r_groups
        var out_size = rows

        var q_tiles_count = ceil_div(cols, tile_size)
        var l_tiles_count = ceil_div(rank, tile_size)
        var q_tiles_size = rows * q_tiles_count
        var l_tiles_size = rows * l_tiles_count

        var ctx = DeviceContext()
        var host_q_vals = ctx.enqueue_create_host_buffer[q_dtype](q_size)
        var host_q_scales = ctx.enqueue_create_host_buffer[float_dtype](q_scales_size)
        var host_l_vals = ctx.enqueue_create_host_buffer[q_dtype](l_size)
        var host_l_scales = ctx.enqueue_create_host_buffer[float_dtype](l_scales_size)
        var host_r_vals = ctx.enqueue_create_host_buffer[q_dtype](r_size)
        var host_r_scales = ctx.enqueue_create_host_buffer[float_dtype](r_scales_size)
        var host_x = ctx.enqueue_create_host_buffer[float_dtype](x_size)
        var host_out = ctx.enqueue_create_host_buffer[float_dtype](out_size)

        for i in range(q_size):
            host_q_vals.unsafe_ptr().offset(i).store(q_vals_ptr.offset(i).load())
        for i in range(q_scales_size):
            host_q_scales.unsafe_ptr().offset(i).store(q_scales_ptr.offset(i).load())
        for i in range(l_size):
            host_l_vals.unsafe_ptr().offset(i).store(l_vals_ptr.offset(i).load())
        for i in range(l_scales_size):
            host_l_scales.unsafe_ptr().offset(i).store(l_scales_ptr.offset(i).load())
        for i in range(r_size):
            host_r_vals.unsafe_ptr().offset(i).store(r_vals_ptr.offset(i).load())
        for i in range(r_scales_size):
            host_r_scales.unsafe_ptr().offset(i).store(r_scales_ptr.offset(i).load())
        for i in range(x_size):
            host_x.unsafe_ptr().offset(i).store(x_ptr.offset(i).load())

        var dev_q_vals = ctx.enqueue_create_buffer[q_dtype](q_size)
        var dev_q_scales = ctx.enqueue_create_buffer[float_dtype](q_scales_size)
        var dev_l_vals = ctx.enqueue_create_buffer[q_dtype](l_size)
        var dev_l_scales = ctx.enqueue_create_buffer[float_dtype](l_scales_size)
        var dev_r_vals = ctx.enqueue_create_buffer[q_dtype](r_size)
        var dev_r_scales = ctx.enqueue_create_buffer[float_dtype](r_scales_size)
        var dev_x = ctx.enqueue_create_buffer[float_dtype](x_size)
        var dev_rx = ctx.enqueue_create_buffer[float_dtype](rank)
        var dev_q_tiles = ctx.enqueue_create_buffer[float_dtype](q_tiles_size)
        var dev_l_tiles = ctx.enqueue_create_buffer[float_dtype](l_tiles_size)
        var dev_out = ctx.enqueue_create_buffer[float_dtype](out_size)

        ctx.enqueue_copy(dst_buf=dev_q_vals, src_buf=host_q_vals)
        ctx.enqueue_copy(dst_buf=dev_q_scales, src_buf=host_q_scales)
        ctx.enqueue_copy(dst_buf=dev_l_vals, src_buf=host_l_vals)
        ctx.enqueue_copy(dst_buf=dev_l_scales, src_buf=host_l_scales)
        ctx.enqueue_copy(dst_buf=dev_r_vals, src_buf=host_r_vals)
        ctx.enqueue_copy(dst_buf=dev_r_scales, src_buf=host_r_scales)
        ctx.enqueue_copy(dst_buf=dev_x, src_buf=host_x)
        ctx.synchronize()

        var q_vals_tensor = LayoutTensor[q_dtype, layout](dev_q_vals)
        var q_scales_tensor = LayoutTensor[float_dtype, layout](dev_q_scales)
        var l_vals_tensor = LayoutTensor[q_dtype, layout](dev_l_vals)
        var l_scales_tensor = LayoutTensor[float_dtype, layout](dev_l_scales)
        var r_vals_tensor = LayoutTensor[q_dtype, layout](dev_r_vals)
        var r_scales_tensor = LayoutTensor[float_dtype, layout](dev_r_scales)
        var x_tensor = LayoutTensor[float_dtype, layout](dev_x)
        var rx_tensor = LayoutTensor[float_dtype, layout](dev_rx)
        var q_tiles_tensor = LayoutTensor[float_dtype, layout](dev_q_tiles)
        var l_tiles_tensor = LayoutTensor[float_dtype, layout](dev_l_tiles)
        var out_tensor = LayoutTensor[float_dtype, layout](dev_out)

        var block_size = 128
        var total_threads = rank
        var num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[compute_rx_kernel, compute_rx_kernel](
            r_vals_tensor,
            r_scales_tensor,
            x_tensor,
            rx_tensor,
            rank,
            cols,
            r_groups,
            group_size,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        total_threads = rows * q_tiles_count
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[compute_qx_tile_kernel, compute_qx_tile_kernel](
            q_vals_tensor,
            q_scales_tensor,
            x_tensor,
            q_tiles_tensor,
            rows,
            cols,
            q_groups,
            group_size,
            tile_size,
            q_tiles_count,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        total_threads = rows * l_tiles_count
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[compute_lrx_tile_kernel, compute_lrx_tile_kernel](
            l_vals_tensor,
            l_scales_tensor,
            rx_tensor,
            l_tiles_tensor,
            rows,
            rank,
            l_groups,
            group_size,
            tile_size,
            l_tiles_count,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        total_threads = rows
        num_blocks = (total_threads + block_size - 1) // block_size
        ctx.enqueue_function_checked[reduce_tiles_kernel, reduce_tiles_kernel](
            q_tiles_tensor,
            l_tiles_tensor,
            out_tensor,
            rows,
            q_tiles_count,
            l_tiles_count,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=host_out, src_buf=dev_out)
        ctx.synchronize()

        for i in range(out_size):
            out_ptr.offset(i).store(host_out.unsafe_ptr().offset(i).load())
    except e:
        return -1

    return 0


@export("q_lr_fused_host_batched", ABI="C")
fn q_lr_fused_host_batched(
    q_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    q_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    l_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    l_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    r_vals_ptr: UnsafePointer[Int32, MutOrigin.external],
    r_scales_ptr: UnsafePointer[Float32, MutOrigin.external],
    x_ptr: UnsafePointer[Float32, MutOrigin.external],
    rows: Int,
    cols: Int,
    rank: Int,
    group_size: Int,
    q_groups: Int,
    l_groups: Int,
    r_groups: Int,
    iters: Int,
    out_ptr: MutUnsafePointer[Float32, MutOrigin.external],
) -> Int:
    try:
        if iters < 1:
            return -1

        var q_size = rows * cols
        var l_size = rows * rank
        var r_size = rank * cols
        var x_size = cols
        var q_scales_size = rows * q_groups
        var l_scales_size = rows * l_groups
        var r_scales_size = rank * r_groups
        var out_size = rows

        var q_tiles_count = ceil_div(cols, tile_size)
        var l_tiles_count = ceil_div(rank, tile_size)
        var q_tiles_size = rows * q_tiles_count
        var l_tiles_size = rows * l_tiles_count

        var ctx = DeviceContext()
        var host_q_vals = ctx.enqueue_create_host_buffer[q_dtype](q_size)
        var host_q_scales = ctx.enqueue_create_host_buffer[float_dtype](q_scales_size)
        var host_l_vals = ctx.enqueue_create_host_buffer[q_dtype](l_size)
        var host_l_scales = ctx.enqueue_create_host_buffer[float_dtype](l_scales_size)
        var host_r_vals = ctx.enqueue_create_host_buffer[q_dtype](r_size)
        var host_r_scales = ctx.enqueue_create_host_buffer[float_dtype](r_scales_size)
        var host_x = ctx.enqueue_create_host_buffer[float_dtype](x_size)
        var host_out = ctx.enqueue_create_host_buffer[float_dtype](out_size)

        for i in range(q_size):
            host_q_vals.unsafe_ptr().offset(i).store(q_vals_ptr.offset(i).load())
        for i in range(q_scales_size):
            host_q_scales.unsafe_ptr().offset(i).store(q_scales_ptr.offset(i).load())
        for i in range(l_size):
            host_l_vals.unsafe_ptr().offset(i).store(l_vals_ptr.offset(i).load())
        for i in range(l_scales_size):
            host_l_scales.unsafe_ptr().offset(i).store(l_scales_ptr.offset(i).load())
        for i in range(r_size):
            host_r_vals.unsafe_ptr().offset(i).store(r_vals_ptr.offset(i).load())
        for i in range(r_scales_size):
            host_r_scales.unsafe_ptr().offset(i).store(r_scales_ptr.offset(i).load())
        for i in range(x_size):
            host_x.unsafe_ptr().offset(i).store(x_ptr.offset(i).load())

        var dev_q_vals = ctx.enqueue_create_buffer[q_dtype](q_size)
        var dev_q_scales = ctx.enqueue_create_buffer[float_dtype](q_scales_size)
        var dev_l_vals = ctx.enqueue_create_buffer[q_dtype](l_size)
        var dev_l_scales = ctx.enqueue_create_buffer[float_dtype](l_scales_size)
        var dev_r_vals = ctx.enqueue_create_buffer[q_dtype](r_size)
        var dev_r_scales = ctx.enqueue_create_buffer[float_dtype](r_scales_size)
        var dev_x = ctx.enqueue_create_buffer[float_dtype](x_size)
        var dev_rx = ctx.enqueue_create_buffer[float_dtype](rank)
        var dev_q_tiles = ctx.enqueue_create_buffer[float_dtype](q_tiles_size)
        var dev_l_tiles = ctx.enqueue_create_buffer[float_dtype](l_tiles_size)
        var dev_out = ctx.enqueue_create_buffer[float_dtype](out_size)

        ctx.enqueue_copy(dst_buf=dev_q_vals, src_buf=host_q_vals)
        ctx.enqueue_copy(dst_buf=dev_q_scales, src_buf=host_q_scales)
        ctx.enqueue_copy(dst_buf=dev_l_vals, src_buf=host_l_vals)
        ctx.enqueue_copy(dst_buf=dev_l_scales, src_buf=host_l_scales)
        ctx.enqueue_copy(dst_buf=dev_r_vals, src_buf=host_r_vals)
        ctx.enqueue_copy(dst_buf=dev_r_scales, src_buf=host_r_scales)
        ctx.enqueue_copy(dst_buf=dev_x, src_buf=host_x)
        ctx.synchronize()

        var q_vals_tensor = LayoutTensor[q_dtype, layout](dev_q_vals)
        var q_scales_tensor = LayoutTensor[float_dtype, layout](dev_q_scales)
        var l_vals_tensor = LayoutTensor[q_dtype, layout](dev_l_vals)
        var l_scales_tensor = LayoutTensor[float_dtype, layout](dev_l_scales)
        var r_vals_tensor = LayoutTensor[q_dtype, layout](dev_r_vals)
        var r_scales_tensor = LayoutTensor[float_dtype, layout](dev_r_scales)
        var x_tensor = LayoutTensor[float_dtype, layout](dev_x)
        var rx_tensor = LayoutTensor[float_dtype, layout](dev_rx)
        var q_tiles_tensor = LayoutTensor[float_dtype, layout](dev_q_tiles)
        var l_tiles_tensor = LayoutTensor[float_dtype, layout](dev_l_tiles)
        var out_tensor = LayoutTensor[float_dtype, layout](dev_out)

        var block_size = 128
        for _ in range(iters):
            var total_threads = rank
            var num_blocks = (total_threads + block_size - 1) // block_size
            ctx.enqueue_function_checked[compute_rx_kernel, compute_rx_kernel](
                r_vals_tensor,
                r_scales_tensor,
                x_tensor,
                rx_tensor,
                rank,
                cols,
                r_groups,
                group_size,
                grid_dim=num_blocks,
                block_dim=block_size,
            )

            total_threads = rows * q_tiles_count
            num_blocks = (total_threads + block_size - 1) // block_size
            ctx.enqueue_function_checked[compute_qx_tile_kernel, compute_qx_tile_kernel](
                q_vals_tensor,
                q_scales_tensor,
                x_tensor,
                q_tiles_tensor,
                rows,
                cols,
                q_groups,
                group_size,
                tile_size,
                q_tiles_count,
                grid_dim=num_blocks,
                block_dim=block_size,
            )

            total_threads = rows * l_tiles_count
            num_blocks = (total_threads + block_size - 1) // block_size
            ctx.enqueue_function_checked[compute_lrx_tile_kernel, compute_lrx_tile_kernel](
                l_vals_tensor,
                l_scales_tensor,
                rx_tensor,
                l_tiles_tensor,
                rows,
                rank,
                l_groups,
                group_size,
                tile_size,
                l_tiles_count,
                grid_dim=num_blocks,
                block_dim=block_size,
            )

            total_threads = rows
            num_blocks = (total_threads + block_size - 1) // block_size
            ctx.enqueue_function_checked[reduce_tiles_kernel, reduce_tiles_kernel](
                q_tiles_tensor,
                l_tiles_tensor,
                out_tensor,
                rows,
                q_tiles_count,
                l_tiles_count,
                grid_dim=num_blocks,
                block_dim=block_size,
            )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=host_out, src_buf=dev_out)
        ctx.synchronize()

        for i in range(out_size):
            out_ptr.offset(i).store(host_out.unsafe_ptr().offset(i).load())
    except e:
        return -1

    return 0
