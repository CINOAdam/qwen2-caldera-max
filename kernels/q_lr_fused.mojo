"""
Q + LR fused prototype (CPU) in Mojo v0.26+.

This is a research-grade baseline for a fused kernel:
  y = Q x + L (R x)

Quantization is symmetric, groupwise per-row.

Usage:
  mojo kernels/q_lr_fused.mojo
"""

from memory import UnsafePointer, MutUnsafePointer, alloc
from random import random_float64
from time import perf_counter_ns


fn ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


fn abs_f(value: Float32) -> Float32:
    if value < 0:
        return -value
    return value


fn round_to_int(value: Float32) -> Int32:
    if value >= 0:
        return Int32(value + 0.5)
    return Int32(value - 0.5)


fn fill_random(ptr: MutUnsafePointer[Float32], size: Int, scale: Float32):
    for i in range(size):
        var v = random_float64().cast[DType.float32]()
        v = v * 2.0 - 1.0
        ptr[i] = v * scale


fn quantize_matrix(
    input: UnsafePointer[Float32],
    rows: Int,
    cols: Int,
    group_size: Int,
    num_bits: Int,
) -> Tuple[UnsafePointer[Int32], UnsafePointer[Float32], Int]:
    var groups = ceil_div(cols, group_size)
    var values = alloc[Int32](rows * cols)
    var scales = alloc[Float32](rows * groups)

    var qmax = (1 << (num_bits - 1)) - 1
    var qmin = -qmax - 1

    for r in range(rows):
        var row_base = r * cols
        for g in range(groups):
            var start = g * group_size
            var end = start + group_size
            if end > cols:
                end = cols

            var max_abs: Float32 = 0.0
            for c in range(start, end):
                var v = input[row_base + c]
                var av = abs_f(v)
                if av > max_abs:
                    max_abs = av

            var scale: Float32 = 1.0
            if max_abs > 0.0:
                scale = max_abs / Float32(qmax)
            scales[r * groups + g] = scale

            for c in range(start, end):
                var v = input[row_base + c]
                var q = round_to_int(v / scale)
                if q < qmin:
                    q = qmin
                elif q > qmax:
                    q = qmax
                values[row_base + c] = q

    return (values, scales, groups)


fn fused_q_lr(
    q_vals: UnsafePointer[Int32],
    q_scales: UnsafePointer[Float32],
    q_groups: Int,
    l_vals: UnsafePointer[Int32],
    l_scales: UnsafePointer[Float32],
    l_groups: Int,
    r_vals: UnsafePointer[Int32],
    r_scales: UnsafePointer[Float32],
    r_groups: Int,
    rows: Int,
    cols: Int,
    rank: Int,
    group_size: Int,
    x: UnsafePointer[Float32],
    out: MutUnsafePointer[Float32],
):
    var rx = alloc[Float32](rank)

    # Compute R x
    for k in range(rank):
        var sum: Float32 = 0.0
        var row_base = k * cols
        for c in range(cols):
            var g = c // group_size
            var scale = r_scales[k * r_groups + g]
            var qv = r_vals[row_base + c]
            sum += Float32(qv) * scale * x[c]
        rx[k] = sum

    # Compute Q x + L (R x)
    for r in range(rows):
        var sum_q: Float32 = 0.0
        var row_base = r * cols
        for c in range(cols):
            var g = c // group_size
            var scale = q_scales[r * q_groups + g]
            var qv = q_vals[row_base + c]
            sum_q += Float32(qv) * scale * x[c]

        var sum_l: Float32 = 0.0
        var l_row_base = r * rank
        for k in range(rank):
            var g = k // group_size
            var scale = l_scales[r * l_groups + g]
            var qv = l_vals[l_row_base + k]
            sum_l += Float32(qv) * scale * rx[k]

        out[r] = sum_q + sum_l

    rx.free()


fn benchmark():
    var rows = 1024
    var cols = 1024
    var rank = 128
    var group_size = 128
    var bits_q = 2
    var bits_lr = 4
    var iters = 5

    var q = alloc[Float32](rows * cols)
    var l = alloc[Float32](rows * rank)
    var r = alloc[Float32](rank * cols)
    var x = alloc[Float32](cols)
    var out = alloc[Float32](rows)

    fill_random(q, rows * cols, 0.02)
    fill_random(l, rows * rank, 0.02)
    fill_random(r, rank * cols, 0.02)
    fill_random(x, cols, 1.0)

    var q_vals: UnsafePointer[Int32]
    var q_scales: UnsafePointer[Float32]
    var q_groups: Int
    (q_vals, q_scales, q_groups) = quantize_matrix(q, rows, cols, group_size, bits_q)

    var l_vals: UnsafePointer[Int32]
    var l_scales: UnsafePointer[Float32]
    var l_groups: Int
    (l_vals, l_scales, l_groups) = quantize_matrix(l, rows, rank, group_size, bits_lr)

    var r_vals: UnsafePointer[Int32]
    var r_scales: UnsafePointer[Float32]
    var r_groups: Int
    (r_vals, r_scales, r_groups) = quantize_matrix(r, rank, cols, group_size, bits_lr)

    # Warmup
    fused_q_lr(
        q_vals,
        q_scales,
        q_groups,
        l_vals,
        l_scales,
        l_groups,
        r_vals,
        r_scales,
        r_groups,
        rows,
        cols,
        rank,
        group_size,
        x,
        out,
    )

    var start = perf_counter_ns()
    for _ in range(iters):
        fused_q_lr(
            q_vals,
            q_scales,
            q_groups,
            l_vals,
            l_scales,
            l_groups,
            r_vals,
            r_scales,
            r_groups,
            rows,
            cols,
            rank,
            group_size,
            x,
            out,
        )
    var elapsed_ns = perf_counter_ns() - start
    var elapsed_s = Float64(elapsed_ns) / 1_000_000_000.0

    print("Q + LR fused prototype (CPU)")
    print("Rows:", rows, "Cols:", cols, "Rank:", rank)
    print("Group size:", group_size, "bits Q:", bits_q, "bits LR:", bits_lr)
    print("Iterations:", iters)
    print("Total time:", elapsed_s, "s")
    print("Per iter:", elapsed_s / iters * 1000, "ms")

    q.free()
    l.free()
    r.free()
    x.free()
    out.free()
    q_vals.free()
    q_scales.free()
    l_vals.free()
    l_scales.free()
    r_vals.free()
    r_scales.free()


fn main():
    benchmark()
