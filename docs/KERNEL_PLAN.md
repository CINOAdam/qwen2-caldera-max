# Mojo/MAX Kernel Plan

## Target op

Compute:

  y = Q x + L (R x)

Where Q, L, R are quantized. The kernel should fuse dequant + matmul to
minimize memory traffic and kernel launches.

## Phased implementation

1) Reference path (Python)
   - Dequant Q, L, R on CPU/GPU
   - Unfused matmul: y0 = Qx, y1 = L(Rx), y = y0 + y1
   - Implemented in `src/q_lr_reference.py` with a small benchmark harness

1.5) Mojo fused CPU prototype
   - Implements groupwise quantization + fused Q + LR compute in Mojo
   - File: `kernels/q_lr_fused.mojo`
   - Run: `mojo kernels/q_lr_fused.mojo`

2) Fused GPU kernel (Mojo/MAX)
   - Tile over output rows for Q and L
   - Dequant in-register for each tile
   - Accumulate Qx and L(Rx) in a single kernel

3) Quant scheme variants
   - Start with groupwise int4 for L/R
   - Keep Q at 2-bit or int4, depending on kernel complexity
   - Add support for E8 later if needed

## Benchmarks

- Compare fused vs unfused throughput and memory bandwidth
- Validate numerical parity within tolerance
- Measure end-to-end tok/s at batch=1, seq=1 and seq=128
